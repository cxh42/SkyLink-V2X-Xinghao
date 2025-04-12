# -*- coding: utf-8 -*-
"""
Scenario Manager Module

This module sets up the CARLA simulation and integrates drone control using AirSim.
It supports three scenarios:

1. Tracking (default): A vehicle is spawned and the drone takes off nearby,
   aligns with the vehicle (using yaw = vehicle_yaw - 90) and continuously tracks it.
2. Navigation: Two random waypoints are chosen (origin and destination). The drone
   spawns at the origin, takes off, and navigates directly to the destination.
3. Monitoring: The drone is spawned at a waypoint, takes off, and hovers while
   monitoring its state.

Author: Keshu Wu <keshuw@tamu.edu>
License: TDG-Attribution-NonCommercial-NoDistrib
"""

import math
import random
import sys
import os
import time
import cv2
import numpy as np
import carla
import airsim
import logging
from typing import Optional

# Import the DroneManager and utility functions from drone_manager.py
from drone_manager import DroneManager, convert_carla_to_airsim, convert_airsim_to_carla

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScenarioManager:
    """
    Manages the overall CARLA simulation and drone scenarios.
    
    Supports multiple scenarios via the scenario_params dictionary:
      - "tracking": Spawn a vehicle and have the drone track it.
      - "navigation": Use two random waypoints as origin and destination for direct navigation.
      - "monitoring": Spawn the drone, take off, and hover while monitoring its state.
      
    The default scenario is "tracking".
    """
    def __init__(self, scenario_params: dict, apply_ml: bool, carla_version: str,
                 xodr_path: Optional[str] = None, town: Optional[str] = None,
                 cav_world: Optional[object] = None, comm_manager: Optional[object] = None) -> None:
        self.scenario_params = scenario_params
        self.carla_version = carla_version

        simulation_config = scenario_params['world']
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])

        self.client = carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)

        # Initialize AirSim client for drone control.
        self.airsim_client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.airsim_client.confirmConnection()
        self.airsim_client.enableApiControl(True)
        self.airsim_client.armDisarm(True)

        if xodr_path:
            # Assume load_customized_world is defined elsewhere.
            self.world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                self.world = self.client.load_world(town)
            except RuntimeError:
                sys.exit(f"Town {town} is not found in your CARLA repo!")
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        # self.origin_settings = self.world.get_settings()
        # new_settings = self.world.get_settings()
        # if simulation_config['sync_mode']:
        #     new_settings.synchronous_mode = True
        #     new_settings.fixed_delta_seconds = simulation_config['fixed_delta_seconds']
        # else:
        #     sys.exit('ERROR: Only synchronous simulation mode is supported.')
        # self.world.apply_settings(new_settings)

        if 'weather' in simulation_config:
            self.world.set_weather(carla.WeatherParameters(**simulation_config['weather']))

        self.carla_map = self.world.get_map()
        self.apply_ml = apply_ml
        self.comm_manager = comm_manager
        self.cav_world = cav_world

        # Drone parameters.
        self.DRONE_TAKEOFF_HEIGHT: float = 60  # m
        self.DRONE_HOVER_OFFSET: float = self.DRONE_TAKEOFF_HEIGHT
        self.DRONE_SPEED: float = 6            # m/s
        self.DRONE_GROUND_SHIFT: float = 5.0     # m

        self.TOP_DOWN_OFFSET: float = 5.0        # m
        self.UPDATE_INTERVAL: float = 0.033      # seconds

    def run_drone(self) -> None:
        """
        Runs the desired drone scenario based on scenario_params.
        
        "tracking": Spawn a vehicle and track it.
        "navigation": Select two random waypoints as origin and destination and navigate.
        "monitoring": Spawn the drone and have it take off and hover.
        Default scenario is "tracking".
        """
        scenario_type = self.scenario_params.get("scenario_type", "tracking")
        blueprint_library = self.world.get_blueprint_library()

        if scenario_type == "tracking":
            # Spawn vehicle.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available in the CARLA map!")
            vehicle_spawn_point = random.choice(spawn_points)
            vehicle_bp = blueprint_library.filter('vehicle.*')[0]
            vehicle = self.world.spawn_actor(vehicle_bp, vehicle_spawn_point)
            print(f"Spawned CARLA vehicle at spawn point: {vehicle_spawn_point}")

            # Compute nearby drone spawn location.
            drone_spawn_location = vehicle_spawn_point.location - carla.Location(
                x=self.DRONE_GROUND_SHIFT, y=self.DRONE_GROUND_SHIFT, z=0)

            # Initialize drone visualizer and AirSim pose.
            drone_actor, camera_sensor = self.initialize_drone(blueprint_library, drone_spawn_location)

            # Create DroneManager instance.
            drone_manager = DroneManager(self.airsim_client, drone_actor, camera_sensor, self.world,
                                         self.UPDATE_INTERVAL, self.TOP_DOWN_OFFSET, drone_speed=self.DRONE_SPEED)
            drone_manager.set_camera_callback()

            # Set spectator view.
            initial_loc = drone_actor.get_transform().location
            spectator = self.world.get_spectator()
            top_down_loc = carla.Location(x=initial_loc.x, y=initial_loc.y, z=initial_loc.z + self.TOP_DOWN_OFFSET)
            top_down_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator.set_transform(carla.Transform(top_down_loc, top_down_rot))
            print("Set CARLA spectator to top-down view.")

            # Define takeoff target using the vehicle's spawn point.
            takeoff_target = airsim.Vector3r(
                convert_carla_to_airsim(vehicle_spawn_point.location, 0).x_val,
                convert_carla_to_airsim(vehicle_spawn_point.location, 0).y_val,
                -self.DRONE_TAKEOFF_HEIGHT
            )

            # Run the tracking scenario.
            drone_manager.run(vehicle, takeoff_target, self.DRONE_HOVER_OFFSET, scenario="tracking")

        elif scenario_type == "navigation":
            # For navigation, no vehicle is spawned.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available in the CARLA map!")
            origin_transform, destination_transform = random.sample(spawn_points, 2)
            print(f"Selected origin: {origin_transform} and destination: {destination_transform}")

            # Spawn drone at the origin.
            drone_actor, camera_sensor = self.initialize_drone(blueprint_library, origin_transform.location)

            # Create DroneManager instance.
            drone_manager = DroneManager(self.airsim_client, drone_actor, camera_sensor, self.world,
                                         self.UPDATE_INTERVAL, self.TOP_DOWN_OFFSET, drone_speed=self.DRONE_SPEED)
            drone_manager.set_camera_callback()

            # Set spectator view.
            initial_loc = drone_actor.get_transform().location
            spectator = self.world.get_spectator()
            top_down_loc = carla.Location(x=initial_loc.x, y=initial_loc.y, z=initial_loc.z + self.TOP_DOWN_OFFSET)
            top_down_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator.set_transform(carla.Transform(top_down_loc, top_down_rot))
            print("Set CARLA spectator to top-down view.")

            # Define takeoff target using the origin's location.
            takeoff_target = airsim.Vector3r(
                convert_carla_to_airsim(origin_transform.location, 0).x_val,
                convert_carla_to_airsim(origin_transform.location, 0).y_val,
                -self.DRONE_TAKEOFF_HEIGHT
            )

            # Command takeoff and ascend, then navigate to destination.
            drone_manager.takeoff_and_ascend(takeoff_target, None)
            destination_airsim = convert_carla_to_airsim(destination_transform.location, self.DRONE_HOVER_OFFSET)
            drone_manager.direct_navigation(destination_airsim)
            logging.info("Landing drone after navigation...")
            self.airsim_client.landAsync().join()

        elif scenario_type == "monitoring":
            # For monitoring, no vehicle is spawned.
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available in the CARLA map!")
            monitor_spawn_point = random.choice(spawn_points)
            print(f"Selected monitoring spawn point: {monitor_spawn_point}")

            # Spawn drone at the monitoring location.
            drone_actor, camera_sensor = self.initialize_drone(blueprint_library, monitor_spawn_point.location)

            # Create DroneManager instance.
            drone_manager = DroneManager(self.airsim_client, drone_actor, camera_sensor, self.world,
                                         self.UPDATE_INTERVAL, self.TOP_DOWN_OFFSET, drone_speed=self.DRONE_SPEED)
            drone_manager.set_camera_callback()

            # Set spectator view.
            initial_loc = drone_actor.get_transform().location
            spectator = self.world.get_spectator()
            top_down_loc = carla.Location(x=initial_loc.x, y=initial_loc.y, z=initial_loc.z + self.TOP_DOWN_OFFSET)
            top_down_rot = carla.Rotation(pitch=-90, yaw=90, roll=0)
            spectator.set_transform(carla.Transform(top_down_loc, top_down_rot))
            print("Set CARLA spectator to top-down view.")

            # Define takeoff target using the monitoring spawn point.
            takeoff_target = airsim.Vector3r(
                convert_carla_to_airsim(monitor_spawn_point.location, 0).x_val,
                convert_carla_to_airsim(monitor_spawn_point.location, 0).y_val,
                -self.DRONE_TAKEOFF_HEIGHT
            )

            # Run the monitoring scenario.
            drone_manager.run(vehicle=None, takeoff_target=takeoff_target,
                              hover_offset=self.DRONE_HOVER_OFFSET, scenario="monitoring")
        else:
            print(f"Unknown scenario type: {scenario_type}")

    def initialize_drone(self, blueprint_library: carla.BlueprintLibrary, drone_spawn_location: carla.Location) -> tuple:
        """
        Initializes the drone visualizer and sets the AirSim pose.
        
        Returns:
            tuple: (drone_actor, camera_sensor)
        """
        return self.initialize_drone_on_ground(self.airsim_client, blueprint_library, drone_spawn_location)

    def initialize_drone_on_ground(self, airsim_client: airsim.MultirotorClient, blueprint_library: carla.BlueprintLibrary,
                                   drone_spawn_location: carla.Location) -> tuple:
        """
        Spawns the CARLA drone visualizer and sets the corresponding AirSim pose.
        
        Returns:
            tuple: (drone_actor, camera_sensor)
        """
        drone_transform = carla.Transform(drone_spawn_location, carla.Rotation(pitch=0, yaw=0, roll=0))
        drone_blueprint = blueprint_library.find('static.prop.shoppingcart')
        # drone_blueprint = blueprint_library.find('sensor.other.collision')
        drone_actor = self.world.spawn_actor(drone_blueprint, drone_transform)
        print(f"Spawned CARLA drone visualizer at location: {drone_spawn_location}")

        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "480")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(
            carla.Location(x=0, y=0, z=-1),
            carla.Rotation(pitch=-90, yaw=90, roll=0)
        )
        camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=drone_actor)
        print("Attached camera sensor to drone visualizer.")

        airsim_ground_position = convert_carla_to_airsim(drone_spawn_location, hover_offset=0)
        drone_initial_orientation = airsim.to_quaternion(0, 0, 0)
        drone_initial_pose = airsim.Pose(airsim_ground_position, drone_initial_orientation)
        airsim_client.simSetVehiclePose(drone_initial_pose, True)
        print(f"Initialized AirSim drone at position: {airsim_ground_position}")

        return drone_actor, camera_sensor

    def destroy_actors(self) -> None:
        """
        Destroys all actors in the CARLA world.
        """
        for actor in self.world.get_actors():
            actor.destroy()

    def close(self) -> None:
        """
        Restores the original CARLA world settings.
        """
        self.world.apply_settings(self.origin_settings)


if __name__ == "__main__":
    scenario_params = {
        'world': {
            'client_port': 2000,
            'sync_mode': False,
            'fixed_delta_seconds': 0.033,
            'weather': {
                'sun_altitude_angle': 70,
                'cloudiness': 10,
                'precipitation': 0,
                'precipitation_deposits': 0,
                'wind_intensity': 0,
                'fog_density': 0,
                'fog_distance': 0,
                'fog_falloff': 0,
                'wetness': 0
            },
            'seed': 42
        },
        # Set "scenario_type" to "tracking", "navigation", or "monitoring"
        "scenario_type": "navigation"
    }
    sm = ScenarioManager(scenario_params, apply_ml=False, carla_version='0.9.15')
    sm.run_drone()
