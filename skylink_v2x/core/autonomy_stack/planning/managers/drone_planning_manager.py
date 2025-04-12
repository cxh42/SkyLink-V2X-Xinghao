# -*- coding: utf-8 -*-
"""
Drone planning manager module
"""

import math
import numpy as np
import carla

from skylink_v2x.core.autonomy_stack.planning.core.collision_detection import CollisionDetector
from skylink_v2x.core.autonomy_stack.planning.core.trajectory_generation import LocalTrajectoryPlanner
from skylink_v2x.core.autonomy_stack.planning.core.route_planning import GlobalRoutePlanner
from skylink_v2x.core.autonomy_stack.planning.core.route_planning import GlobalRouteDatabase
from skylink_v2x.core.autonomy_stack.planning.visualization.debug_helper import PlanningDebugHelper
from skylink_v2x.core.autonomy_stack.planning.managers.base_manager import BasePlanningManager
from skylink_v2x.core.autonomy_stack.planning.core.common_utils import (
    get_vehicle_speed, compute_distance, distance_to_vehicle
)

class DroneMode:
    """Enumeration of drone operation modes"""
    HOVER = "hover"
    FOLLOW_VEHICLE = "follow_vehicle"
    FOLLOW_TRAJECTORY = "follow_trajectory"

class DronePlanningManager(BasePlanningManager):
    """
    Planning manager for autonomous drones.
    
    This class handles behavior planning and execution for drones with
    three operation modes: hover, follow_vehicle, and follow_trajectory.
    
    Parameters
    ----------
    drone : carla.Vehicle
        The drone vehicle object
    carla_map : carla.Map
        Map of the simulation world
    config : dict
        Configuration dictionary
    """
    
    def __init__(self, drone, carla_map, config):
        super().__init__(config)
        self.drone = drone
        self._map = carla_map
        
        # Position and speed
        self._drone_pos = None
        self._drone_speed = 0.0
        
        # Mode configuration
        self.mode = DroneMode.HOVER
        self.hover_location = None
        self.follow_vehicle = None
        self.follow_height = config.get('follow_height', 20.0)  # meters above target
        
        # Planning components
        self._global_planner = None
        self._local_planner = LocalTrajectoryPlanner(self, carla_map, config['local_planner'])
        self._sampling_resolution = config.get('sample_resolution', 1.0)
        
        # Collision detection
        self._collision_detector = CollisionDetector(
            time_ahead=config.get('collision_time_ahead', 1.2)
        )
        
        # Speed limits
        self.max_speed = config.get('max_speed', 30.0)
        self.hover_precision = config.get('hover_precision', 2.0)
        
        # Perception
        self.objects = {}
        
        # Debug
        self.debug_helper = PlanningDebugHelper(self.drone.id)
        self.debug = config.get('debug', False)
        
    def update_information(self, position, speed, objects):
        """
        Update drone with latest information
        
        Parameters
        ----------
        position : carla.Transform
            Current position of the drone
        speed : float
            Current speed in km/h
        objects : dict
            Dictionary of detected objects
        """
        self._drone_pos = position
        self._drone_speed = speed
        self.objects = objects if objects else {}
        
        # Update trajectory planner
        self._local_planner.update_information(position, speed)
        
        # Update debug helper
        self.debug_helper.update(speed, 1000)  # No TTC for drones by default
    
    def set_mode(self, mode, target=None):
        """
        Set the drone's operation mode
        
        Parameters
        ----------
        mode : DroneMode
            The desired operation mode
        target : object
            Target for the mode (location for hover, vehicle for follow)
        """
        if mode not in [DroneMode.HOVER, DroneMode.FOLLOW_VEHICLE, DroneMode.FOLLOW_TRAJECTORY]:
            print(f"Warning: Invalid drone mode {mode}, defaulting to HOVER")
            mode = DroneMode.HOVER
            
        self.mode = mode
        
        if mode == DroneMode.HOVER and target:
            if isinstance(target, carla.Location):
                self.hover_location = target
            else:
                print("Warning: Hover target should be a carla.Location")
                
        elif mode == DroneMode.FOLLOW_VEHICLE and target:
            if hasattr(target, 'get_location'):
                self.follow_vehicle = target
            else:
                print("Warning: Follow target should have get_location() method")
                
        elif mode == DroneMode.FOLLOW_TRAJECTORY and target:
            if isinstance(target, carla.Location):
                # Set destination for trajectory following
                self.set_destination(self._drone_pos.location, target)
            else:
                print("Warning: Trajectory target should be a carla.Location")
    
    def set_destination(self, start_location, end_location, clean=False, end_reset=True, clean_history=False):
        """
        Set the destination for trajectory following
        
        Parameters
        ----------
        start_location : carla.Location
            Starting location
        end_location : carla.Location
            Destination location
        clean : bool
            Whether to clean existing waypoints
        end_reset : bool
            Whether to reset the end waypoint
        clean_history : bool
            Whether to clean waypoint history
        """
        if self.mode != DroneMode.FOLLOW_TRAJECTORY:
            print("Warning: Setting destination while not in FOLLOW_TRAJECTORY mode")
            
        # Initialize global planner if needed
        if self._global_planner is None:
            world = self.drone.get_world()
            database = GlobalRouteDatabase(
                world.get_map(), 
                sampling_resolution=self._sampling_resolution
            )
            planner = GlobalRoutePlanner(database)
            planner.setup()
            self._global_planner = planner
            
        # Find start and end waypoints
        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)
        
        # Trace route
        route_trace = self._global_planner.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location
        )
            
        # Set the global plan in local planner
        self._local_planner.set_global_plan(route_trace, clean)
    
    def calculate_hover_control(self):
        """
        Calculate control for hover mode
        
        Returns
        -------
        tuple
            (target_speed, target_location)
        """
        if not self.hover_location:
            # Default to current position if no hover point specified
            self.hover_location = self._drone_pos.location
            
        # Calculate 3D distance to hover point
        current_loc = self._drone_pos.location
        dx = self.hover_location.x - current_loc.x
        dy = self.hover_location.y - current_loc.y
        dz = self.hover_location.z - current_loc.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < self.hover_precision:
            # Close enough to hover point, maintain position
            return 0, self.hover_location
        else:
            # Move towards hover point at speed proportional to distance
            target_speed = min(self.max_speed, distance * 2)
            return target_speed, self.hover_location
    
    def calculate_follow_control(self):
        """
        Calculate control for vehicle following mode
        
        Returns
        -------
        tuple
            (target_speed, target_location)
        """
        if not self.follow_vehicle:
            # No vehicle to follow, hover in place
            return self.calculate_hover_control()
            
        try:
            # Get the target vehicle's position
            vehicle_pos = self.follow_vehicle.get_location()
            
            # Calculate target position above vehicle
            target_location = carla.Location(
                x=vehicle_pos.x,
                y=vehicle_pos.y,
                z=vehicle_pos.z + self.follow_height
            )
            
            # Calculate 3D distance to target position
            current_loc = self._drone_pos.location
            distance = math.sqrt(
                (target_location.x - current_loc.x)**2 +
                (target_location.y - current_loc.y)**2 +
                (target_location.z - current_loc.z)**2
            )
            
            # Adjust speed based on distance to target
            target_speed = min(
                self.max_speed,
                distance * 2.0  # Proportional to distance
            )
            
            return target_speed, target_location
            
        except Exception as e:
            print(f"Error following vehicle: {e}")
            return self.calculate_hover_control()
    
    def calculate_trajectory_control(self):
        """
        Calculate control for trajectory following mode
        
        Returns
        -------
        tuple
            (target_speed, target_location)
        """
        try:
            # Generate path using local planner
            path_x, path_y, path_curvature, path_yaw = self._local_planner.generate_path()
            
            if not path_x or not path_y:
                # No valid path, hover in place
                print("Warning: No valid trajectory path")
                return self.calculate_hover_control()
            
            # Calculate target speed and location using local planner
            target_speed, target_loc = self._local_planner.run_step(
                path_x, path_y, path_curvature, 
                target_speed=self.max_speed
            )
            
            return target_speed, target_loc
            
        except Exception as e:
            print(f"Error following trajectory: {e}")
            return self.calculate_hover_control()
    
    def run_step(self, target_speed=None):
        """
        Execute one planning step based on the current mode
        
        Parameters
        ----------
        target_speed : float, optional
            Override target speed if specified
            
        Returns
        -------
        tuple
            (target_speed, target_location)
        """
        # Handle different modes
        if self.mode == DroneMode.HOVER:
            return self.calculate_hover_control()
            
        elif self.mode == DroneMode.FOLLOW_VEHICLE:
            return self.calculate_follow_control()
            
        elif self.mode == DroneMode.FOLLOW_TRAJECTORY:
            return self.calculate_trajectory_control()
            
        # Default: hover mode
        return self.calculate_hover_control()