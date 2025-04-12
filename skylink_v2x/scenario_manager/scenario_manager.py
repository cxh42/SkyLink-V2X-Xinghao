# -*- coding: utf-8 -*-
"""
Utilize scenario manager to manage CARLA simulation construction. This script
is used for carla simulation only, and if you want to manage the Co-simulation,
please use cosim_api.py.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
import random
import sys
import json
import time
from random import shuffle
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from skylink_v2x.carla_helper import OBJECT_CONFIG

import carla
import airsim
import numpy as np

from skylink_v2x.core.agent_stack.vehicle_manager import VehicleManager
from skylink_v2x.core.agent_stack.drone_manager import DroneManager
from skylink_v2x.core.agent_stack.rsu_manager import RSUManager
# from opencda.core.common.cav_world import CavWorld
from skylink_v2x.core.communication_manager.communication_manager import CommunicationManager
from skylink_v2x.utils import \
    convert_carla_to_airsim, convert_airsim_to_carla, convert_airsim_quaternion_to_carla_rotation, load_customized_world


class ScenarioManager:
    """
    The manager that controls simulation construction, backgound traffic
    generation and CAVs spawning.

    Parameters
    ----------
    scenario_params : dict
        The dictionary contains all simulation configurations.

    carla_version : str
        CARLA simulator version, it currently supports 0.9.11 and 0.9.12

    xodr_path : str
        The xodr file to the customized map, default: None.

    town : str
        Town name if not using customized map, eg. 'Town06'.

    apply_ml : bool
        Whether need to load dl/ml model(pytorch required) in this simulation.

    Attributes
    ----------
    client : carla.client
        The client that connects to carla server.

    world : carla.world
        Carla simulation server.

    origin_settings : dict
        The origin setting of the simulation server.

    cav_world : opencda object
        CAV World that contains the information of all CAVs.

    carla_map : carla.map
        Car;a HD Map.

    """

    def __init__(self, scenario_params,
                 apply_ml,
                 carla_version,
                 xodr_path=None,
                 cav_world=None):
        
        self.timestamp = 0
        town = scenario_params['world']['town']
        self.scenario_params = scenario_params
        self.carla_version = carla_version

        simulation_config = scenario_params['world']

        # set random seed if stated
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])

        self.client = \
            carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)
        if xodr_path:
            self.carla_world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                self.carla_world = self.client.load_world(town)
            except RuntimeError:
                print(
                    f"%s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!" % town)
        else:
            self.carla_world = self.client.get_world()

        if not self.carla_world:
            sys.exit('World loading failed')

        self.origin_settings = self.carla_world.get_settings()
        new_settings = self.carla_world.get_settings()
        self.carla_world_delta_seconds = simulation_config['fixed_delta_seconds']
        if simulation_config['sync_mode']:
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = self.carla_world_delta_seconds
        else:
            sys.exit(
                'ERROR: Current version only supports sync simulation mode')

        self.carla_world.apply_settings(new_settings)

        # set weather
        weather = self.set_weather(simulation_config['weather'])
        self.carla_world.set_weather(weather)

        # Define probabilities for each type of blueprint
        self.use_multi_class_bp = scenario_params["blueprint"][
            'use_multi_class_bp'] if 'blueprint' in scenario_params else False
        if self.use_multi_class_bp:
            # bbx/blueprint meta
            with open(scenario_params['blueprint']['bp_meta_path']) as f:
                self.bp_meta = json.load(f)
            self.bp_class_sample_prob = scenario_params['blueprint'][
                'bp_class_sample_prob']

            # normalize probability
            self.bp_class_sample_prob = {
                k: v / sum(self.bp_class_sample_prob.values()) for k, v in
                self.bp_class_sample_prob.items()}

        self.cav_world = cav_world
        self.carla_map = self.carla_world.get_map()
        self.apply_ml = apply_ml
        self.comm_manager = CommunicationManager(scenario_params['communication'])
        
        self.init_drone()
        
    def init_drone(self):
        # Initialize AirSim client for drone control.
        self.airsim_client = airsim.MultirotorClient(ip="127.0.0.1", port=41451)
        self.airsim_client.confirmConnection()
        self.airsim_client.reset()
        self.airsim_client.simPause(True)
        # self.DRONE_GROUND_SHIFT: float = 5.0     # meters TODO: make this to a config

    @staticmethod
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather
    
    def create_agent_managers(self):
        self.agent_dict = dict()
        for i, agent_config in enumerate(self.scenario_params['scenario']['agent_list']):
            if agent_config.type == 'vehicle':
                agent_config = OmegaConf.merge(self.scenario_params['agent_base']['vehicle'], agent_config)
                agent_manager = self.create_vehicle_manager(agent_config)
            elif agent_config.type == 'rsu':
                agent_config = OmegaConf.merge(self.scenario_params['agent_base']['rsu'], agent_config)
                agent_manager = self.create_rsu_manager(agent_config)
                pass
            elif agent_config.type == 'drone':
                agent_config = OmegaConf.merge(self.scenario_params['agent_base']['drone'], agent_config)
                agent_manager = self.create_drone_manager(agent_config)
            else:
                raise ValueError('Unknown agent type. Currently only support vehicle, rsu and drone.')
            if agent_config.name in self.agent_dict:
                raise ValueError(f"Duplicated agent name: {agent_config.name}.")
            self.agent_dict[agent_config.name] = agent_manager
        return self.agent_dict
            
    def create_vehicle_manager(self, agent_config):
        
        location = carla.Location(
                    x=agent_config['spawn_position'][0],
                    y=agent_config['spawn_position'][1],
                    z=agent_config['spawn_position'][2])
        way_point = self.carla_map.get_waypoint(location).transform
        rotation = carla.Rotation(
            way_point.rotation.pitch,
            way_point.rotation.yaw,
            way_point.rotation.roll)
        spawn_transform = carla.Transform(location, rotation)
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        bp = self.carla_world.get_blueprint_library().find(default_model)
        vehicle = self.carla_world.spawn_actor(bp, spawn_transform)
        vehicle_manager = VehicleManager(
            vehicle = vehicle,
            carla_map = self.carla_map,
            config = agent_config,
            comm_manager = self.comm_manager)
        self.carla_world.tick()
        destination = carla.Location(x=agent_config['destination'][0],
                                         y=agent_config['destination'][1],
                                         z=agent_config['destination'][2])
        vehicle_manager.set_destination(
            vehicle_manager.vehicle.get_location(),
            destination,
            clean=True)
        return vehicle_manager
    
    def create_drone_manager(self, agent_config):
        mode = agent_config['planning_config']['mode']
        drone_blueprint = self.carla_world.get_blueprint_library().find('static.prop.shoppingcart') # TODO: Replace with actual drone blueprint
        if mode == 'escort':
            attached_vehicle = self.agent_dict[agent_config['attached_vehicle']].vehicle
        else:
            attached_vehicle = None
        spawn_transform = carla.Transform(
            carla.Location(
                x=agent_config['spawn_position'][0],
                y=agent_config['spawn_position'][1],
                z=agent_config['spawn_position'][2]),
            carla.Rotation(
                pitch=agent_config['spawn_position'][5],
                yaw=agent_config['spawn_position'][4],
                roll=agent_config['spawn_position'][3]))
        drone = self.carla_world.spawn_actor(drone_blueprint, spawn_transform)
        pose = airsim.Pose(convert_carla_to_airsim(spawn_transform.location),
                   airsim.to_quaternion(
                       math.radians(spawn_transform.rotation.roll),
                       math.radians(spawn_transform.rotation.pitch),
                       math.radians(spawn_transform.rotation.yaw)))
        self.airsim_client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=agent_config['name'])
        drone_manager = DroneManager(
            airsim_client = self.airsim_client,
            drone_name = agent_config['name'],  
            drone = drone,
            mode = mode,
            carla_map = self.carla_map,
            config = agent_config,
            comm_manager = self.comm_manager,
            attached_vehicle=attached_vehicle)
        self.carla_world.tick()
        return drone_manager
    
    
    def create_rsu_manager(self, agent_config):
        # Parse spawn position from config
        spawn_location = carla.Location(
            x=agent_config['spawn_position'][0],
            y=agent_config['spawn_position'][1],
            z=agent_config['spawn_position'][2])
        
        # Find the closest traffic light to the specified spawn position
        all_traffic_lights = self.carla_world.get_actors().filter('traffic.traffic_light')
        
        closest_traffic_light = None
        min_distance = float('inf')
        
        for traffic_light in all_traffic_lights:
            distance = spawn_location.distance(traffic_light.get_location())
            if distance < min_distance:
                min_distance = distance
                closest_traffic_light = traffic_light
        
        # Check if any traffic light was found
        if closest_traffic_light is None:
            print(f"ERROR: No traffic light found for RSU {agent_config['name']}.")
            return None
        
        # Warning if the closest traffic light is more than 20m away
        if min_distance > 20.0:
            print(f"WARNING: Closest traffic light for RSU {agent_config['name']} is {min_distance:.2f}m away from specified spawn point.")
        
        # Create RSU manager with the closest traffic light
        rsu_manager = RSUManager(
            rsu=closest_traffic_light,
            carla_map=self.carla_map,
            config=agent_config,
            comm_manager=self.comm_manager)
        
        self.carla_world.tick()
        return rsu_manager

        

    def spawn_vehicles_by_list(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given list.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """
        blueprint_library = self.carla_world.get_blueprint_library()
        if self.use_multi_class_bp:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]
        else:
            ego_vehicle_random_list = [] # list of carla blueprint objects
            for bp_config in OBJECT_CONFIG.get_all_vehicles():
                ego_vehicle_random_list.extend([blueprint_library.find(bp) for bp in bp_config.blueprints])

        for i, vehicle_config in enumerate(traffic_config['vehicle_list']):
            spawn_transform = carla.Transform(
                carla.Location(
                    x=vehicle_config['spawn_position'][0],
                    y=vehicle_config['spawn_position'][1],
                    z=vehicle_config['spawn_position'][2]),
                carla.Rotation(
                    pitch=vehicle_config['spawn_position'][5],
                    yaw=vehicle_config['spawn_position'][4],
                    roll=vehicle_config['spawn_position'][3]))

            if self.use_multi_class_bp:
                label = np.random.choice(label_list, p=prob)
                ego_vehicle_random_list = [blueprint_library.find(bp) for bp in OBJECT_CONFIG.get_by_name(label).blueprints]
                
            ego_vehicle_bp = random.choice(ego_vehicle_random_list)

            if ego_vehicle_bp.has_attribute("color"):
                color = random.choice(ego_vehicle_bp.get_attribute('color').recommended_values)
                ego_vehicle_bp.set_attribute('color', color)
            
            vehicle = self.carla_world.spawn_actor(ego_vehicle_bp, spawn_transform)
            vehicle.set_autopilot(True, 8000)

            if 'vehicle_speed_perc' in vehicle_config:
                tm.vehicle_percentage_speed_difference(
                    vehicle, vehicle_config['vehicle_speed_perc'])
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            bg_list.append(vehicle)
        return bg_list

    def spawn_vehicle_by_range(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """
        blueprint_library = self.carla_world.get_blueprint_library()
        if self.use_multi_class_bp:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]
        else:
            ego_vehicle_random_list = [] # list of carla blueprint objects
            for bp_config in OBJECT_CONFIG.get_all_vehicles():
                ego_vehicle_random_list.extend([blueprint_library.find(bp) for bp in bp_config.blueprints])

        spawn_ranges = traffic_config['range']
        spawn_set = set()
        spawn_num = 0

        for spawn_range in spawn_ranges:
            spawn_num += spawn_range[6]
            x_min, x_max, y_min, y_max = \
                math.floor(spawn_range[0]), math.ceil(spawn_range[1]), \
                math.floor(spawn_range[2]), math.ceil(spawn_range[3])

            for x in range(x_min, x_max, int(spawn_range[4])):
                for y in range(y_min, y_max, int(spawn_range[5])):
                    location = carla.Location(x=x, y=y, z=0.3)
                    way_point = self.carla_map.get_waypoint(location).transform

                    spawn_set.add((way_point.location.x,
                                   way_point.location.y,
                                   way_point.location.z,
                                   way_point.rotation.roll,
                                   way_point.rotation.yaw,
                                   way_point.rotation.pitch))
                    
        count = 0
        spawn_list = list(spawn_set)
        shuffle(spawn_list)

        while count < spawn_num:
            if len(spawn_list) == 0:
                break

            coordinates = spawn_list[0]
            spawn_list.pop(0)

            spawn_transform = carla.Transform(carla.Location(x=coordinates[0],
                                                             y=coordinates[1],
                                                             z=coordinates[2] + 0.3),
                                              carla.Rotation(roll=coordinates[3],
                                                             yaw=coordinates[4],
                                                             pitch=coordinates[5]))

            if self.use_multi_class_bp:
                label = np.random.choice(label_list, p=prob)
                ego_vehicle_random_list = [blueprint_library.find(bp) for bp in OBJECT_CONFIG.get_by_name(label).blueprints]
                
            ego_vehicle_bp = random.choice(ego_vehicle_random_list)

            if ego_vehicle_bp.has_attribute("color"):
                color = random.choice(ego_vehicle_bp.get_attribute('color').recommended_values)
                ego_vehicle_bp.set_attribute('color', color)

            vehicle = self.carla_world.try_spawn_actor(ego_vehicle_bp, spawn_transform)

            if not vehicle:
                continue

            vehicle.set_autopilot(True, 8000)
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            if 'ignore_lights_percentage' in traffic_config:
                tm.ignore_lights_percentage(vehicle,
                                            traffic_config[
                                                'ignore_lights_percentage'])

            # each vehicle have slight different speed
            tm.vehicle_percentage_speed_difference(
                vehicle,
                traffic_config['global_speed_perc'] + random.randint(-30, 30))

            bg_list.append(vehicle)
            count += 1

        return bg_list

    def create_traffic_carla(self):
        """
        Create traffic flow.

        Returns
        -------
        tm : carla.traffic_manager
            Carla traffic manager.

        bg_list : list
            The list that contains all the background traffic vehicles.
        """
        print('Spawning CARLA traffic flow.')
        traffic_config = self.scenario_params['carla_traffic_manager']
        tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        bg_list = []

        if isinstance(traffic_config['vehicle_list'], list) or \
                isinstance(traffic_config['vehicle_list'], ListConfig):
            bg_list = self.spawn_vehicles_by_list(tm,
                                                  traffic_config,
                                                  bg_list)

        else:
            bg_list = self.spawn_vehicle_by_range(tm, traffic_config, bg_list)

        print('CARLA traffic flow generated.')
        return tm, bg_list

    # def tick(self):
    #     """
    #     Tick the server.
    #     """
    #     self.airsim_client.simContinueForTime(self.carla_world_delta_seconds)
    #     time.sleep(self.carla_world_delta_seconds)
    #     self.sync_drone_pos()
    #     self.carla_world.tick()
    
    def tick(self):
        DT = self.carla_world_delta_seconds

        self.airsim_client.simPause(False)  
        self.airsim_client.simContinueForTime(DT) 
        self.airsim_client.simPause(True)
        self.sync_drone_pos()
        
        self.carla_world.tick()
        
        # 每50帧执行一次垃圾回收
        if self.timestamp % 50 == 0:  # 使用self.timestamp而不是self._timestamp
            import gc
            gc.collect()
            
        self.timestamp += 1  # 更新计数器
        
        
    def sync_drone_pos(self):
        """
        Sync drone position with the carla world.
        """
        for agent in self.agent_dict.values():
            if agent.type == 'drone':
                drone_pos = self.airsim_client.getMultirotorState(vehicle_name=agent.drone_name).kinematics_estimated
                new_loc = convert_airsim_to_carla(drone_pos.position)
                new_rot = convert_airsim_quaternion_to_carla_rotation(drone_pos.orientation, force_bev=True)
                new_transform = carla.Transform(new_loc, new_rot)
                agent.drone.set_transform(new_transform)
                
        
    def run_step(self):
        """
        Run the simulation step.
        """
        for agent in self.agent_dict.values():
            agent.update_localization()
            agent.update_perception()
        self.comm_manager.sync_localization()
        self.comm_manager.sync_perception()
        localization_data = self.comm_manager.get_localization_data()
        perception_data = self.comm_manager.get_perception_data()
        
        for agent in self.agent_dict.values():
            agent.update_mapping(localization_data=localization_data, 
                                 perception_data=perception_data)
        self.comm_manager.sync_mapping()
        mapping_data = self.comm_manager.get_mapping_data()
        
        for agent in self.agent_dict.values():
            agent.update_planning(localization_data=localization_data, 
                                  perception_data=perception_data,
                                  mapping_data=mapping_data)
        self.comm_manager.sync_planning()
        planning_outputs = self.comm_manager.get_planning_data()
        for agent in self.agent_dict.values():
            control_signal = agent.update_control(localization_data=localization_data,
                                                  planning_outputs=planning_outputs)
        self.comm_manager.tick()

    def destroyActors(self):
        """
        Destroy all actors in the world.
        """

        actor_list = self.carla_world.get_actors()
        for actor in actor_list:
            actor.destroy()

    def close(self):
        """
        关闭仿真环境，清理资源
        """
        # 清理AirSim资源
        try:
            # 关闭所有无人机
            self.airsim_client.armDisarm(False)
            self.airsim_client.enableApiControl(False)
            self.airsim_client.reset()
        except:
            pass
        
        # 清理通信管理器
        try:
            if hasattr(self, 'comm_manager'):
                if hasattr(self.comm_manager, 'clear_resources'):
                    self.comm_manager.clear_resources()
        except:
            pass
        
        # 恢复原始设置
        try:
            self.carla_world.apply_settings(self.origin_settings)
        except:
            pass
        
        # 强制垃圾回收
        import gc
        gc.collect()
