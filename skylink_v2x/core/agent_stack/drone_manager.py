# Author: Fengze Yang <fred.yang@utah.edu>
# License:


import uuid
from .dynamic_manager import DynamicManager
# from skylink_v2x.core.autonomy_stack.planning.DronePlanner import DronePlanner
from skylink_v2x.core.autonomy_stack.perception.perception_manager import DronePerceptionManager
# from skylink_v2x.core.autonomy_stack.control.VehicleController import VehicleController
from skylink_v2x.core.autonomy_stack.mapping.map_manager import MapManager
from skylink_v2x.core.autonomy_stack.localization.localization_manager import DroneLocalizationManager
# from skylink_v2x.core.communication_manager.communication_manager import CommunicationManager
from skylink_v2x.core.autonomy_stack.plan.drone_planner import DronePlanner
from skylink_v2x.core.autonomy_stack.actuation.control_manager import DroneControlManager
from skylink_v2x.utils import read_yaml
import airsim
import time
from omegaconf import OmegaConf
import inspect


class DroneManager(DynamicManager):
    """
    Class for managing a dynamic drone agent.
    Update agent manager in SimScenario after initialization.
    
    Attributes
    ----------
    drone : Airsim Client
        The Airsim client managed by this manager.
    config : dict
        The configuration parameters for drone planning.
    is_log : bool
        Indicator whether logging is enabled.
    planner : DronePlanner
        The planner that computes the drone's target transform or trajectory.
    target_transform : object or list
        The target transform (or trajectory) computed by the planner.
    """
    def __init__(self,
                 airsim_client: airsim.MultirotorClient, 
                 drone_name: str,
                 drone,
                 mode: str,
                 carla_map,
                 config,
                 comm_manager=None,
                 attached_vehicle=None):
        
        super().__init__(drone.id, drone.get_world(), config)  # TODO: make sure how to get the Drone ID from AirSim.
        self.airsim_client = airsim_client
        self.type = "drone"
        self.drone = drone
        self.mode = mode
        self.attached_vehicle = attached_vehicle
        self.drone_name = drone_name
        self.airsim_client.enableApiControl(True, vehicle_name=drone_name)
        self.airsim_client.armDisarm(True, vehicle_name=drone_name)

        self.airsim_client.simPause(False)
        self.airsim_client.takeoffAsync(vehicle_name=drone_name).join()
        self.airsim_client.simPause(True)
        
        

        # Retrieve the configure for different modules
        perception_config = read_yaml(config['perception']['base'])
        localization_config = read_yaml(config['localization']['base'])
        map_config = read_yaml(config['mapping']['base'])
        # planner_config = read_yaml(config['planning']['base'])
        # control_config = read_yaml(config['control']['base'])
        planner_config = OmegaConf.merge(read_yaml(config['planning']['base']), config.get('planning_config', {}))
        control_config = OmegaConf.merge(read_yaml(config['control']['base']), config.get('control_config', {}))
        assert self.mode == planner_config['mode'], "The mode in config and the mode in VehicleManager should be the same."

        self.perception_manager = DronePerceptionManager(carla_world=self.world,
                                                           agent_id=self.agent_id,
                                                           config=perception_config,
                                                           comm_manager=comm_manager)
        
        self.localizer = DroneLocalizationManager(drone, 
                                                    localization_config, 
                                                    comm_manager=comm_manager)
        
        self.map_manager = MapManager(carla_world=self.world,
                                      agent_id=self.agent_id, 
                                      carla_map=carla_map, 
                                      config=map_config,
                                      comm_manager=comm_manager)
        
        self.planner = DronePlanner(drone=drone, 
                                      config_yaml=planner_config,
                                      comm_manager=comm_manager,
                                      attached_vehicle=self.attached_vehicle)
        
        self.controller = DroneControlManager(control_config=control_config, 
                                              airsim_client=airsim_client,
                                              drone_name=drone_name)
        
        # # TODO: Make sure following functions' inputs.
        # self.controller = VehicleController()
        
        # # Instantiate the DronePlanner with the drone actor and configuration.
        # self.planner = DronePlanner(drone, planner_config)
        
        comm_manager.register(self)

        

    def set_destination(self, 
                        start_location, 
                        end_location, 
                        clean=False, 
                        end_reset=True) -> None:
        """
        Set the destination for the dynamic vehicle.
        
        Parameters
        ----------
        start_location : object
            The starting location (could be a tuple, object, etc.).
        end_location : object
            The destination location.
        clean : bool
            Indicator to clear the existing waypoint queue.
        end_reset : bool
            Indicator to reset the endpoint.
        """
        # print(f"DroneManager {self.agent_id}: setting destination from {start_location}\
        #        to {end_location}, clean: {clean}, end_reset: {end_reset}")
        
        # Delegate destination setting to the planner.
        # self.planner.set_destination(start_location, end_location, clean, end_reset)  


    def update_localization(self):
        """
        Localize the vehicle agent.
        """
        return self.localizer.localize()
    
    
    def update_perception(self):
        """
        Detect objects in the environment.
        """
        return self.perception_manager.detect_objects()
    
    
    def update_mapping(self,
                       localization_data,
                       perception_data, # object detection data, etc.
                       sensing_data=None  # Raw sensor data including LIDAR, camera, etc.
                       ):
        """
        Update the map information.
        """
        self.map_manager.update_info(center=localization_data[self.agent_id]['position'])
        return self.map_manager.get_current_map(localization_data=localization_data,
                                                perception_data=perception_data,
                                                sensing_data=sensing_data)
    
    def update_planning(self, 
                        localization_data=None, 
                        perception_data=None,
                        mapping_data=None,
                        target_speed=None):
        """
        Update the planning module.
        """
        self.planner.update_information(ego_pos=localization_data[self.agent_id]['position'])
        self.planner.run_step()

    def update_control(self, 
                       localization_data=None, 
                       planning_outputs=None):
        """
        Update the control module.
        """
        control_signal = self.controller.run_step(*planning_outputs[self.agent_id])
        return control_signal
    
    
    def remove(self) -> None:
        """
        清理资源，添加异常处理
        """
        print(f"{self.__class__.__name__} {self.agent_id}: cleaning up resources")
        
        # 清理感知管理器
        try:
            if hasattr(self, 'perception_manager') and self.perception_manager:
                self.perception_manager.destroy()
        except Exception as e:
            print(f"清理感知管理器失败: {str(e)}")
        
        # 清理地图管理器
        try:
            if hasattr(self, 'map_manager') and self.map_manager:
                if hasattr(self.map_manager, 'remove'):
                    self.map_manager.remove()
                else:
                    # 尝试手动清理地图资源
                    if hasattr(self.map_manager, '_map_surface'):
                        self.map_manager._map_surface = None
        except Exception as e:
            print(f"清理地图管理器失败: {str(e)}")
        
        # 清理定位管理器
        try:
            if hasattr(self, 'localizer') and self.localizer:
                if hasattr(self.localizer, 'remove'):
                    self.localizer.remove()
                else:
                    print(f"定位管理器没有remove方法")
        except Exception as e:
            print(f"清理定位管理器失败: {str(e)}")
        
        # 清理车辆/无人机
        try:
            if hasattr(self, 'vehicle') and self.vehicle:
                self.vehicle.destroy()
        except Exception as e:
            print(f"清理车辆失败: {str(e)}")
        
        # 如果是无人机还需要清理AirSim资源
        if hasattr(self, 'airsim_client') and hasattr(self, 'drone_name'):
            try:
                self.airsim_client.landAsync(vehicle_name=self.drone_name).join()
                self.airsim_client.armDisarm(False, vehicle_name=self.drone_name)
                self.airsim_client.enableApiControl(False, vehicle_name=self.drone_name)
            except Exception as e:
                print(f"清理AirSim资源失败: {str(e)}")