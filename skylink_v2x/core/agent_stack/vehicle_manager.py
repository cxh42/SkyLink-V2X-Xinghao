# Author: Fengze Yang <fred.yang@utah.edu>
# License:


from .dynamic_manager import DynamicManager
from skylink_v2x.core.autonomy_stack.perception.perception_manager import VehiclePerceptionManager
from skylink_v2x.core.autonomy_stack.safety.safety_manager import SafetyManager
from skylink_v2x.core.autonomy_stack.plan.behavior_agent import BehaviorAgent as VehiclePlanningManager
from skylink_v2x.core.autonomy_stack.actuation.control_manager import ControlManager as VehicleController
from skylink_v2x.core.autonomy_stack.mapping.map_manager import MapManager
from skylink_v2x.core.autonomy_stack.localization.localization_manager import VehicleLocalizationManager
from skylink_v2x.utils import read_yaml, velocity_to_speed_2d, velocity_to_speed_3d

class VehicleManager(DynamicManager):
    """
    Class for managing a dynamic vehicle agent.
    Update agent manager in SimScenario after initialization.

    Attributes
    ----------
    vehicle: Vehicle
        The vehicle agent managed by this manager.  
    sim_map: Map
        The map associated with the vehicle.
    config: dict
        The configuration of the vehicle.
    app_list: list
        The list of applications to be run by the vehicle.
    is_log: bool
        Indicator to log data.
    curr_time: datetime
        The current time of the vehicle manager.
    v2x_manager: V2XManager
        The V2X manager of the vehicle manager.
    map_manager: MapManager
        The map manager of the vehicle manager.

    Methods
    ----------
    set_destination(self, start_location, end_location, clean: bool, end_reset: bool)
        Set the destination of the vehicle.
    update(self)
        Update the internal state of the vehicle manager.
    run(self)
        Execute the vehicle manager's control loop.
    remove(self)
        Clean up resources associated with this vehicle manager
    """
    def __init__(self,
                 vehicle,
                 carla_map,
                 config,
                 comm_manager=None):
        
        super().__init__(vehicle.id, vehicle.get_world(), config)

        self.type = "vehicle"
        self.vehicle = vehicle

        # Retrieve the configure for different modules
        perception_config = read_yaml(config['perception']['base'])
        localization_config = read_yaml(config['localization']['base'])
        map_config = read_yaml(config['mapping']['base'])
        planner_config = read_yaml(config['planning']['base'])
        control_config = read_yaml(config['control']['base'])

        self.perception_manager = VehiclePerceptionManager(carla_world=self.world,
                                                           agent_id=self.agent_id,
                                                           config=perception_config,
                                                           comm_manager=comm_manager)
        
        self.localizer = VehicleLocalizationManager(vehicle, 
                                                    localization_config, 
                                                    comm_manager=comm_manager)
        
        self.map_manager = MapManager(carla_world=self.world,
                                      agent_id=self.agent_id, 
                                      carla_map=carla_map, 
                                      config=map_config,
                                      comm_manager=comm_manager)
        
        # self.safety_manager: SafetyManager = SafetyManager(comm_manager=comm_manager)
        # self.planner = VehiclePlanningManager(vehicle=vehicle, 
        #                               carla_map=carla_map, 
        #                               config=planner_config,
        #                               comm_manager=comm_manager,
        #                               autonomy_option=autonomy_option)
        # self.controller = VehicleController(vehicle=vehicle, 
        #                                     config=control_config,
        #                                     comm_manager=comm_manager,
        #                                     autonomy_option=autonomy_option)
        self.planner = VehiclePlanningManager(vehicle=vehicle, 
                                      carla_map=carla_map, 
                                      config_yaml=planner_config,
                                      comm_manager=comm_manager)
        self.controller = VehicleController(control_config=control_config)
        comm_manager.register(self)

        # if curr_time:
        #     self.logger.set_path(curr_time)

        # self.logger.set_veh_sensors(self.perception_manager)
    

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
        print(f"VehicleManager {self.agent_id}: setting destination from {start_location}\
               to {end_location}, clean: {clean}, end_reset: {end_reset}")
        
        # Delegate destination setting to the planner.
        self.planner.set_destination(start_location, end_location, clean, end_reset)  


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
        self.planner.update_information(ego_pos=localization_data[self.agent_id]['position'],
                                        ego_speed=velocity_to_speed_3d(localization_data[self.agent_id]['velocity']),
                                        objects=perception_data[self.agent_id])
        
        self.planner.run_step()

    def update_control(self, 
                       localization_data=None, 
                       planning_outputs=None):
        """
        Update the control module.
        """
        # control_signal = self.controller.apply_control(*planning_outputs)
        self.controller.update_info(ego_pos=localization_data[self.agent_id]['position'],
                                    ego_speed=velocity_to_speed_3d(localization_data[self.agent_id]['velocity']))
        control_signal = self.controller.run_step(*planning_outputs[self.agent_id])
        self.vehicle.apply_control(control_signal)
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
            


