# Author: Fengze Yang <fred.yang@utah.edu>
# License:


from skylink_v2x.core.agent_stack.agent_manager import AgentManager
from skylink_v2x.core.autonomy_stack.perception.perception_manager import VehiclePerceptionManager
from skylink_v2x.core.autonomy_stack.safety.safety_manager import SafetyManager
from skylink_v2x.core.autonomy_stack.plan.behavior_agent import BehaviorAgent as VehiclePlanningManager
from skylink_v2x.core.autonomy_stack.actuation.control_manager import ControlManager as VehicleController
from skylink_v2x.core.autonomy_stack.mapping.map_manager import MapManager
from skylink_v2x.core.autonomy_stack.localization.localization_manager import VehicleLocalizationManager
from skylink_v2x.utils import read_yaml, velocity_to_speed_2d, velocity_to_speed_3d

class RSUManager(AgentManager):
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
                 rsu,
                 carla_map,
                 config,
                 comm_manager=None):
        
        super().__init__(rsu.id, rsu.get_world(), config)

        self.type = "rsu"
        self.rsu = rsu

        # Retrieve the configure for different modules
        perception_config = read_yaml(config['perception']['base'])
        localization_config = read_yaml(config['localization']['base'])
        map_config = read_yaml(config['mapping']['base'])
        # planner_config = read_yaml(config['planning']['base'])

        self.perception_manager = VehiclePerceptionManager(carla_world=self.world,
                                                           agent_id=self.agent_id,
                                                           config=perception_config,
                                                           comm_manager=comm_manager)
        
        self.localizer = VehicleLocalizationManager(rsu, 
                                                    localization_config, 
                                                    comm_manager=comm_manager)
        
        self.map_manager = MapManager(carla_world=self.world,
                                      agent_id=self.agent_id, 
                                      carla_map=carla_map, 
                                      config=map_config,
                                      comm_manager=comm_manager)
        
        # RSUPlanning Manager is used for planning for other vehicles. Currently not used.
        # self.planner = RSUPlanningManager(vehicle=vehicle, 
        #                               carla_map=carla_map, 
        #                               config_yaml=planner_config,
        #                               comm_manager=comm_manager)
        comm_manager.register(self)

    # RSUPlanning Manager should be used for setting destination for other vehicles. Currently not used.
    def set_destination(self, 
                        start_location, 
                        end_location, 
                        clean=False, 
                        end_reset=True) -> None:
        pass


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
        # self.planner.update_information(ego_pos=localization_data[self.agent_id]['position'],
        #                                 ego_speed=velocity_to_speed_3d(localization_data[self.agent_id]['velocity']),
        #                                 objects=perception_data[self.agent_id])
        
        # self.planner.run_step()
        pass

    def update_control(self, 
                       localization_data=None, 
                       planning_outputs=None):
        """
        Update the control module.
        """
        pass
    
    
    def remove(self) -> None:
        """
        Clean up resources associated with this vehicle manager.
        
        Invokes cleanup routines on all integrated modules.
        """
        print(f"RSUManager {self.agent_id}: cleaning up resources")
        # TODO: Implement remove() method for all modules.
        self.perception_manager.destroy()  # Assumes remove() method exists.
        self.map_manager.remove()
        self.localizer.remove()
        


