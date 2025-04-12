# Author: Fengze Yang <fred.yang@utah.edu>
# License:

from datetime import datetime
import os
from .vehicle_logger import VehicleLogger
from .drone_logger import DroneLogger
from .rsu_logger import RSULogger
from skylink_v2x.core.agent_stack.agent_manager import AgentManager
from skylink_v2x.utils import velocity_to_speed_3d
from skylink_v2x.core.communication_manager.data_logger.utils import (
    get_position_list, save_yaml, get_extent_list
)
from skylink_v2x.carla_helper import OBJECT_CONFIG

class DataLogger:
    """
    A class to log data from various sensors and managers.
    
    This class orchestrates the logging of data from different agent types
    (vehicles, drones, RSUs) by delegating to specialized logger classes.
    """
    def __init__(self, logger_config):
        self.logger_config = logger_config
        self.vehicle_logger = VehicleLogger()
        self.drone_logger = DroneLogger()
        self.rsu_logger = RSULogger()

    # def set_veh_sensors(self, perception_manager):
    #     """Set vehicle sensors from the perception manager."""
    #     self.vehicle_logger.set_sensors(perception_manager)

    # def set_drone_sensors(self, perception_manager):
    #     """Set drone sensors from the perception manager."""
    #     self.drone_logger.set_sensors(perception_manager)

    # def set_rsu_sensors(self, perception_manager):
    #     """Set RSU-related data from the perception manager."""
    #     self.rsu_logger.set_sensors(perception_manager)
    
    def save_objects(self, detected_objects):
        obj_dict = {}

        for obj in detected_objects:
            actor = obj['actor']
            actor_id = actor.id
            trans = actor.get_transform()
            bbox = actor.bounding_box #TODO: some objects such as traffic lights do not have bounding box
            extend = bbox.extent 
            speed = velocity_to_speed_3d(actor.get_velocity(), True)

            assert actor_id != -1, "Invalid vehicle id; check perception settings."
            
            obj_dict[actor_id] = {'bp_id': actor.type_id,
                                'location': get_position_list(trans),
                                'center': get_position_list(bbox),
                                'extent': get_extent_list(extend),
                                'class': OBJECT_CONFIG.get_by_blueprint(actor.type_id).class_idx,
                                'speed': speed}
            
        objects_info = os.path.join(self.scene_folder, f"objects.yaml")
        save_yaml(obj_dict, objects_info)

    def set_path(self, start_time, agent_id, timestamp):
        """Set the base folder path for saving logs."""
        # current_path = os.path.dirname(os.path.realpath(__file__))
        scene_folder = os.path.join(self.logger_config['data_path'], 
                                   start_time, 
                                   f"timestamp_{timestamp:06d}")
        save_folder = os.path.join(scene_folder,
                                   f"agent_{agent_id:06d}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        self.save_folder = save_folder
        self.scene_folder = scene_folder
        
    def save_data(self,
                  agent_comm_helper,
                  agent_manager: AgentManager,
                  agent_id):
        # self.set_path(curr_time)
        if agent_manager.type == 'vehicle':
            self.vehicle_logger.save(self.save_folder, agent_comm_helper, agent_id)
        elif agent_manager.type == 'drone':
            self.drone_logger.save(self.save_folder, agent_comm_helper, agent_id)
        elif agent_manager.type == 'rsu':
            self.rsu_logger.save(self.save_folder, agent_comm_helper, agent_id)
        else:
            raise ValueError("Agent manager type not recognized.")
            
    
            
