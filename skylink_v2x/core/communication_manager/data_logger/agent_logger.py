# Author: Fengze Yang <fred.yang@utah.edu>

import os
from abc import ABC, abstractmethod

from skylink_v2x.core.autonomy_stack.perception.sensors import (
    CameraSensor, LidarSensor, SemanticLidarSensor
)
from skylink_v2x.core.communication_manager.data_logger.utils import (
    get_position_list, save_yaml, save_rgb_image, 
    save_lidar_points, save_semanticlidar_points,
    compute_camera_params, save_mapping_data
)
from skylink_v2x.core.autonomy_stack.perception.sensor_transformer import SensorTransformer

class AgentLogger(ABC):
    """
    Abstract base class for agent data loggers (vehicle, drone, RSU).
    
    This class provides common functionality for logging sensor data and
    agent states, with specialized implementations in subclasses.
    """
    def __init__(self):
        self.data_log = {}
        
    def update_data(self, agent_comm_helper, agent_id):
        """Update the data log with the latest sensor and agent state data."""
        localization_data = agent_comm_helper.get_localization_data()[-1]
        self.data_log['agent_type'] = agent_comm_helper.get_agent_type()
        self.data_log['odometry'] = {
            'ego_pos': get_position_list(localization_data['position']),
            'ego_speed': localization_data['speed_mps'],
        }
        perception_raw_data = agent_comm_helper.get_perception_data()[-1]['raw_data']
        
        # Find reference lidar sensor for coordinate transformations
        lidar_sensors = [data for name, data in perception_raw_data.items() 
                        if data['sensor_type'] == LidarSensor.__name__]
        assert len(lidar_sensors) <= 1, "Current version uses Lidar sensor for coordinate transfer, therefore should not have more than one lidar sensor."
        
        ref_data = lidar_sensors[0] if len(lidar_sensors) == 1 else None
        
        for sensor_name, sensor_data in perception_raw_data.items():
            if sensor_data['sensor_type'] == CameraSensor.__name__:
                self.data_log[sensor_name] = self._update_camera_data(sensor_data, ref_data)
            elif sensor_data['sensor_type'] == LidarSensor.__name__:
                self.data_log[sensor_name] = self._update_lidar_data(sensor_data)
            elif sensor_data['sensor_type'] == SemanticLidarSensor.__name__:
                self.data_log[sensor_name] = self._update_semanticlidar_data(sensor_data)
            else:
                raise ValueError(f"Sensor type {sensor_data['sensor_type']} not recognized.")

    def save(self, save_folder, agent_comm_helper, agent_id):
        """Save agent data to disk."""
        # Update all data
        self.update_data(agent_comm_helper, agent_id)
        
        perception_raw_data = agent_comm_helper.get_perception_data()[-1]['raw_data']
        for sensor_name, perception_data in perception_raw_data.items():
            
            if perception_data['sensor_type'] == CameraSensor.__name__:
                save_rgb_image(sensor_name, save_folder, perception_data)
            elif perception_data['sensor_type'] == LidarSensor.__name__:
                save_lidar_points(sensor_name, save_folder, perception_data)
            elif perception_data['sensor_type'] == SemanticLidarSensor.__name__:
                save_semanticlidar_points(sensor_name, save_folder, perception_data)
            else:
                raise ValueError(f"Sensor type {perception_data['sensor_type']} not recognized.")
            
        mapping_data = agent_comm_helper.get_mapping_data()[-1]
        save_mapping_data(mapping_data, save_folder)
            
        # Save YAML metadata
        metadata = os.path.join(save_folder, f"metadata.yaml")
        save_yaml(self.data_log, metadata)
        
    
    def _update_camera_data(self, camera_data, ref_data=None):
        """Update the log for camera sensors."""
        return compute_camera_params(camera_data, ref_data)
    
    def _update_lidar_data(self, lidar_data):
        """Update the log for the lidar sensor."""
        if lidar_data is not None:
            trans = lidar_data['trans']
            return {"lidar_pose": get_position_list(trans)}
        else:
            return {}
        
    def _update_semanticlidar_data(self, lidar_data):
        """Update the log for the semantic lidar sensor."""
        if lidar_data is not None:
            trans = lidar_data['trans']
            return {"lidar_pose": get_position_list(trans)}
        else:
            return {}
