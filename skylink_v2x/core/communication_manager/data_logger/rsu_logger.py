# Author: Fengze Yang <fred.yang@utah.edu>
# License:

import os
from .agent_logger import AgentLogger

class RSULogger(AgentLogger):
    """Logger for RSU (Road Side Unit) agents."""
    def __init__(self):
        super().__init__()
        # self.rgb_camera = None
        # self.lidar = None
        
    # def set_sensors(self, perception_manager):
    #     """Set RSU sensors from the perception manager."""
    #     assert hasattr(perception_manager, 'rgb_camera') \
    #         and hasattr(perception_manager, 'lidar'), \
    #         "RSU sensors not found; check perception settings."
        
    #     self.rgb_camera = perception_manager.rgb_camera
    #     self.lidar = perception_manager.lidar
        
    # def update_data(self, localization_manager, planner=None):
    #     """Update RSU data including position and sensor data."""
    #     # Initialize data log sections
    #     if 'rsu' not in self.data_log:
    #         self.data_log['rsu'] = {}
            
    #     # Update RSU position
    #     self.rsu_pos = localization_manager.get_ego_pos()

    #     self.data_log['rsu'].update({
    #         'rsu_true_pos': self._get_position_list(self.rsu_pos)
    #     })
        
    #     # Update sensor data
    #     self.data_log["rsu_cameras"] = self._update_camera_data(self.rgb_camera, self.lidar)
    #     self.data_log["rsu_lidar"] = self._update_lidar_data(self.lidar)
        
    # def save(self, localization_manager, planner, agent_id):
    #     """Save RSU-related data for a specific agent."""
    #     folder = self._get_agent_folder(agent_id)
    #     self.file_count += 1

    #     # Update all data
    #     self.update_data(localization_manager, planner)

    #     # Save image and point cloud files
    #     self.save_rgb_image(self.file_count, folder+"/camera_images", self.rgb_camera)
    #     self.save_lidar_points(folder+"/lidar_points", self.lidar)

    #     # Save YAML metadata
    #     rsu_yaml = os.path.join(folder, f"{self.file_count:06d}_rsu.yaml")
    #     self.save_yaml(self.data_log, rsu_yaml)
