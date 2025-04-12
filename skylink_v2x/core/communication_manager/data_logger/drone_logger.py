# Author: Fengze Yang <fred.yang@utah.edu>
# License:

import os
from .agent_logger import AgentLogger


class DroneLogger(AgentLogger):
    """Logger for drone agents."""
    def __init__(self):
        super().__init__()
        # self.rgb_camera = None
        # self.lidar = None
        
    # def set_sensors(self, perception_manager):
    #     """Set drone sensors from the perception manager."""
    #     assert hasattr(perception_manager, 'rgb_camera') \
    #         and hasattr(perception_manager, 'lidar'), \
    #         "Drone sensors not found; check perception settings."

    #     self.rgb_camera = perception_manager.rgb_camera
    #     self.lidar = perception_manager.lidar
        
    # def update_data(self, 
    #                 vehicle_manager: VehicleManager):
    #     # Initialize data log sections
    #     perception_manager = vehicle_manager.perception_manager
    #     localization_manager = vehicle_manager.localizer
        
    #     localization_result = localization_manager.localize()

    #     # if planner:
    #     #     trajectory_list = self._get_trajectory_list(planner)
    #     #     self.data_log['planning'].update({'plan_trajectory': trajectory_list})

    #     self.data_log['odometry'].update({
    #         'ego_pos': self._get_position_list(localization_result['position']),
    #         'ego_speed': localization_result['speed_mps'],
    #     })
        
    #     ref = None
    #     for sensor_name, sensor in perception_manager.get_sensors().items():
    #         if isinstance(sensor, LidarSensor):
    #             ref = sensor
    #     assert ref is not None, "Current version use Lidar sensor for corrdiate transfer, therefore must be provided." 
    #     # ref = vehicle_manager.localizer.get_ego_pos() # This is Ideal
        
    #     for sensor_name, sensor in perception_manager.get_sensors().items():
    #         if isinstance(sensor, CameraSensor):
    #             self.data_log[sensor_name] = self._update_camera_data(sensor, ref)
    #         elif isinstance(sensor, LidarSensor):
    #             self.data_log[sensor_name] = self._update_lidar_data(sensor)
        
        
    # def save(self, vehicle_manager, agent_id):
    #     """Save drone-related data for a specific agent."""
    #     folder = self.save_folder
    #     localization_manager = vehicle_manager.localizer
    #     perception_manager = vehicle_manager.perception_manager
    #     planner = vehicle_manager.planner
    #     self.file_count += 1

    #     # Update all data
    #     self.update_data(localization_manager, planner)
        
    #     for sensor in self._sensors

    #     # Save image and point cloud files
    #     self.save_rgb_image(self.file_count, folder+"/camera_images", self.rgb_camera)
    #     self.save_lidar_points(folder+"/lidar_points", self.lidar)

    #     # Save YAML metadata
    #     drone_yaml = os.path.join(folder, f"{self.file_count:06d}_drone.yaml")
    #     self.save_yaml(self.data_log, drone_yaml)
        
        
        
