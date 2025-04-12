# Author: Fengze Yang <fred.yang@utah.edu>
# License:

import os
from .agent_logger import AgentLogger

class VehicleLogger(AgentLogger):
    """Logger for vehicle agents."""
    def __init__(self):
        super().__init__()
        # self.rgb_camera = None
        # self.lidar = None
        # self.vehicle_list = []
        
    # def set_sensors(self, perception_manager):
    #     """Set vehicle sensors from the perception manager."""
    #     assert hasattr(perception_manager, 'rgb_camera') \
    #         and hasattr(perception_manager, 'lidar'), \
    #         "Vehicle sensors not found; check perception settings."
        
    #     self.rgb_camera = perception_manager.rgb_camera
    #     self.lidar = perception_manager.lidar
    #     self.vehicle_list = perception_manager.detect_objects('vehicles', [])
        
    # def update_data(self, localization_manager, planner=None):
    #     """Update vehicle data including position, speed, and detected vehicles."""
    #     # Initialize data log sections
    #     if 'ego_veh' not in self.data_log:
    #         self.data_log['ego_veh'] = {}
            
    #     # Update ego vehicle data
    #     self.ego_pos_predict = localization_manager.get_ego_pos()
        
    #     if hasattr(localization_manager, 'vehicle'):
    #         self.ego_pos_GT = localization_manager.vehicle.get_transform()
    #     else:
    #         raise ValueError("Ego vehicle position not found in location manager.")
        
    #     self.ego_speed = localization_manager.get_ego_speed()

    #     if planner:
    #         trajectory_list = self._get_trajectory_list(planner)
    #         self.data_log['ego_veh'].update({'plan_trajectory': trajectory_list})

    #     self.data_log['ego_veh'].update({
    #         'predicted_ego_pos': self._get_position_list(self.ego_pos_predict),
    #         'true_ego_pos': self._get_position_list(self.ego_pos_GT),
    #         'ego_speed': float(self.ego_speed)
    #     })
        
    #     # Update surrounding vehicles data
    #     self._update_other_vehicles()
        
    #     # Update sensor data
    #     self.data_log["veh_cameras"] = self._update_camera_data(self.rgb_camera, self.lidar)
    #     self.data_log["veh_lidar"] = self._update_lidar_data(self.lidar)
        
    # def _update_other_vehicles(self):
    #     """Update the log for surrounding vehicles."""
    #     veh_dict = {}

    #     for veh in self.vehicle_list:
    #         veh_id = veh.carla_id
    #         pos = veh.obj_position
    #         bbx = veh.bounding_box
    #         speed = veh.get_speed() if hasattr(veh, 'get_speed') else 0.0

    #         assert veh_id != -1, "Invalid vehicle id; check perception settings."
            
    #         veh_dict[veh_id] = {'bp_id': veh.type_id,
    #                             'color': veh.color,
    #                             'location': self._get_position_list(pos.location),
    #                             'center': self._get_position_list(bbx.location),
    #                             'angle': self._get_rotation_list(pos.rotation),
    #                             'extent': self._get_extent_list(bbx.extent),
    #                             'speed': speed}
            
    #     self.data_log['other_veh'] = veh_dict
        
    # def save(self, localization_manager, planner, agent_id):
    #     """Save vehicle-related data for a specific agent."""
    #     folder = self._get_agent_folder(agent_id)

    #     # Update all data
    #     self.update_data(localization_manager, planner)

    #     # Save image and point cloud files
    #     self.save_rgb_image(sensor_name, 
    #                         os.path.join(folder, sensor_name),
    #                         self.rgb_camera)
    #     self.save_lidar_points(folder+"/lidar_points", self.lidar)

    #     # Save YAML metadata
    #     veh_yaml = os.path.join(folder, f"{self.file_count:06d}_veh.yaml")
    #     self.save_yaml(self.data_log, veh_yaml)
