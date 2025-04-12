# Author: Fengze Yang <fred.yang@utah.edu>

import os
import yaml
import json
import cv2
import numpy as np
import open3d as o3d
from skylink_v2x.core.autonomy_stack.perception.sensor_transformer import SensorTransformer


#######
# File Related Functions
#######

def matrix2list(matrix):
    """Convert a numpy matrix to a list for serialization."""
    assert len(matrix.shape) == 2
    return matrix.tolist()

def save_yaml(data, file_path):
    """Save data as a YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


#######
# Sensor Related Functions
#######

def get_position_list(pos):
    """Extract position and orientation as a list from a position object."""
    return [pos.location.x,
            pos.location.y,
            pos.location.z,
            pos.rotation.roll,
            pos.rotation.yaw,
            pos.rotation.pitch]

def get_extent_list(extent):
    """Get the extent list from an extent object."""
    return [extent.x, extent.y, extent.z]

def get_trajectory_list(planner):
    """Get the trajectory list from the planner."""
    trajectory_deque = planner.get_local_planner().get_trajectory()
    trajectory_list = []

    for i in range(len(trajectory_deque)):
        tmp_buffer = trajectory_deque.popleft()
        x = tmp_buffer[0].location.x
        y = tmp_buffer[0].location.y
        spd = tmp_buffer[1]
        trajectory_list.append([x, y, spd])

    return trajectory_list

def save_rgb_image(sensor_name, folder, data):
    """Save raw RGB images from cameras."""
    ensure_directory(folder)
    image_name = f"{sensor_name}.jpeg"
    cv2.imwrite(os.path.join(folder, image_name), data['image'])

def save_lidar_points(sensor_name, folder, data):
    """Save raw lidar point cloud data."""
    ensure_directory(folder)
    
    point_cloud = data['points']

    point_xyz = point_cloud[:, :-1]
    point_intensity = point_cloud[:, -1]
    point_intensity = np.c_[point_intensity,
                           np.zeros_like(point_intensity),
                           np.zeros_like(point_intensity)]
    
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

    pcd_name = f"{sensor_name}.pcd"
    o3d.io.write_point_cloud(os.path.join(folder, pcd_name),
                            o3d_pcd,
                            write_ascii=True)

def save_semanticlidar_points(sensor_name, folder, data):
    """Save raw semantic lidar point cloud data."""
    ensure_directory(folder)
    
    point_xyz = data['points']
    obj_tag = data['obj_tag']
    
    normalized_tags = np.zeros(len(point_xyz))
    if len(obj_tag) > 0:
        max_tag = np.max(obj_tag)
        if max_tag > 0:
            normalized_tags = obj_tag / max_tag

    point_colors = np.c_[normalized_tags,
                       np.zeros_like(normalized_tags),
                       np.zeros_like(normalized_tags)]
    
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
    o3d_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    pcd_name = f"{sensor_name}.pcd"
    o3d.io.write_point_cloud(os.path.join(folder, pcd_name),
                          o3d_pcd,
                          write_ascii=True)
    
    semantic_file = os.path.join(folder, f"{sensor_name}_semantic.npz")
    np.savez(semantic_file, obj_tag=obj_tag, obj_idx=data['obj_idx'])



def compute_camera_params(camera_data, ref_data=None):
    """Compute camera intrinsic and extrinsic parameters."""
    
    # Basic camera parameters
    camera_param = {}
    camera_param["cords"] = get_position_list(camera_data["trans"])
    # Camera intrinsic matrix
    intrinsic = SensorTransformer.compute_camera_intrinsics(camera_data)
    camera_param["intrinsic"] = matrix2list(intrinsic)

    # Calculate extrinsic matrix if reference lidar is provided
    if ref_data is not None:
        T_lidar_world = SensorTransformer.transform_to_world_matrix(
            ref_data['trans'])
        T_camera_world = SensorTransformer.transform_to_world_matrix(
            camera_data['trans'])
        T_world_to_camera = np.linalg.inv(T_camera_world)
        T_lidar_to_camera = T_world_to_camera @ T_lidar_world
        camera_param["extrinsic"] = matrix2list(T_lidar_to_camera)

    return camera_param

def save_mapping_data(mapping_data, folder):
    """

    Args:
        mapping_data (Dict): dictionary containing static_bev, dynamic_bev, vis_bev, and vector_map
        folder (str): folder to save the mapping data
    """
    
    static_bev = mapping_data['static_bev']
    dynamic_bev = mapping_data['dynamic_bev']
    dynamic_bev_layers = mapping_data['dynamic_bev_layers']
    vis_bev = mapping_data['vis_bev']
    vector_map = mapping_data['vector_map']
    
    ensure_directory(folder)
    cv2.imwrite(os.path.join(folder, f"map_dynamic_bev.jpeg"), dynamic_bev)
    cv2.imwrite(os.path.join(folder, f"map_static_bev.jpeg"), static_bev)
    cv2.imwrite(os.path.join(folder, f"map_vis_bev.jpeg"), vis_bev)
    for index in range(dynamic_bev_layers.shape[0]):
        cv2.imwrite(os.path.join(folder, f"map_dynamic_bev_layer_{index}.jpeg"), dynamic_bev_layers[index])
    save_vector_map_to_json(vector_map, os.path.join(folder, "vector_map.json"))

def save_vector_map_to_json(vector_map, filepath):
    with open(filepath, 'w') as f:
        json.dump(vector_map, f, indent=2)
