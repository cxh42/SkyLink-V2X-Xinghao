"""
Coordinate transformation utilities for map management.
"""
import numpy as np
import carla

def lateral_shift(transform, shift):
    """Convert a transform to a location with lateral shift."""
    transform_loc = transform.location
    transform_rot = transform.rotation
    
    # Convert to radians
    yaw_rad = np.radians(transform_rot.yaw)
    
    # Apply shift - positive is right, negative is left
    x_shift = -shift * np.sin(yaw_rad)
    y_shift = shift * np.cos(yaw_rad)
    
    shifted_loc = carla.Location(
        x=transform_loc.x + x_shift,
        y=transform_loc.y + y_shift,
        z=transform_loc.z
    )
    return shifted_loc

def list_loc2array(loc_list):
    """Convert a list of carla.Location objects to a numpy array."""
    return np.array([[loc.x, loc.y, loc.z] for loc in loc_list])

def list_wpt2array(waypoint_list):
    """Convert a list of carla.Waypoint objects to a numpy array of their locations."""
    return np.array([[wpt.transform.location.x, wpt.transform.location.y, wpt.transform.location.z] 
                     for wpt in waypoint_list])

def world_to_sensor(points, sensor_transform):
    """Transform points from world coordinates to sensor coordinates."""
    # Create transformation matrix
    rotation = sensor_transform.rotation
    location = sensor_transform.location
    
    # Convert to radians
    roll, pitch, yaw = np.radians(rotation.roll), np.radians(rotation.pitch), np.radians(rotation.yaw)
    
    # Create rotation matrices and combine them
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    
    roll_matrix = np.array([
        [1, 0, 0],
        [0, cos_roll, -sin_roll],
        [0, sin_roll, cos_roll]
    ])
    
    pitch_matrix = np.array([
        [cos_pitch, 0, sin_pitch],
        [0, 1, 0],
        [-sin_pitch, 0, cos_pitch]
    ])
    
    yaw_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    rotation_matrix = np.dot(np.dot(yaw_matrix, pitch_matrix), roll_matrix)
    
    # Create sensor-to-world transform
    sensor_to_world = np.identity(4)
    sensor_to_world[:3, :3] = rotation_matrix
    sensor_to_world[:3, 3] = [location.x, location.y, location.z]
    
    # Invert to get world-to-sensor transform
    world_to_sensor_matrix = np.linalg.inv(sensor_to_world)
    
    # Apply transformation
    return np.dot(world_to_sensor_matrix, points)


def cv2_subpixel(polygon):
    """Format polygon for OpenCV drawing with subpixel precision."""
    polygon = np.array(polygon, dtype=np.float32)
    
    if len(polygon.shape) == 2:
        return polygon.reshape((-1, 1, 2))
    elif len(polygon.shape) == 3:
        return polygon.reshape((polygon.shape[0] * polygon.shape[1], 1, 2))
    
    return polygon
