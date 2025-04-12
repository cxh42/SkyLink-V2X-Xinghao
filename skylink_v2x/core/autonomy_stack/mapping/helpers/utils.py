"""
Utility functions for map management.
"""
import numpy as np

def convert_tl_status(tl_state):
    """Convert CARLA traffic light state to string representation."""
    state_map = {
        0: 'red',     # carla.TrafficLightState.Red
        1: 'yellow',  # carla.TrafficLightState.Yellow
        2: 'green',   # carla.TrafficLightState.Green
        3: 'normal'   # carla.TrafficLightState.Off
    }
    return state_map.get(int(tl_state), 'normal')

def get_bounds(left_points, right_points):
    """Calculate bounding box of a lane from its left and right points."""
    # Combine left and right points
    all_points = np.vstack([left_points, right_points])
    
    # Extract x and y coordinates
    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]
    
    # Calculate bounds
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # Create bounds array with shape (1, 2, 2)
    return np.array([[[x_min, y_min], [x_max, y_max]]])
