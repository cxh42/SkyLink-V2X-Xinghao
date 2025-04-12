"""
Visualization utilities for rendering map data.
"""
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

def draw_agent(corner_list, image):
    """Draw dynamic agents on the BEV image."""
    result = image.copy()
    
    # Class to color mapping
    agent_colors = {
        0: (0, 255, 0),       # Pedestrian (green)
        1: (255, 165, 0),     # Bicycle (orange)
        2: (128, 0, 128),     # Motorcycle (purple)
        3: (0, 0, 255),       # Car (blue)
        4: (255, 0, 0),       # Truck (red)
        5: (139, 69, 19),     # Trailer (brown)
        6: (255, 255, 0),     # Bus (yellow)
        7: (128, 128, 128),   # Others (gray)
        8: (255, 255, 0),   # Traffic Light (yellow)
        9: (0, 255, 255),     # Traffic Sign (cyan)
        10: (255, 105, 180),  # Traffic Cone (pink)
    }
    
    for item in corner_list:
        corners = item['corner']
        class_idx = item['class_idx']
        
        # Skip if corners is empty
        if len(corners) == 0:
            continue
        
        # Get color based on class index
        color = agent_colors[class_idx]
        
        # Draw filled polygon and outline
        cv2.fillPoly(result, [corners.astype(np.int32)], color)
        cv2.polylines(result, [corners.astype(np.int32)], True, (0, 0, 0), 1)
    
    return result

def draw_agent_layers(corner_list, height, width):
    """
    Draw dynamic agents on separate layers.

    Parameters:
      corner_list: list of dicts with keys 'corner' (numpy array of polygon vertices)
                   and 'class_idx' (int from 0 to 10).
      height: canvas height (pixels)
      width: canvas width (pixels)

    Returns:
      A numpy array of shape (11, height, width) with uint8 values (0-255),
      where each channel corresponds to one agent class.
    """
    layers = np.zeros((11, height, width), dtype=np.uint8)

    for item in corner_list:
        corners = item['corner']
        class_idx = item['class_idx']
        
        if len(corners) == 0:
            continue
        
        # Create a temporary layer to ensure contiguous memory.
        cv2.fillPoly(layers[class_idx], [corners.astype(np.int32)], 255)
        cv2.polylines(layers[class_idx], [corners.astype(np.int32)], True, 128, 1)
    
    return layers

def draw_road(lanes_area_list, image):
    """Draw road surface on the BEV image."""
    result = image.copy()
    road_color = (128, 128, 128)  # Gray
    
    for lane_area in lanes_area_list:
        # Skip if lane_area is empty
        if len(lane_area) == 0:
            continue
        cv2.fillPoly(result, [lane_area.astype(np.int32)], road_color)
    
    return result

def draw_lane(lanes_area_list, lane_type_list, image):
    """Draw lane markings on the BEV image."""
    result = image.copy()
    
    # Lane type colors
    lane_colors = {
        'normal': (255, 255, 255),  # White
        'red': (0, 0, 255),         # Red
        'yellow': (0, 255, 255),    # Yellow
        'green': (0, 255, 0)        # Green
    }
    
    for lane_area, lane_type in zip(lanes_area_list, lane_type_list):
        # Skip if lane_area is empty
        if len(lane_area) == 0:
            continue
        
        color = lane_colors.get(lane_type, lane_colors['normal'])
        cv2.polylines(result, [lane_area.astype(np.int32)], True, color, 2)
    
    return result

def load_vector_map(map_dir):
    """Load vector map data from a directory."""
    vector_map = {}
    try:
        with open(f"{map_dir}", 'r') as file:
            vector_map = json.load(file)
    except FileNotFoundError:
        print(f"Vector map file not found in {map_dir}.")
    
    return vector_map

def draw_vector_map(vector_map=None, save_path=None, map_dir=None):
    # """Visualize the vector map using matplotlib."""
    # import matplotlib.pyplot as plt
    
    # fig, ax = plt.subplots()
    
    # for lane in vector_map['lanes']:
    #     centerline = lane['centerline']
    #     ax.plot(centerline[:, 0], centerline[:, 1], label=f"Lane {lane['id']}")
    
    # ax.set_title("Vector Map Visualization")
    # ax.set_xlabel("X Coordinate")
    # ax.set_ylabel("Y Coordinate")
    # ax.legend()
    
    # if save_path:
    #     plt.savefig(save_path)
    # else:
    #     plt.show()
    
    if map_dir:
        # Load the vector map from the specified directory
        vector_map = load_vector_map(map_dir)
    image = visualize_vector_map(vector_map)
    cv2.imwrite(save_path, image)
        


def visualize_vector_map(json_file_or_data, output_file=None):
    """
    Visualize vector map data in a simple and clean way
    
    Args:
        json_file_or_data: Path to JSON file or JSON dictionary
        output_file: Optional path to save the visualization
        
    Returns:
        The visualization image
    """
    # Load data
    if isinstance(json_file_or_data, str):
        with open(json_file_or_data, 'r') as f:
            data = json.load(f)
    else:
        data = json_file_or_data
    
    # Extract all points to determine map dimensions
    all_points = []
    
    for lane in data.get('lanes', []):
        if 'centerline' in lane:
            all_points.extend([point[:2] for point in lane['centerline']])
        if 'left_boundary' in lane:
            all_points.extend([point[:2] for point in lane['left_boundary']])
        if 'right_boundary' in lane:
            all_points.extend([point[:2] for point in lane['right_boundary']])
    
    if not all_points:
        print("No points found in the map data")
        return None
    
    # Calculate bounds and scale
    all_points = np.array(all_points)
    min_x, min_y = np.min(all_points, axis=0) - 1
    max_x, max_y = np.max(all_points, axis=0) + 1
    
    # Create canvas
    width, height = 800, 800
    scale = min(width / (max_x - min_x), height / (max_y - min_y)) * 0.9
    
    # Function to transform map coordinates to image coordinates
    def transform(point):
        x, y = point[:2]
        px = int((x - min_x) * scale) + 50
        py = height - int((y - min_y) * scale) - 50  # Flip y-axis for image coordinates
        return (px, py)
    
    # Create blank canvas
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw each lane
    for lane in data.get('lanes', []):
        # Draw centerline in blue
        if 'centerline' in lane and lane['centerline']:
            points = [transform(point) for point in lane['centerline']]
            
            # Draw lines
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], (255, 0, 0), 2)  # Blue
            
            # Draw points
            for point in points:
                cv2.circle(canvas, point, 3, (0, 0, 255), -1)  # Red points
        
        # Draw left_boundary in green
        if 'left_boundary' in lane and lane['left_boundary']:
            points = [transform(point) for point in lane['left_boundary']]
            
            # Draw lines
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], (0, 200, 0), 2)  # Green
            
            # Draw points
            for point in points:
                cv2.circle(canvas, point, 3, (0, 100, 0), -1)  # Dark green points
        
        # Draw right_boundary in yellow/orange
        if 'right_boundary' in lane and lane['right_boundary']:
            points = [transform(point) for point in lane['right_boundary']]
            
            # Draw lines
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], (0, 200, 255), 2)  # Yellow/orange
            
            # Draw points
            for point in points:
                cv2.circle(canvas, point, 3, (0, 128, 255), -1)  # Darker orange points
    
    # Show and save
    if output_file:
        cv2.imwrite(output_file, canvas)
    
    return canvas

if __name__ == "__main__":
    # Example usage
    map_dir = "/home/xiangbog/Folder/Research/SkyLink/skylink/skylink_data/debug/2025_04_06_13_42_44/timestamp_000365/agent_000239/vector_map.json"
    save_path = "/home/xiangbog/Folder/Research/SkyLink/skylink/debug/vector_map.png"
    draw_vector_map(vector_map=None, save_path=save_path, map_dir=map_dir)
    