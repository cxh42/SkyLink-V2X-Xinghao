"""
Map Manager for CARLA environments.
Handles lane information, static and dynamic objects, and BEV maps.
"""
import math
import uuid
import time

import cv2
import carla
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon

from skylink_v2x.carla_helper import OBJECT_CONFIG
from skylink_v2x.core.autonomy_stack.mapping.helpers.coordinate_transforms import (
    lateral_shift, list_loc2array, list_wpt2array, world_to_sensor, cv2_subpixel
)
from skylink_v2x.core.autonomy_stack.mapping.helpers.vector_map import calculate_lane_heading, calculate_lane_curvature
from skylink_v2x.core.autonomy_stack.mapping.helpers.visualization import draw_agent, draw_road, draw_lane, draw_agent_layers
from skylink_v2x.core.autonomy_stack.mapping.helpers.utils import convert_tl_status, get_bounds

class MapManager:
    """
    Manager for map-related operations including lane detection, 
    object tracking, and BEV visualization.
    """
    
    def __init__(self, carla_world, agent_id, carla_map, config, comm_manager=None):
        """Initialize the map manager with configuration settings."""
        self.carla_world = carla_world
        self.agent_id = agent_id
        self.carla_map = carla_map
        self._comm_manager = comm_manager
        self.map_types = getattr(config, 'map_types', [])
        self.center = None

        self.pixel_size = getattr(config, 'pixel_size', 0.5)  # meters
        self.pixels_per_meter = 1.0 / self.pixel_size
        self.resolution = np.array([config['resolution'][0], config['resolution'][1]])
        self.raster_size = (self.resolution[0], self.resolution[1])
        self.lane_seg_size = getattr(config, 'lane_seg_size', 0.1)  # meters
        self.raster_radius = float(np.linalg.norm(self.resolution * np.array([self.pixel_size, self.pixel_size]))) / 2

        self.topology = [x[0] for x in carla_map.get_topology()]

        # Initialize data containers
        self.lane_info = {}
        self.crosswalk_info = {}
        self.static_object_info = {}
        self.traffic_light_info = {}
        self.dynamic_object_info = {}
        self.bound_info = {
            'lanes': {'ids': [], 'bounds': np.empty((0, 2, 2))},
            'crosswalks': {}
        }

        # Check for online mapping flag
        if hasattr(config, 'online_mapping'):
            self.online_mapping = config['online_mapping']
        else:
            self.online_mapping = False
            print('Warning: Online mapping flag is not provided. It is disabled by default.')
            
        if not self.online_mapping:
            self.load_static_objects(None)
            self.prepare_lane(None)

        # BEV maps
        self.seg_bev = dict()
        self.vector_map = None
        self.static_bev = None
        self.dynamic_bev = None
        self.vis_bev = None
        self.dynamic_bev_layers = None
        
    def get_current_map(self, 
                        localization_data=None,
                        perception_data=None, 
                        sensing_data=None):
        """Update and return current map data."""
        self.update_info(center=localization_data[self.agent_id]['position'])
        if self.online_mapping:
            self.load_static_objects(perception_data)
        # self.prepare_lane(sensing_data)
            
        self.load_dynamic_objects(perception_data)
        
        # Generate vector map before rasterization
        if 'vector' in self.map_types:
            self.generate_vector_map()
        
        # Generate raster maps
        if 'seg' in self.map_types:
            self.rasterize_static()
            self.rasterize_dynamic()
        
        
        # Return all map data
        current_map = {
            'static_bev': self.static_bev,
            'dynamic_bev': self.dynamic_bev,
            'dynamic_bev_layers': self.dynamic_bev_layers,
            'vis_bev': self.vis_bev,
            'vector_map': self.vector_map
        }
        self._comm_manager.buffer_mapping(self.agent_id, current_map)
        self.clean()
        
        return current_map
    
    def clean(self):
        self.lane_info = {}
        self.crosswalk_info = {}
        self.static_object_info = {}
        self.traffic_light_info = {}
        self.dynamic_object_info = {}
        self.bound_info = {
            'lanes': {'ids': [], 'bounds': np.empty((0, 2, 2))},
            'crosswalks': {}
        }
        self.seg_bev = {}
        self.vector_map = None
        self.static_bev = None
        self.dynamic_bev = None
        self.vis_bev = None
        self.dynamic_bev_layers = None


    def update_info(self, 
                    center=None):
        """Update the center transform for map generation."""
        self.center = center
            
        
    def filter_object_in_range(self, radius, object_dict):
        """Filter objects within specified radius of the center."""
        objects_in_range = {}
        
        if self.center is None:
            return objects_in_range

        center = [self.center.location.x, self.center.location.y]

        for oid, obj in object_dict.items():
            location = obj['location']
            if isinstance(location, carla.Location):
                location = [location.x, location.y]
            
            distance = math.sqrt((location[0] - center[0])**2 + (location[1] - center[1])**2)
            if distance < radius:
                objects_in_range[oid] = obj

        return objects_in_range

    def indices_in_bounds(self, bounds, half_extent):
        """Get indices of elements whose bounds intersect with center region."""
        if self.center is None or bounds.shape[0] == 0:
            return np.array([], dtype=np.int32)

        x_center, y_center = self.center.location.x, self.center.location.y

        x_min_in = x_center > bounds[:, 0, 0] - half_extent
        y_min_in = y_center > bounds[:, 0, 1] - half_extent
        x_max_in = x_center < bounds[:, 1, 0] + half_extent
        y_max_in = y_center < bounds[:, 1, 1] + half_extent
        
        return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]

    def associate_lane_tl(self, lane):
        """Associate a lane with a traffic light."""
        associate_tl_id = ''

        for tl_id, tl in self.traffic_light_info.items():
            tl_path = Path(tl['corners'])
            if tl_path.contains_points(lane[:, :2]).any():
                associate_tl_id = tl_id
                break
            
        return associate_tl_id

    def prepare_lane(self, sensing_data):
        """Process lane information from topology or sensing data."""
        if self.online_mapping:
            assert sensing_data is not None
            raise NotImplementedError('Online mapping is not supported in this version.')
        else:
            self.prepare_lane_offline()
            
    def prepare_lane_offline(self):
        """Process lane information from CARLA topology."""
        # Reset lane info
        self.lane_info = {}
        
        # Reset bounds
        lanes_id = []
        lanes_bound = np.empty((0, 2, 2))

        # Process all waypoints in topology
        for waypoint in self.topology:
            # Generate unique lane ID
            lane_id = uuid.uuid4().hex[:6].upper()
            lanes_id.append(lane_id)

            # Collect waypoints for this lane segment
            waypoints = [waypoint]
            nxt = waypoint.next(self.lane_seg_size)[0]
            while nxt.road_id == waypoint.road_id and nxt.lane_id == waypoint.lane_id:
                waypoints.append(nxt)
                nxt_list = nxt.next(self.lane_seg_size)
                if len(nxt_list) == 0:
                    break
                nxt = nxt_list[0]

            # Calculate lane boundaries
            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
            
            # Convert to numpy arrays
            left_marking = list_loc2array(left_marking)
            right_marking = list_loc2array(right_marking)
            mid_lane = list_wpt2array(waypoints)

            # Get boundary information
            bound = get_bounds(left_marking, right_marking)
            lanes_bound = np.append(lanes_bound, bound, axis=0)

            # Associate with traffic light
            tl_id = self.associate_lane_tl(mid_lane)

            # Store lane information
            self.lane_info[lane_id] = {
                'xyz_left': left_marking,
                'xyz_right': right_marking,
                'xyz_mid': mid_lane,
                'tl_id': tl_id
            }
            
        # Update boundary information
        self.bound_info['lanes']['ids'] = lanes_id
        self.bound_info['lanes']['bounds'] = lanes_bound

    def generate_lane_area(self, xyz_left, xyz_right):
        """Generate lane area polygon in raster coordinates."""
        if self.center is None:
            return np.zeros((0, 1, 2))

        # Initialize lane area array
        lane_area = np.zeros((2, xyz_left.shape[0], 2))
        
        # Convert coordinates to center's coordinate frame
        xyz_left = xyz_left.T
        xyz_left = np.r_[xyz_left, [np.ones(xyz_left.shape[1])]]
        xyz_right = xyz_right.T
        xyz_right = np.r_[xyz_right, [np.ones(xyz_right.shape[1])]]

        # Transform to ego's coordinate frame
        xyz_left = world_to_sensor(xyz_left, self.center).T
        xyz_right = world_to_sensor(xyz_right, self.center).T

        # Prepare for image coordinate frame
        lane_area[0] = xyz_left[:, :2]
        lane_area[1] = xyz_right[::-1, :2]
        
        # Switch x and y, revert y
        lane_area = lane_area[..., ::-1]
        lane_area[:, :, 1] = -lane_area[:, :, 1]

        # Scale to pixels and offset to image center
        lane_area[:, :, 0] = lane_area[:, :, 0] * self.pixels_per_meter + self.raster_size[0] // 2
        lane_area[:, :, 1] = lane_area[:, :, 1] * self.pixels_per_meter + self.raster_size[1] // 2

        # Format for precise polygon drawing
        return cv2_subpixel(lane_area)

    def generate_agent_area(self, corners):
        """Convert agent bounding box corners to raster coordinates."""
        if self.center is None or not corners:
            return np.zeros((0, 1, 2))

        # Convert to numpy array and prepare for transformation
        corners = np.array(corners)
        corners = corners.T
        corners = np.r_[corners, [np.ones(corners.shape[1])]]
        
        # Transform to ego coordinate frame
        corners = world_to_sensor(corners, self.center).T
        corners = corners[:, :2]

        # Switch x and y, revert y
        corners = corners[..., ::-1]
        corners[:, 1] = -corners[:, 1]

        # Scale to pixels and offset to image center
        corners[:, 0] = corners[:, 0] * self.pixels_per_meter + self.raster_size[0] // 2
        corners[:, 1] = corners[:, 1] * self.pixels_per_meter + self.raster_size[1] // 2

        # Format for precise polygon drawing
        return cv2_subpixel(corners[:, :2])
    
    def load_dynamic_objects_offline(self):
        """Load dynamic objects from CARLA world."""
        # Clear previous dynamic object info
        
        # Get all actors
        obj_list = self.carla_world.get_actors()
        
        for obj in obj_list:
            try:
                oid = obj.id
                obj_class_config = OBJECT_CONFIG.get_by_blueprint(obj.type_id)
                
                if obj_class_config and (OBJECT_CONFIG.is_dynamic(obj_class_config.class_idx) \
                    or OBJECT_CONFIG.is_static(obj_class_config.class_idx)):
                    obj_transform = obj.get_transform()
                    obj_yaw = obj_transform.rotation.yaw
                    
                    # Get bounding box corners
                    bb = obj.bounding_box.extent
                    corners = [
                        carla.Location(x=-bb.x, y=-bb.y),
                        carla.Location(x=-bb.x, y=bb.y),
                        carla.Location(x=bb.x, y=bb.y),
                        carla.Location(x=bb.x, y=-bb.y)
                    ]
                    corners = [obj_transform.transform(corner) for corner in corners]
                    corners_reformat = [[x.x, x.y, x.z] for x in corners]

                    self.dynamic_object_info[oid] = {
                        'object': obj,
                        'class_config': obj_class_config,
                        'class_idx': obj_class_config.class_idx,
                        'location': obj_transform.location,
                        'yaw': obj_yaw,
                        'corners': corners_reformat
                    }
            except Exception as e:
                raise Exception(f"Error processing dynamic object {obj.type_id}: {e}")

    def load_dynamic_objects(self, object_detection_results):
        """Load dynamic objects based on detection results or from CARLA."""
        if self.online_mapping:
            assert object_detection_results is not None
            raise NotImplementedError('Online mapping is not supported in this version.')
        else:
            self.load_dynamic_objects_offline()
                
    def load_static_objects(self, object_detection_results):
        """Load static objects based on detection results or from CARLA."""
        if self.online_mapping:
            assert object_detection_results is not None
            raise NotImplementedError('Online mapping is not supported in this version.')
        else:
            self.load_static_objects_offline()
                
    def load_static_objects_offline(self):
        """Load static objects from CARLA world."""
        # Clear previous static object info
        self.static_object_info = {}
        self.traffic_light_info = {}
        
        # Get all actors
        obj_list = self.carla_world.get_actors()
        
        for obj in obj_list:
            try:
                obj_class_config = OBJECT_CONFIG.get_by_blueprint(obj.type_id)
                
                if obj_class_config and OBJECT_CONFIG.is_static(obj_class_config.class_idx):
                    obj_transform = obj.get_transform()
                    obj_yaw = obj_transform.rotation.yaw
                    
                    # Use UUID for static object IDs
                    oid = uuid.uuid4().hex[:4].upper()
                    
                    # Process trigger volume for effective area
                    if hasattr(obj, 'trigger_volume'):
                        eff_loc = obj_transform.transform(obj.trigger_volume.location)
                        area_transform = carla.Transform(eff_loc, carla.Rotation(yaw=obj_yaw))
                        
                        ext = obj.trigger_volume.extent
                        # Enlarge the y axis
                        ext.y += 0.5
                        ext_corners = np.array([
                            [-ext.x, -ext.y],
                            [ext.x, -ext.y],
                            [ext.x, ext.y],
                            [-ext.x, ext.y]
                        ])
                        
                        # Transform corners to world coordinates
                        corners_world = []
                        for i in range(ext_corners.shape[0]):
                            corner_loc = area_transform.transform(
                                carla.Location(ext_corners[i][0], ext_corners[i][1])
                            )
                            corners_world.append([corner_loc.x, corner_loc.y])
                        
                        # Create polygon from corners
                        corner_poly = Polygon(corners_world)
                        
                        # Create object info
                        obj_info = {
                            'object': obj,
                            'class_config': obj_class_config,
                            'location': eff_loc,
                            'yaw': obj_yaw,
                            'corners': corner_poly
                        }
                        
                        # Store traffic lights separately
                        if obj_class_config.class_idx == 6:  # Traffic light
                            self.traffic_light_info[oid] = obj_info
                        
                        self.static_object_info[oid] = obj_info
            except Exception as e:
                print(f"Error processing static object {obj.type_id}: {e}")
            
    def rasterize_dynamic(self):
        """Generate BEV map with dynamic objects."""
        if self.center is None:
            self.dynamic_bev = None
            return
        
        if self.vis_bev is None:
            self.vis_bev = 255 * np.zeros(
                shape=(self.resolution[1], self.resolution[0], 3),
                dtype=np.uint8
            )
            
        # Create empty image
        if self.dynamic_bev is None:
            self.dynamic_bev = 255 * np.zeros(
                shape=(self.resolution[1], self.resolution[0], 3),
                dtype=np.uint8
            )

        # Filter objects in range
        objects_in_range = self.filter_object_in_range(self.raster_radius, self.dynamic_object_info)

        # Generate corners for each object
        corner_list = []
        for obj in objects_in_range.values():
            agent_corner = self.generate_agent_area(obj['corners'])
            if len(agent_corner) > 0:
                corner_list.append({
                    'corner': agent_corner,
                    'class_idx': obj['class_idx']
                })
        
        # Draw objects on the map
        if corner_list:
            self.dynamic_bev = draw_agent(corner_list, self.dynamic_bev)
            self.vis_bev = draw_agent(corner_list, self.vis_bev)
            self.dynamic_bev_layers = draw_agent_layers(corner_list, self.resolution[1], self.resolution[0])

    def rasterize_static(self):
        """Generate BEV map with static elements (roads, lanes)."""
        if self.center is None:
            self.static_bev = None
            self.vis_bev = None
            return
            
        # Create empty images
        
        if self.static_bev is None:
            self.static_bev = 255 * np.zeros(
                shape=(self.raster_size[1], self.raster_size[0], 3),
                dtype=np.uint8
            )
        
        if self.vis_bev is None:
            self.vis_bev = 255 * np.zeros(
                shape=(self.raster_size[1], self.raster_size[0], 3),
                dtype=np.uint8
            )

        # Get lane indices within bounds
        lane_indices = self.indices_in_bounds(self.bound_info['lanes']['bounds'], self.raster_radius)
        
        # Process each lane
        lanes_area_list = []
        lane_type_list = []

        for lane_idx in lane_indices:
            lane_id = self.bound_info['lanes']['ids'][lane_idx]
            lane_info = self.lane_info[lane_id]
            xyz_left, xyz_right = lane_info['xyz_left'], lane_info['xyz_right']

            # Generate lane area in raster coordinates
            lane_area = self.generate_lane_area(xyz_left, xyz_right)
            if len(lane_area) > 0:
                lanes_area_list.append(lane_area)

                # Check for associated traffic light
                associated_tl_id = lane_info['tl_id']
                if associated_tl_id and associated_tl_id in self.traffic_light_info:
                    tl_actor = self.traffic_light_info[associated_tl_id]['object']
                    status = convert_tl_status(tl_actor.get_state())
                    lane_type_list.append(status)
                else:
                    lane_type_list.append('normal')

        # Draw lanes if any
        if lanes_area_list:
            self.static_bev = draw_road(lanes_area_list, self.static_bev)
            self.static_bev = draw_lane(lanes_area_list, lane_type_list, self.static_bev)

            self.vis_bev = draw_road(lanes_area_list, self.vis_bev)
            self.vis_bev = draw_lane(lanes_area_list, lane_type_list, self.vis_bev)
            # self.vis_bev = cv2.cvtColor(self.vis_bev, cv2.COLOR_RGB2BGR)
            
        
            
            
    def generate_vector_map(self):
        """
        Generate a structured vector map representation of the surrounding area.
        The vector map includes lanes, intersections, and traffic signals in a format 
        suitable for path planning and navigation.
        """
        if self.center is None:
            self.vector_map = None
            return None
            
        vector_map = {
            'lanes': [],
            'traffic_lights': [],
            'intersections': [],
            'timestamp': time.time(),
            'ego_position': [
                self.center.location.x,
                self.center.location.y,
                self.center.location.z
            ],
            'ego_rotation': [
                self.center.rotation.roll,
                self.center.rotation.pitch,
                self.center.rotation.yaw
            ]
        }
        
        # Filter lanes in range
        lanes_in_range = self.filter_lanes_in_range(self.raster_radius)
        
        # Process lanes
        for lane_id, lane_info in lanes_in_range.items():
            # Extract lane centerline and boundaries
            centerline = lane_info['xyz_mid']
            left_boundary = lane_info['xyz_left']
            right_boundary = lane_info['xyz_right']
            
            # Calculate additional lane properties
            headings = calculate_lane_heading(centerline)
            curvatures = calculate_lane_curvature(centerline, headings)
            
            # Get lane connections
            predecessors, successors = self.get_lane_connections(lane_id)
            
            # Get traffic light association
            tl_id = lane_info.get('tl_id', '')
            
            # Create lane object
            lane_object = {
                'id': lane_id,
                'centerline': centerline.tolist(),
                'left_boundary': left_boundary.tolist(),
                'right_boundary': right_boundary.tolist(),
                'headings': headings,
                'curvatures': curvatures,
                'predecessors': predecessors,
                'successors': successors,
                'traffic_light_id': tl_id,
                'speed_limit': 30.0  # Default value - would be from CARLA in real implementation
            }
            
            vector_map['lanes'].append(lane_object)
        
        # Process traffic lights
        traffic_lights_in_range = self.filter_object_in_range(self.raster_radius, self.traffic_light_info)
        for tl_id, tl_info in traffic_lights_in_range.items():
            location = tl_info['location']
            
            tl_object = {
                'id': tl_id,
                'position': [location.x, location.y, location.z],
                'yaw': tl_info['yaw'],
                'state': self.get_traffic_light_state(tl_id),
                'controlled_lanes': self.get_controlled_lanes(tl_id)
            }
            vector_map['traffic_lights'].append(tl_object)
        
        # Process intersections (identified by connected lanes)
        intersection_lanes = self.identify_intersection_lanes(lanes_in_range)
        for intersection_id, lane_ids in intersection_lanes.items():
            intersection_object = {
                'id': intersection_id,
                'lanes': lane_ids,
                'traffic_lights': self.get_intersection_traffic_lights(lane_ids)
            }
            vector_map['intersections'].append(intersection_object)
        
        self.vector_map = vector_map
        return vector_map

    def filter_lanes_in_range(self, radius):
        """Filter lanes that are within the specified radius of the ego vehicle."""
        lanes_in_range = {}
        
        if not self.bound_info['lanes']['ids']:
            return lanes_in_range
            
        lane_indices = self.indices_in_bounds(
            self.bound_info['lanes']['bounds'], 
            radius
        )
        
        for lane_idx in lane_indices:
            if lane_idx < len(self.bound_info['lanes']['ids']):
                lane_id = self.bound_info['lanes']['ids'][lane_idx]
                if lane_id in self.lane_info:
                    lanes_in_range[lane_id] = self.lane_info[lane_id]
        
        return lanes_in_range

    def get_lane_connections(self, lane_id):
        """
        Get the predecessor and successor lanes of the given lane.
        This is a simplified implementation that finds connections based on proximity.
        """
        if lane_id not in self.lane_info:
            return [], []
            
        predecessors = []
        successors = []
        
        lane_start = self.lane_info[lane_id]['xyz_mid'][0][:2]  # First point (x,y)
        lane_end = self.lane_info[lane_id]['xyz_mid'][-1][:2]   # Last point (x,y)
        
        # Get lane direction at start and end
        if len(self.lane_info[lane_id]['xyz_mid']) > 1:
            start_dir = self.lane_info[lane_id]['xyz_mid'][1][:2] - lane_start
            start_dir = start_dir / np.linalg.norm(start_dir) if np.linalg.norm(start_dir) > 0 else np.array([0, 0])
            
            end_dir = lane_end - self.lane_info[lane_id]['xyz_mid'][-2][:2]
            end_dir = end_dir / np.linalg.norm(end_dir) if np.linalg.norm(end_dir) > 0 else np.array([0, 0])
        else:
            start_dir = np.array([0, 0])
            end_dir = np.array([0, 0])
        
        # Distance threshold for connecting lanes
        connect_threshold = 5.0  # meters
        
        # Search for connecting lanes
        for other_id, other_info in self.lane_info.items():
            if other_id == lane_id or len(other_info['xyz_mid']) < 2:
                continue
                
            other_start = other_info['xyz_mid'][0][:2]
            other_end = other_info['xyz_mid'][-1][:2]
            
            # Calculate directions
            other_start_dir = other_info['xyz_mid'][1][:2] - other_start
            other_start_dir = other_start_dir / np.linalg.norm(other_start_dir) if np.linalg.norm(other_start_dir) > 0 else np.array([0, 0])
            
            other_end_dir = other_end - other_info['xyz_mid'][-2][:2]
            other_end_dir = other_end_dir / np.linalg.norm(other_end_dir) if np.linalg.norm(other_end_dir) > 0 else np.array([0, 0])
            
            # Check for successor: if this lane's end is close to other lane's start
            end_to_start_dist = np.linalg.norm(lane_end - other_start)
            if end_to_start_dist < connect_threshold:
                # Check if directions align
                direction_alignment = np.dot(end_dir, other_start_dir)
                if direction_alignment > 0.7:  # Cosine similarity threshold
                    successors.append(other_id)
            
            # Check for predecessor: if this lane's start is close to other lane's end
            start_to_end_dist = np.linalg.norm(lane_start - other_end)
            if start_to_end_dist < connect_threshold:
                # Check if directions align
                direction_alignment = np.dot(start_dir, other_end_dir)
                if direction_alignment > 0.7:  # Cosine similarity threshold
                    predecessors.append(other_id)
        
        return predecessors, successors

    def get_traffic_light_state(self, tl_id):
        """Get the current state of a traffic light."""
        if tl_id in self.traffic_light_info:
            tl_actor = self.traffic_light_info[tl_id]['object']
            return convert_tl_status(tl_actor.get_state())
        return "unknown"

    def get_controlled_lanes(self, tl_id):
        """Get the lanes controlled by a traffic light."""
        controlled_lanes = []
        
        for lane_id, lane_info in self.lane_info.items():
            if lane_info.get('tl_id', '') == tl_id:
                controlled_lanes.append(lane_id)
        
        return controlled_lanes

    def identify_intersection_lanes(self, lanes_in_range):
        """
        Identify intersections by finding connected lane groups.
        Returns a dictionary mapping intersection IDs to lists of lane IDs.
        """
        intersection_lanes = {}
        
        # A lane is part of an intersection if it has multiple predecessors or successors
        for lane_id, lane_info in lanes_in_range.items():
            predecessors, successors = self.get_lane_connections(lane_id)
            
            if len(predecessors) > 1 or len(successors) > 1:
                # This is an intersection lane
                intersection_id = f"intersection_{uuid.uuid4().hex[:4]}"
                
                if intersection_id not in intersection_lanes:
                    intersection_lanes[intersection_id] = []
                
                intersection_lanes[intersection_id].append(lane_id)
                
                # Also add connected lanes
                for pred_id in predecessors:
                    if pred_id in lanes_in_range:
                        intersection_lanes[intersection_id].append(pred_id)
                
                for succ_id in successors:
                    if succ_id in lanes_in_range:
                        intersection_lanes[intersection_id].append(succ_id)
        
        # Remove duplicates from each intersection's lane list
        for intersection_id in intersection_lanes:
            intersection_lanes[intersection_id] = list(set(intersection_lanes[intersection_id]))
        
        return intersection_lanes

    def get_intersection_traffic_lights(self, lane_ids):
        """Get traffic lights controlling lanes in an intersection."""
        traffic_lights = []
        
        for lane_id in lane_ids:
            if lane_id in self.lane_info:
                tl_id = self.lane_info[lane_id].get('tl_id', '')
                if tl_id and tl_id not in traffic_lights:
                    traffic_lights.append(tl_id)
        
        return traffic_lights

    def destroy(self):
        """Clean up resources."""
        cv2.destroyAllWindows()

    def remove(self):
        """
        清理地图管理器资源
        """
        # 清理地图表面
        if hasattr(self, '_map_surface'):
            self._map_surface = None
        
        # 清理缓存的路径点
        if hasattr(self, '_waypoints_buffer'):
            self._waypoints_buffer.clear()
        
        # 清理缓存的规划路径
        if hasattr(self, '_path_buffer'):
            self._path_buffer.clear()
        
        # 清理其他可能的大型数据结构
        if hasattr(self, '_dynamic_objects'):
            self._dynamic_objects.clear()