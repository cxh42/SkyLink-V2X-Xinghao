
import math
import uuid

import cv2
import carla
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon

from skylink_v2x.carla_helper import OBJECT_CONFIG

class MapManager(object):
    
    def __init__(self, carla_world, agent_id, carla_map, config, comm_manager=None):
        self.carla_world = carla_world
        self.agent_id = agent_id
        self.carla_map = carla_map
        self.comm_manager = comm_manager
        self.map_types = getattr(config, 'map_types', [])
        self.center = None

        self.pixel_size = getattr(config, 'pixel_size', 0.5) # meters
        self.resolution = np.array([config['resolution'][0],
                                     config['resolution'][1]])
        self.lane_seg_size = getattr(config, 'lane_seg_size', 0.1) # meters
        self.raster_radius = float(np.linalg.norm(self.resolution * np.array([self.pixel_size, self.pixel_size]))) / 2

        self.topology = [x[0] for x in carla_map.get_topology()]
        # sort by altitude
        # self.topology = sorted(topology, key=lambda w: w.transform.location.z)

        self.lane_info = {}
        self.crosswalk_info = {}
        self.static_object_info = {}
        self.dynamic_object_info = {}
        self.bound_info = {'lanes': {},
                           'crosswalks': {}}

        if hasattr(config, 'online_mapping'):
            self.online_mapping = config['online_mapping']
        else:
            self.online_mapping = False
            print('Warning: Online mapping flag is not provided. It is disabled by default.')
            
        if not self.online_mapping:
            self.load_static_objects(None)
            self.prepare_lane(None)

        # bev maps
        self.seg_bev = dict()
        self.vector_map = None # vector map data
        
    def get_current_map(self, object_detection_results, sensing_data):
        if self.online_mapping:
            self.load_static_objects(object_detection_results)
            self.prepare_lane(sensing_data)
        self.load_dynamic_objects(object_detection_results)
        self.rasterize_dynamic()
        self.rasterize_static()
        
        # TODO
        # return all maps (bev map, vector map, etc) in dict

    def update_info(self, center):
        self.center = center
        
    def filter_object_in_range(self, radius, object_dict):
        objects_in_range = dict()

        # TODO: write helper function
        center = [self.center.location.x, self.center.location.y]

        for oid, object in object_dict.items():
            location = object['location']
            distance = math.sqrt((location[0] - center[0]) ** 2 + \
                                 (location[1] - center[1]) ** 2)
            if distance < radius:
                objects_in_range.update({oid: object})

        return objects_in_range

    def indices_in_bounds(self,
                          bounds: np.ndarray,
                          half_extent: float) -> np.ndarray:
        """
        Get indices of elements for which the bounding box described by bounds
        intersects the one defined around center (square with side 2*half_side)

        Parameters
        ----------
        bounds :np.ndarray
            array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]

        half_extent : float
            half the side of the bounding box centered around center

        Returns
        -------
        np.ndarray: indices of elements inside radius from center
        """
        x_center, y_center = self.center.location.x, self.center.location.y

        x_min_in = x_center > bounds[:, 0, 0] - half_extent
        y_min_in = y_center > bounds[:, 0, 1] - half_extent
        x_max_in = x_center < bounds[:, 1, 0] + half_extent
        y_max_in = y_center < bounds[:, 1, 1] + half_extent
        return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]

    def associate_lane_tl(self, lane):
        associate_tl_id = ''

        for tl_id, tl in self.traffic_light_info.items():
            tl_path = Path(tl['corners'])
            if tl_path.contains_points(lane[:, :2]).any():
                associate_tl_id = tl_id
                break
            
        return associate_tl_id

    def prepare_lane(self, sensing_data):
        """
        From the topology generate all lane and crosswalk
        information in a dictionary under world's coordinate frame.
        """
        if self.online_mapping:
            assert sensing_data is not None
            raise NotImplementedError('Online mapping is not supported in this version.')
        else:
            self.prepare_lane_offline()
            
    def prepare_lane_offline(self):
        
        # list of str
        lanes_id = []

        # loop all waypoints to get lane information
        for idx, waypoint in enumerate(self.topology):
            # unique id for each lane TODO: should we use random number for generating id?
            lane_id = uuid.uuid4().hex[:6].upper()
            lanes_id.append(lane_id)

            waypoints = [waypoint]
            nxt = waypoint.next(self.lane_seg_size)[0]
            # looping until next lane
            while nxt.road_id == waypoint.road_id and nxt.lane_id == waypoint.lane_id:
                waypoints.append(nxt)
                nxt_list = nxt.next(self.lane_seg_size)
                if len(nxt_list) == 0:
                    break
                nxt = nxt_list[0]

            # waypoint is the centerline, we need to calculate left lane mark
            left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for
                            w in waypoints]
            right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for
                             w in waypoints]
            # convert the list of carla.Location to np.array
            left_marking = list_loc2array(left_marking)
            right_marking = list_loc2array(right_marking)
            mid_lane = list_wpt2array(waypoints)

            # get boundary information
            bound = self.get_bounds(left_marking, right_marking)
            lanes_bounds = np.append(lanes_bounds, bound, axis=0)

            # associate with traffic light
            tl_id = self.associate_lane_tl(mid_lane)

            self.lane_info.update({lane_id: {'xyz_left': left_marking,
                                             'xyz_right': right_marking,
                                             'xyz_mid': mid_lane,
                                             'tl_id': tl_id}})
            # boundary information
            self.bound_info['lanes']['ids'] = lanes_id
            self.bound_info['lanes']['bounds'] = lanes_bound

    def generate_lane_area(self, xyz_left, xyz_right):
        """
        Generate the lane area poly under rasterization map's center
        coordinate frame.

        Parameters
        ----------
        xyz_left : np.ndarray
            Left lanemarking of a lane, shape: (n, 3).
        xyz_right : np.ndarray
            Right lanemarking of a lane, shape: (n, 3).

        Returns
        -------
        lane_area : np.ndarray
            Combine left and right lane together to form a polygon.
        """
        lane_area = np.zeros((2, xyz_left.shape[0], 2))
        # convert coordinates to center's coordinate frame
        xyz_left = xyz_left.T
        xyz_left = np.r_[
            xyz_left, [np.ones(xyz_left.shape[1])]]
        xyz_right = xyz_right.T
        xyz_right = np.r_[
            xyz_right, [np.ones(xyz_right.shape[1])]]

        # ego's coordinate frame
        xyz_left = world_to_sensor(xyz_left, self.center).T
        xyz_right = world_to_sensor(xyz_right, self.center).T

        # to image coordinate frame
        lane_area[0] = xyz_left[:, :2]
        lane_area[1] = xyz_right[::-1, :2]
        # switch x and y
        lane_area = lane_area[..., ::-1]
        # y revert
        lane_area[:, :, 1] = -lane_area[:, :, 1]

        lane_area[:, :, 0] = lane_area[:, :, 0] * self.pixels_per_meter + \
            self.raster_size[0] // 2
        lane_area[:, :, 1] = lane_area[:, :, 1] * self.pixels_per_meter + \
            self.raster_size[1] // 2

        # to make more precise polygon
        lane_area = cv2_subpixel(lane_area)

        return lane_area

    def generate_agent_area(self, corners):
        """
        Convert the agent's bbx corners from world coordinates to
        rasterization coordinates.

        Parameters
        ----------
        corners : list
            The four corners of the agent's bbx under world coordinate.

        Returns
        -------
        agent four corners in image.
        """
        # (4, 3) numpy array
        corners = np.array(corners)
        # for homogeneous transformation
        corners = corners.T
        corners = np.r_[
            corners, [np.ones(corners.shape[1])]]
        # convert to ego's coordinate frame
        corners = world_to_sensor(corners, self.center).T
        corners = corners[:, :2]

        # switch x and y
        corners = corners[..., ::-1]
        # y revert
        corners[:, 1] = -corners[:, 1]

        corners[:, 0] = corners[:, 0] * self.pixels_per_meter + \
            self.raster_size[0] // 2
        corners[:, 1] = corners[:, 1] * self.pixels_per_meter + \
            self.raster_size[1] // 2

        # to make more precise polygon
        corner_area = cv2_subpixel(corners[:, :2])

        return corner_area
    
    def load_dynamic_objects_offline(self):
        obj_list = self.carla_world.get_actors()
        
        for obj in obj_list:
            oid = obj.id
            obj_class_config = OBJECT_CONFIG.get_by_blueprint(obj.type_id)
            obj_transform = obj.get_transform()
            obj_yaw = obj_transform.rotation.yaw
            
            if OBJECT_CONFIG.is_dynamic(obj_class_config.class_idx):
                            
                bb = obj.bounding_box.extent
                corners = [
                    carla.Location(x=-bb.x, y=-bb.y),
                    carla.Location(x=-bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=-bb.y)
                ]
                obj_transform.transform(corners)
                corners_reformat = [[x.x, x.y, x.z] for x in corners]

                self.dynamic_object_info.update(
                    {oid: {
                        'object': obj,  # carla.Actor
                        'class_config': obj_class_config, # ClassConfig
                        'location': obj_transform.location, # carla.Location
                        'yaw': obj_yaw, # float
                        'corners': corners_reformat, # list
                    }})

    def load_dynamic_objects(self, object_detection_results):
        """
        Load all the dynamic agents info from the server directly
        into a dictionary, including vehicles, pedestrians, and other dynamic agents.

        Returns
        -------
        The dictionary contains all agents info in the CARLA world.
        """
        if self.online_mapping:
            assert object_detection_results is not None
            raise NotImplementedError('Online mapping is not supported in this version.')
        else:
            self.load_dynamic_objects_offline()

                
    def load_static_objects(self, object_detection_results):
        """
        Load all the static agents info from the server directly
        into a dictionary, including buildings, traffic lights, and other static agents.

        Returns
        -------
        The dictionary contains all static agents info in the CARLA world.
        """
        
        if self.online_mapping:
            assert object_detection_results is not None
            raise NotImplementedError('Online mapping is not supported in this version.')
        else:
            self.load_static_objects_offline()
                

    def load_static_objects_offline(self):
        """
        Load all the static agents info from the server directly
        into a dictionary, including buildings, traffic lights, and other static agents.

        Returns
        -------
        The dictionary contains all static agents info in the CARLA world.
        """
        obj_list = self.carla_world.get_actors()
        
        for obj in obj_list:
            oid = obj.id
            obj_class_config = OBJECT_CONFIG.get_by_blueprint(obj.type_id)
            obj_transform = obj.get_transform()
            obj_yaw = obj_transform.rotation.yaw
            
            # Traffic light and traffic signal
            # TODO: need to includ RSU
            if OBJECT_CONFIG.is_static(obj_class_config.class_idx):
                # TODO: Is this a good way of applying object id?
                oid = uuid.uuid4().hex[:4].upper()
                # Effective area of the static object
                eff_loc = obj_transform.transform(
                    obj.trigger_volume.location)
                area_transform = carla.Transform(eff_loc, carla.Rotation(yaw=obj_yaw))
                
                ext = obj.trigger_volume.extent
                # enlarge the y axis
                ext.y += 0.5
                ext_corners = np.array([
                    [-ext.x, -ext.y],
                    [ext.x, -ext.y],
                    [ext.x, ext.y],
                    [-ext.x, ext.y]])
                for i in range(ext_corners.shape[0]):
                    corrected_loc = area_transform.transform(
                        carla.Location(ext_corners[i][0], ext_corners[i][1]))
                    ext_corners[i, 0] = corrected_loc.x
                    ext_corners[i, 1] = corrected_loc.y
                
                corner_poly = Polygon(ext_corners)
                self.static_object_info.update(
                    {oid: {'object': obj, # carla.Actor
                        'class_config': obj_class_config, # ClassConfig
                        'location': eff_loc, # carla.Location
                        'yaw': obj_yaw, # float
                        'corners': corner_poly, # shapely.geometry.Polygon
                        }})
                
            
    def rasterize_dynamic(self):
        """
        Rasterize the dynamic agents.

        Returns
        -------
        Rasterization image.
        """
        self.dynamic_bev = 255 * np.zeros(
            shape=(self.resolution[1], self.resolution[0], 3),
            dtype=np.uint8)

        objects_in_range = self.filter_object_in_range(self.raster_radius, self.dynamic_object_info)

        corner_list = []
        for oid, object in objects_in_range.items():
            agent_corner = self.generate_agent_area(object['corners'])
            corner_list.append(
                {'corner': agent_corner, 'class_idx': object['class_idx']}
            )
        self.dynamic_bev = draw_agent(corner_list, self.dynamic_bev)

    def rasterize_static(self):
        """
        Generate the static bev map.
        """
        self.static_bev = 255 * np.ones(
            shape=(self.raster_size[1], self.raster_size[0], 3),
            dtype=np.uint8)
        self.vis_bev = 255 * np.ones(
            shape=(self.raster_size[1], self.raster_size[0], 3),
            dtype=np.uint8)

        lane_indices = self.indices_in_bounds(
            self.bound_info['lanes']['bounds'], 
            self.raster_radius)
        lanes_area_list = []
        lane_type_list = []

        for idx, lane_idx in enumerate(lane_indices):
            lane_idx = self.bound_info['lanes']['ids'][lane_idx]
            lane_info = self.lane_info[lane_idx]
            xyz_left, xyz_right = \
                lane_info['xyz_left'], lane_info['xyz_right']

            # generate lane area
            lane_area = self.generate_lane_area(xyz_left, xyz_right)
            lanes_area_list.append(lane_area)

            # check the associated traffic light
            associated_tl_id = lane_info['tl_id']
            if associated_tl_id:
                tl_actor = self.static_object_info[associated_tl_id]['object']
                status = convert_tl_status(tl_actor.get_state())
                lane_type_list.append(status)
            else:
                lane_type_list.append('normal')

        self.static_bev = draw_road(lanes_area_list,
                                    self.static_bev)
        self.static_bev = draw_lane(lanes_area_list, lane_type_list,
                                    self.static_bev)

        self.vis_bev = draw_road(lanes_area_list,
                                 self.vis_bev)
        self.vis_bev = draw_lane(lanes_area_list, lane_type_list,
                                 self.vis_bev)
        self.vis_bev = cv2.cvtColor(self.vis_bev, cv2.COLOR_RGB2BGR)

    def destroy(self):
        cv2.destroyAllWindows()

