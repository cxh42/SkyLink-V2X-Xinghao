# Author: Fengze Yang <fred.yang@utah.edu>
# License: 

import numpy as np
from matplotlib import cm


class SensorTransformer:
    """
    A utility class for coordinate transformations between world and sensor frames.
    Works with sensor data dictionaries rather than direct sensor objects.
    """
    VIRIDIS_COLORMAP = np.array(cm.get_cmap('viridis').colors)
    INTENSITY_SCALE = np.linspace(0.0, 1.0, VIRIDIS_COLORMAP.shape[0])

    # -------------------------------
    # Camera Intrinsics and Bounding Boxes
    # -------------------------------

    @staticmethod
    def compute_camera_intrinsics(camera_data):
        """
        Compute the intrinsic matrix for a given camera sensor data.

        Parameters
        ----------
        camera_data : dict
            The camera sensor data dictionary containing 'image_size_x', 'image_size_y', and 'fov'

        Returns
        -------
        K : np.ndarray
            A 3x3 intrinsic matrix.
        """
        width = int(camera_data['image_size_x'])
        height = int(camera_data['image_size_y'])
        fov = float(camera_data['fov'])
        K = np.eye(3)
        K[0, 2] = width / 2.0
        K[1, 2] = height / 2.0
        focal_length = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        K[0, 0] = focal_length
        K[1, 1] = focal_length

        return K
    
    @staticmethod
    def compute_bbox_vertices(vehicle):
        """
        Compute the eight vertices of the vehicle's bounding box in homogeneous coordinates.

        Parameters
        ----------
        vehicle : object
            A vehicle instance with a 'bounding_box' attribute.

        Returns
        -------
        vertices : np.ndarray
            An (8,4) array of bounding box vertices.
        """
        ext = vehicle.bounding_box.extent

        verts = np.zeros((8, 4))
        verts[0, :] = np.array([ ext.x,  ext.y, -ext.z, 1])
        verts[1, :] = np.array([-ext.x,  ext.y, -ext.z, 1])
        verts[2, :] = np.array([-ext.x, -ext.y, -ext.z, 1])
        verts[3, :] = np.array([ ext.x, -ext.y, -ext.z, 1])
        verts[4, :] = np.array([ ext.x,  ext.y,  ext.z, 1])
        verts[5, :] = np.array([-ext.x,  ext.y,  ext.z, 1])
        verts[6, :] = np.array([-ext.x, -ext.y,  ext.z, 1])
        verts[7, :] = np.array([ ext.x, -ext.y,  ext.z, 1])

        return verts
    
    @staticmethod
    def project_bbox3d_to_bbox2d(bbox3d):
        """
        Project a 3D bounding box (8 vertices) to a 2D bounding box (two corners).

        Parameters
        ----------
        bbox3d : np.ndarray
            An (8,3) array representing the projected 3D bounding box.

        Returns
        -------
        bbox2d : np.ndarray
            A (2,2) array containing the top-left and bottom-right corners.
        """
        min_x, min_y = np.min(bbox3d[:, 0]), np.min(bbox3d[:, 1])
        max_x, max_y = np.max(bbox3d[:, 0]), np.max(bbox3d[:, 1])

        return np.array([[min_x, min_y], [max_x, max_y]])
    
    @staticmethod
    def compute_2d_bbox(vehicle, camera_data):
        """
        Compute the 2D bounding box of a vehicle in a camera image.

        Parameters
        ----------
        vehicle : object
            The vehicle instance.
        camera_data : dict
            The camera sensor data dictionary.

        Returns
        -------
        bbox2d : np.ndarray
            A (2,2) array representing the bounding box corners.
        """
        bbox3d = SensorTransformer.compute_sensor_bbox(vehicle, camera_data)

        return SensorTransformer.project_bbox3d_to_bbox2d(bbox3d)

    @staticmethod
    def compute_sensor_bbox(vehicle, camera_data):
        """
        Compute the vehicle's bounding box as seen by a camera sensor.

        Parameters
        ----------
        vehicle : object
            The vehicle instance.
        camera_data : dict
            The camera sensor data dictionary.

        Returns
        -------
        sensor_bbox : np.ndarray
            An (8,3) array of projected 3D points in image space.
        """
        K = SensorTransformer.compute_camera_intrinsics(camera_data)
        bbox_vertices = SensorTransformer.compute_bbox_vertices(vehicle)
        sensor_coords = SensorTransformer.vehicle_bbox_to_sensor(bbox_vertices, vehicle, camera_data['cords'])
        sensor_coords = sensor_coords[:3, :]  # Remove homogeneous coordinate.

        # Reorder coordinates to match OpenCV conventions.
        reordered = np.vstack([sensor_coords[1, :], -sensor_coords[2, :], sensor_coords[0, :]])
        proj = K @ reordered
        proj = proj.T
        proj[:, 0] /= proj[:, 2]
        proj[:, 1] /= proj[:, 2]

        return np.hstack([proj[:, 0:1], proj[:, 1:2], proj[:, 2:3]])
    
    @staticmethod
    def vehicle_bbox_to_sensor(vertices, vehicle, sensor_cords):
        """
        Transform vehicle bounding box vertices from vehicle frame to sensor frame.

        Parameters
        ----------
        vertices : np.ndarray
            (8,4) vertices in the vehicle frame.
        vehicle : object
            The vehicle instance.
        sensor_cords : list
            The sensor's coordinates [x, y, z, roll, yaw, pitch].

        Returns
        -------
        sensor_coords : np.ndarray
            Transformed coordinates in sensor space.
        """
        world_coords = SensorTransformer.convert_bbox_to_world(vertices, vehicle)

        return SensorTransformer.world_to_sensor_coords(world_coords, sensor_cords)

    # -------------------------------
    # Coordinate Transformation
    # -------------------------------

    @staticmethod
    def cords_to_transform_matrix(cords):
        """
        Convert a coordinates list [x, y, z, roll, yaw, pitch] to a 4x4 transformation matrix.

        Parameters
        ----------
        cords : list
            A list containing [x, y, z, roll, yaw, pitch].

        Returns
        -------
        T : np.ndarray
            A 4x4 transformation matrix.
        """
        x, y, z, roll, yaw, pitch = cords

        cy = np.cos(np.deg2rad(yaw))
        sy = np.sin(np.deg2rad(yaw))
        cr = np.cos(np.deg2rad(roll))
        sr = np.sin(np.deg2rad(roll))
        cp = np.cos(np.deg2rad(pitch))
        sp = np.sin(np.deg2rad(pitch))

        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z

        T[0, 0] = cp * cy
        T[0, 1] = cy * sp * sr - sy * cr
        T[0, 2] = -cy * sp * cr - sy * sr
        T[1, 0] = sy * cp
        T[1, 1] = sy * sp * sr + cy * cr
        T[1, 2] = -sy * sp * cr + cy * sr
        T[2, 0] = sp
        T[2, 1] = -cp * sr
        T[2, 2] = cp * cr

        return T
    
    @staticmethod
    def transform_to_world_matrix(transform):
        """
        Compute a 4x4 homogeneous transformation matrix from a transform.

        Parameters
        ----------
        transform : carla.Transform
            The transform object.

        Returns
        -------
        T : np.ndarray
            A 4x4 transformation matrix.
        """
        rot = transform.rotation
        loc = transform.location

        cy = np.cos(np.deg2rad(rot.yaw))
        sy = np.sin(np.deg2rad(rot.yaw))
        cr = np.cos(np.deg2rad(rot.roll))
        sr = np.sin(np.deg2rad(rot.roll))
        cp = np.cos(np.deg2rad(rot.pitch))
        sp = np.sin(np.deg2rad(rot.pitch))

        T = np.eye(4)
        T[0, 3] = loc.x
        T[1, 3] = loc.y
        T[2, 3] = loc.z

        T[0, 0] = cp * cy
        T[0, 1] = cy * sp * sr - sy * cr
        T[0, 2] = -cy * sp * cr - sy * sr
        T[1, 0] = sy * cp
        T[1, 1] = sy * sp * sr + cy * cr
        T[1, 2] = -sy * sp * cr + cy * sr
        T[2, 0] = sp
        T[2, 1] = -cp * sr
        T[2, 2] = cp * cr

        return T
    
    @staticmethod
    def convert_bbox_to_world(vertices, vehicle):
        """
        Transform bounding box vertices from vehicle frame to world coordinates.

        Parameters
        ----------
        vertices : np.ndarray
            An (8,4) array of vertices.
        vehicle : object
            The vehicle instance.

        Returns
        -------
        world_coords : np.ndarray
            The transformed vertices in world space.
        """
        T_bb_to_agent = SensorTransformer.transform_to_world_matrix(vehicle.bounding_box)

        T_agent_to_world = SensorTransformer.transform_to_world_matrix(vehicle.get_transform())

        T_bb_to_world = T_agent_to_world @ T_bb_to_agent

        return T_bb_to_world @ vertices.T
    
    @staticmethod
    def world_to_sensor_coords(world_points, sensor_cords):
        """
        Convert points from world coordinates to sensor coordinates.

        Parameters
        ----------
        world_points : np.ndarray
            Homogeneous points (4xn) in world coordinates.
        sensor_cords : list
            The sensor's coordinates [x, y, z, roll, yaw, pitch].

        Returns
        -------
        sensor_points : np.ndarray
            Points in sensor coordinate frame.
        """
        T_sensor = SensorTransformer.cords_to_transform_matrix(sensor_cords)
        T_world_to_sensor = np.linalg.inv(T_sensor)

        return T_world_to_sensor @ world_points
    
    @staticmethod
    def sensor_to_world_coords(sensor_points, sensor_cords):
        """
        Convert points from sensor coordinates to world coordinates.

        Parameters
        ----------
        sensor_points : np.ndarray
            Homogeneous points in sensor coordinates.
        sensor_cords : list
            The sensor's coordinates [x, y, z, roll, yaw, pitch].

        Returns
        -------
        world_points : np.ndarray
            Points in world coordinates.
        """
        T_sensor = SensorTransformer.cords_to_transform_matrix(sensor_cords)

        return T_sensor @ sensor_points

    # -------------------------------
    # Point and Intensity Utilities
    # -------------------------------

    @staticmethod
    def convert_points_to_homogeneous(points):
        """
        Convert a set of 3D points to homogeneous coordinates.

        Parameters
        ----------
        points : np.ndarray
            A (3, n) array.

        Returns
        -------
        homo_points : np.ndarray
            A (4, n) array of homogeneous points.
        """
        ones = np.ones((1, points.shape[1]))

        return np.vstack([points, ones])
    
    @staticmethod
    def filter_points_in_image(points, img_width, img_height):
        """
        Filter projected points to those within image boundaries.

        Parameters
        ----------
        points : np.ndarray
            An (n,3) array of points.
        img_width : int
            Image width.
        img_height : int
            Image height.

        Returns
        -------
        valid_points : np.ndarray
            Points that lie within the image.
        valid_mask : np.ndarray
            Boolean mask indicating valid points.
        """
        mask = ((points[:, 0] > 0) & (points[:, 0] < img_width) &
                (points[:, 1] > 0) & (points[:, 1] < img_height) &
                (points[:, 2] > 0))
        
        return points[mask], mask
    
    @staticmethod
    def map_intensity_to_rgb(intensities):
        """
        Map intensity values to RGB using the Viridis colormap.

        Parameters
        ----------
        intensities : np.ndarray
            1D intensity values.

        Returns
        -------
        rgb_colors : np.ndarray
            An (n, 3) array of RGB values.
        """
        r = np.interp(intensities, SensorTransformer.INTENSITY_SCALE, SensorTransformer.VIRIDIS_COLORMAP[:, 0]) * 255.0
        g = np.interp(intensities, SensorTransformer.INTENSITY_SCALE, SensorTransformer.VIRIDIS_COLORMAP[:, 1]) * 255.0
        b = np.interp(intensities, SensorTransformer.INTENSITY_SCALE, SensorTransformer.VIRIDIS_COLORMAP[:, 2]) * 255.0

        return np.stack([r, g, b], axis=-1).astype(np.int)
    
    @staticmethod
    def convert_ue4_to_opencv(coord):
        """
        Convert a point from UE4 coordinate system to OpenCV's.

        Parameters
        ----------
        coord : np.ndarray
            A 3-element array.

        Returns
        -------
        new_coord : np.ndarray
            The converted 3-element array.
        """
        return np.array([coord[1], -coord[2], coord[0]])

    # -------------------------------
    # Lidar Projection
    # -------------------------------

    @staticmethod
    def project_lidar_to_camera(lidar_data, camera_data, rgb_image):
        """
        Project lidar points onto the camera image.

        Parameters
        ----------
        lidar_data : dict
            The lidar sensor data dictionary containing 'points' and 'cords'.
        camera_data : dict
            The camera sensor data dictionary.
        rgb_image : np.ndarray
            The camera image.

        Returns
        -------
        annotated_img : np.ndarray
            The image with lidar points overlaid.
        proj_points : np.ndarray
            The projected 2D coordinates.
        """
        point_cloud = lidar_data['points']
        intensities = point_cloud[:, 3]
        pts_xyz = point_cloud[:, :3].T

        pts_hom = SensorTransformer.convert_points_to_homogeneous(pts_xyz)
        T_lidar_world = SensorTransformer.cords_to_transform_matrix(lidar_data['cords'])
        world_pts = T_lidar_world @ pts_hom 
        
        sensor_pts = SensorTransformer.world_to_sensor_coords(world_pts, camera_data['cords'])
        cam_coords = np.apply_along_axis(SensorTransformer.convert_ue4_to_opencv, 0, sensor_pts[:3, :])
        K = SensorTransformer.compute_camera_intrinsics(camera_data)
        proj = K @ cam_coords
        proj /= proj[2, :]
        proj = proj.T

        img_width = int(camera_data['image_size_x'])
        img_height = int(camera_data['image_size_y'])
        valid_pts, valid_mask = SensorTransformer.filter_points_in_image(proj, img_width, img_height)
        valid_intensities = intensities[valid_mask]
        colors = SensorTransformer.map_intensity_to_rgb(4 * valid_intensities - 3)
        u_coords = valid_pts[:, 0].astype(np.int)
        v_coords = valid_pts[:, 1].astype(np.int)
        
        annotated_img = rgb_image.copy()

        for idx in range(valid_pts.shape[0]):
            annotated_img[v_coords[idx]-1:v_coords[idx]+1,
                        u_coords[idx]-1:u_coords[idx]+1] = colors[idx]
            
        return annotated_img, proj
