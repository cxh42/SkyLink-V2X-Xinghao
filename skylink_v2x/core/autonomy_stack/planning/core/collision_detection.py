# -*- coding: utf-8 -*-
"""
Collision detection module for planning
"""

import math
from math import sin, cos
from scipy import spatial

import carla
import numpy as np

from skylink_v2x.core.autonomy_stack.planning.core.curve_utils import Spline2D

class CollisionDetector:
    """
    Collision detection module for identifying potential hazards.
    
    Parameters
    ----------
    time_ahead : float
        How many seconds to look ahead for collision detection
    circle_radius : float
        Radius of collision detection circles
    circle_offsets : list
        Lateral offsets for collision detection circles
    """

    def __init__(self, time_ahead=1.2, circle_radius=1.0, circle_offsets=None):
        self.time_ahead = time_ahead
        self._circle_offsets = [-1.0, 0, 1.0] if circle_offsets is None else circle_offsets
        self._circle_radius = circle_radius

    def is_in_range(self, ego_pos, target_vehicle, candidate_vehicle, carla_map):
        """
        Check if a candidate vehicle is between ego and target vehicle
        
        Parameters
        ----------
        carla_map : carla.Map
            Carla map of the current simulation world
        ego_pos : carla.Transform
            Ego vehicle position
        target_vehicle : carla.Vehicle
            Target vehicle that ego is trying to reach
        candidate_vehicle : carla.Vehicle
            Potential obstacle vehicle
            
        Returns
        -------
        bool
            True if candidate vehicle is between ego and target
        """
        ego_loc = ego_pos.location
        target_loc = target_vehicle.get_location()
        candidate_loc = candidate_vehicle.get_location()

        # Define bounding box
        min_x, max_x = min(ego_loc.x, target_loc.x), max(ego_loc.x, target_loc.x)
        min_y, max_y = min(ego_loc.y, target_loc.y), max(ego_loc.y, target_loc.y)

        # Check if candidate is outside the bounding box (with buffer)
        if (candidate_loc.x <= min_x - 2 or candidate_loc.x >= max_x + 2 or
                candidate_loc.y <= min_y - 2 or candidate_loc.y >= max_y + 2):
            return False

        # Get waypoints
        candidate_wpt = carla_map.get_waypoint(candidate_loc)
        target_wpt = carla_map.get_waypoint(target_loc)

        # If in same lane, then it's blocking
        if target_wpt.lane_id == candidate_wpt.lane_id:
            return True

        # Different sections but potentially blocking
        if target_wpt.section_id == candidate_wpt.section_id:
            return False

        # Check angle between vehicles to determine if blocking
        distance, angle = self._calculate_distance_angle(
            target_wpt.transform.location, 
            candidate_wpt.transform.location,
            candidate_wpt.transform.rotation.yaw
        )

        return angle <= 3

    def adjacent_lane_collision_check(self, ego_loc, target_wpt, overtake, carla_map, world):
        """
        Generate check line in adjacent lane for collision detection
        
        Parameters
        ----------
        ego_loc : carla.Location
            Ego vehicle location
        target_wpt : carla.Waypoint
            Target waypoint in adjacent lane
        overtake : bool
            Whether this is for overtaking
        carla_map : carla.Map
            Carla map of the simulation world
        world : carla.World
            Carla simulation world
            
        Returns
        -------
        tuple
            (rx, ry, ryaw) coordinates and orientations of check line
        """
        # Determine forward point based on overtake flag
        if overtake:
            target_wpt_next = target_wpt.next(6)[0]
        else:
            target_wpt_next = target_wpt

        # Calculate distance to ego
        diff_x = target_wpt_next.transform.location.x - ego_loc.x
        diff_y = target_wpt_next.transform.location.y - ego_loc.y
        diff_s = np.hypot(diff_x, diff_y) + 3

        # Get previous waypoint
        target_wpt_previous = target_wpt.previous(diff_s)
        while len(target_wpt_previous) == 0:
            diff_s -= 2
            target_wpt_previous = target_wpt.previous(diff_s)

        target_wpt_previous = target_wpt_previous[0]
        target_wpt_middle = target_wpt_previous.next(diff_s/2)[0]

        # Create spline path
        x = [target_wpt_next.transform.location.x,
             target_wpt_middle.transform.location.x,
             target_wpt_previous.transform.location.x]
        y = [target_wpt_next.transform.location.y,
             target_wpt_middle.transform.location.y,
             target_wpt_previous.transform.location.y]
             
        ds = 0.1
        sp = Spline2D(x, y)
        s = np.arange(sp.s[0], sp.s[-1], ds)

        # Calculate interpolation points
        rx, ry, ryaw = [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))

        return rx, ry, ryaw

    def collision_circle_check(self, path_x, path_y, path_yaw, obstacle_vehicle, 
                              speed, carla_map, adjacent_check=False):
        """
        Check for collisions using circle method
        
        Parameters
        ----------
        path_x : list
            X coordinates of path points
        path_y : list
            Y coordinates of path points
        path_yaw : list
            Yaw angles of path points
        obstacle_vehicle : carla.Vehicle
            Vehicle to check for collision with
        speed : float
            Vehicle speed in m/s
        carla_map : carla.Map
            Carla map
        adjacent_check : bool
            Whether checking adjacent lane
            
        Returns
        -------
        bool
            True if path is collision-free
        """
        collision_free = True
        
        # Determine check distance based on speed
        if not adjacent_check:
            distance_check = min(
                max(int(self.time_ahead * speed / 0.1), 90),
                len(path_x)
            )
        else:
            distance_check = len(path_x)

        # Get obstacle information
        obstacle_loc = obstacle_vehicle.get_location()
        obstacle_yaw = carla_map.get_waypoint(obstacle_loc).transform.rotation.yaw

        # Check points along path
        for i in range(0, distance_check, 10):
            ptx, pty, yaw = path_x[i], path_y[i], path_yaw[i]

            # Calculate circle collision detection points
            circle_locations = np.zeros((len(self._circle_offsets), 2))
            circle_offsets = np.array(self._circle_offsets)
            circle_locations[:, 0] = ptx + circle_offsets * cos(yaw)
            circle_locations[:, 1] = pty + circle_offsets * sin(yaw)

            # Calculate obstacle vehicle bounding box
            corrected_extent_x = obstacle_vehicle.bounding_box.extent.x * math.cos(math.radians(obstacle_yaw))
            corrected_extent_y = obstacle_vehicle.bounding_box.extent.y * math.sin(math.radians(obstacle_yaw))

            # Compute corners of obstacle bounding box
            obstacle_bbx_array = np.array([
                [obstacle_loc.x - corrected_extent_x, obstacle_loc.y - corrected_extent_y],
                [obstacle_loc.x - corrected_extent_x, obstacle_loc.y + corrected_extent_y],
                [obstacle_loc.x, obstacle_loc.y],
                [obstacle_loc.x + corrected_extent_x, obstacle_loc.y - corrected_extent_y],
                [obstacle_loc.x + corrected_extent_x, obstacle_loc.y + corrected_extent_y]
            ])

            # Compute distances and check for collision
            collision_dists = spatial.distance.cdist(obstacle_bbx_array, circle_locations)
            collision_dists = np.subtract(collision_dists, self._circle_radius)
            collision_free = collision_free and not np.any(collision_dists < 0)

            if not collision_free:
                break

        return collision_free
        
    def _calculate_distance_angle(self, target_location, current_location, orientation):
        """
        Calculate distance and angle between two points
        
        Parameters
        ----------
        target_location : carla.Location
            Target location
        current_location : carla.Location
            Current location
        orientation : float
            Current orientation in degrees
            
        Returns
        -------
        tuple
            (distance, angle) between points
        """
        target_vector = np.array([
            target_location.x - current_location.x,
            target_location.y - current_location.y
        ])
        
        norm_target = np.linalg.norm(target_vector)
        
        if norm_target < 0.001:
            return 0.0, 0.0
        
        forward_vector = np.array([
            math.cos(math.radians(orientation)),
            math.sin(math.radians(orientation))
        ])
        
        d_angle = math.degrees(
            math.acos(np.clip(
                np.dot(forward_vector, target_vector) / norm_target,
                -1.0, 1.0
            ))
        )
        
        return norm_target, d_angle
    