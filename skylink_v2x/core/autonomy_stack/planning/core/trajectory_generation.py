# -*- coding: utf-8 -*-
"""
Local trajectory planning and generation
"""

import math
import statistics
from collections import deque

import carla
import numpy as np

from skylink_v2x.core.autonomy_stack.planning.core.curve_utils import Spline2D
from skylink_v2x.core.autonomy_stack.planning.core.common_utils import (
    calculate_angle_distance, compute_distance, distance_to_vehicle, draw_trajectory_points
)

class LocalTrajectoryPlanner:
    """
    Local trajectory planner for generating drivable paths.
    
    This class handles the generation of smooth, drivable trajectories
    based on waypoints from the global route.
    
    Parameters
    ----------
    agent : BasePlanningManager
        The planning manager
    carla_map : carla.Map
        Map of the simulation world
    config_yaml : dict
        Configuration dictionary
    """

    def __init__(self, agent, carla_map, config_yaml):
        self._agent = agent
        self._vehicle = agent.vehicle if hasattr(agent, 'vehicle') else None
        self._map = carla_map
        
        # Vehicle state
        self._ego_pos = None
        self._ego_speed = None
        self._target_speed = 0.0
        
        # Waypoint thresholds
        self._min_distance = config_yaml['min_dist']
        self._buffer_size = config_yaml['buffer_size']
        
        # Queues and buffers
        self.waypoints_queue = deque(maxlen=20000)
        self._waypoint_buffer = deque(maxlen=self._buffer_size)
        self._trajectory_buffer = deque(maxlen=30)
        self._history_buffer = deque(maxlen=3)
        
        # Target waypoint
        self.target_waypoint = None
        
        # Update frequencies
        self.trajectory_update_freq = config_yaml['trajectory_update_freq']
        self.waypoint_update_freq = config_yaml['waypoint_update_freq']
        
        # Trajectory parameters
        self.dt = config_yaml['trajectory_dt']
        
        # State flags
        self.potential_curved_road = False
        self.lane_id_change = False
        self.lane_lateral_change = False
        
        # Debug options
        self.debug = config_yaml['debug']
        self.debug_trajectory = config_yaml['debug_trajectory']
        self._long_plan_debug = []

    def update_information(self, ego_pos, ego_speed):
        """
        Update planner with latest vehicle state
        
        Parameters
        ----------
        ego_pos : carla.Transform
            Current vehicle position and orientation
        ego_speed : float
            Current vehicle speed in km/h
        """
        self._ego_pos = ego_pos
        self._ego_speed = ego_speed

    def set_global_plan(self, current_plan, clean=False):
        """
        Set the global route plan
        
        Parameters
        ----------
        current_plan : list
            List of (waypoint, road_option) tuples in the route
        clean : bool
            Whether to clear existing waypoints
        """
        for waypoint_option in current_plan:
            self.waypoints_queue.append(waypoint_option)
            
        if clean:
            self._waypoint_buffer.clear()
            for _ in range(self._buffer_size):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(self.waypoints_queue.popleft())
                else:
                    break

    def get_waypoints_queue(self):
        """Get the global waypoints queue"""
        return self.waypoints_queue

    def get_waypoint_buffer(self):
        """Get the local waypoint buffer"""
        return self._waypoint_buffer

    def get_trajectory(self):
        """Get the current trajectory buffer"""
        return self._trajectory_buffer

    def get_history_buffer(self):
        """Get the history of visited waypoints"""
        return self._history_buffer

    def generate_path(self):
        """
        Generate a smooth path using cubic spline interpolation
        
        Returns
        -------
        tuple
            (path_x, path_y, path_curvature, path_yaw) of the generated path
        """
        # Lists to store spline nodes
        x, y = [], []
        
        # Clean waypoint buffer
        self._filter_waypoint_buffer()
        
        # Sampling distance
        ds = 0.1
        
        # Get current location and orientation
        current_location = self._ego_pos.location
        current_yaw = self._ego_pos.rotation.yaw
        
        # Get current waypoint
        current_waypoint = self._map.get_waypoint(current_location).next(1)[0]
        current_waypoint_loc = current_waypoint.transform.location
        
        # Check for lane changes by comparing future and past waypoints
        future_waypoint = self._waypoint_buffer[-1][0] if self._waypoint_buffer else current_waypoint
        previous_waypoint = self._history_buffer[0][0] if self._history_buffer else current_waypoint
        
        # Calculate lateral offset between previous and future waypoints
        vec_norm, angle = calculate_angle_distance(
            previous_waypoint.transform.location,
            future_waypoint.transform.location,
            future_waypoint.transform.rotation.yaw
        )
        
        # Lateral difference
        lateral_diff = abs(
            vec_norm * math.sin(math.radians(angle - 1 if angle > 90 else angle + 1))
        )
        
        # Get vehicle and lane widths
        if hasattr(self._vehicle, 'bounding_box'):
            bounding_box = self._vehicle.bounding_box
            vehicle_width = 2 * abs(bounding_box.location.y - bounding_box.extent.y)
        else:
            vehicle_width = 2.0  # Default width
            
        lane_width = current_waypoint.lane_width
        
        # Check if lane change detected
        self.lane_lateral_change = vehicle_width < lateral_diff
        self.lane_id_change = (
            future_waypoint.lane_id != current_waypoint.lane_id or
            previous_waypoint.lane_id != future_waypoint.lane_id
        )
        self.potential_curved_road = self.lane_id_change or self.lane_lateral_change
        
        # Check angle to first waypoint in buffer
        _, angle_to_first = calculate_angle_distance(
            self._waypoint_buffer[0][0].transform.location if self._waypoint_buffer else current_waypoint_loc,
            current_location,
            current_yaw
        )
        
        # Include history waypoints in spline calculation
        history_count = 0
        for i in range(len(self._history_buffer)):
            prev_waypoint_loc = self._history_buffer[i][0].transform.location
            _, angle = calculate_angle_distance(prev_waypoint_loc, current_location, current_yaw)
            
            # Only include history point if it's already passed
            if angle > 90 and not self.potential_curved_road:
                x.append(prev_waypoint_loc.x)
                y.append(prev_waypoint_loc.y)
                history_count += 1
            # Include all history points during lane change
            elif self.potential_curved_road:
                x.append(prev_waypoint_loc.x)
                y.append(prev_waypoint_loc.y)
                history_count += 1
        
        # Add current position to path
        if self.potential_curved_road:
            _, angle = calculate_angle_distance(
                self._waypoint_buffer[0][0].transform.location if self._waypoint_buffer else current_waypoint_loc,
                current_location,
                current_yaw
            )
            
            # If starting lane change with no history
            if len(x) == 0 or len(y) == 0:
                x.append(current_location.x)
                y.append(current_location.y)
        else:
            _, angle = calculate_angle_distance(current_waypoint_loc, current_location, current_yaw)
            
            # Prefer waypoint if it's in front of us (centered in lane)
            if angle < 90:
                x.append(current_waypoint_loc.x)
                y.append(current_waypoint_loc.y)
            else:
                x.append(current_location.x)
                y.append(current_location.y)
        
        # Filter waypoints that are too close to each other
        history_index = max(0, history_count - 1) if self.potential_curved_road else history_count
        prev_x = x[history_index] if history_index < len(x) else x[-1]
        prev_y = y[history_index] if history_index < len(y) else y[-1]
        
        # Add future waypoints from buffer
        for i in range(len(self._waypoint_buffer)):
            waypoint_loc = self._waypoint_buffer[i][0].transform.location
            cur_x, cur_y = waypoint_loc.x, waypoint_loc.y
            
            # Skip if too close to previous point
            if abs(prev_x - cur_x) < 0.5 and abs(prev_y - cur_y) < 0.5:
                continue
                
            prev_x, prev_y = cur_x, cur_y
            x.append(cur_x)
            y.append(cur_y)
        
        # Cannot generate spline with fewer than 2 points
        if len(x) < 2 or len(y) < 2:
            return [], [], [], []
        
        # Create spline and sample points
        sp = Spline2D(x, y)
        
        # Calculate distance from current position to spline start
        diff_x = current_location.x - sp.sx.a[0]
        diff_y = current_location.y - sp.sy.a[0]
        diff_s = np.hypot(diff_x, diff_y)
        
        # Sample points along spline
        s = np.arange(diff_s, sp.s[-1], ds)
        
        # Generate path points
        path_x, path_y, path_yaw, path_k = [], [], [], []
        self._long_plan_debug = []
        
        for i, i_s in enumerate(s):
            ix, iy = sp.calc_position(i_s)
            
            # Skip if too close to a history point
            if abs(ix - x[history_index]) <= ds and abs(iy - y[history_index]) <= ds:
                continue
                
            # Save first half of points for debugging
            if i <= len(s) // 2:
                self._long_plan_debug.append(carla.Transform(carla.Location(ix, iy, 0)))
                
            path_x.append(ix)
            path_y.append(iy)
            path_k.append(max(min(sp.calc_curvature(i_s), 0.2), -0.2))  # Limit curvature
            path_yaw.append(sp.calc_yaw(i_s))
        
        return path_x, path_y, path_k, path_yaw

    def generate_trajectory(self, path_x, path_y, path_k):
        """
        Generate speed profile for the path
        
        Parameters
        ----------
        path_x : list
            X coordinates of path
        path_y : list
            Y coordinates of path
        path_k : list
            Curvatures along path
        """
        # Sampling parameters
        ds = 0.1  # spatial resolution
        dt = self.dt  # temporal resolution
        
        # Get target and current speeds
        target_speed = self._target_speed
        current_speed = self._ego_speed / 3.6  # Convert to m/s
        
        # Number of samples to generate (2.0 seconds ahead)
        sample_count = int(2.0 / dt)
        
        # Initialize variables
        break_flag = False
        sample_distance = 0
        
        # Calculate mean curvature to adjust speed
        mean_k = 0.0001 if len(path_k) < 2 else abs(statistics.mean(path_k))
        
        # Limit speed based on curvature (v^2 = a_lat/k)
        # Assume max lateral acceleration of 5.0 m/s^2
        target_speed = min(target_speed, np.sqrt(5.0 / (mean_k + 1e-6)) * 3.6)
        
        # Calculate acceleration to reach target speed
        max_accel = 3.5  # m/s^2
        acceleration = max(
            min(max_accel, (target_speed / 3.6 - current_speed) / dt),
            -6.5  # Max deceleration
        )
        
        # Clear trajectory buffer
        self._trajectory_buffer.clear()
        
        # Generate trajectory points
        for i in range(1, sample_count + 1):
            # Update distance based on kinematic equations
            sample_distance += current_speed * dt + 0.5 * acceleration * dt**2
            current_speed += acceleration * dt
            
            # Get point at this distance
            if int(sample_distance / ds) >= len(path_x):
                sample_x = path_x[-1]
                sample_y = path_y[-1]
                break_flag = True
            else:
                index = max(0, int(sample_distance / ds))
                sample_x = path_x[index]
                sample_y = path_y[index]
            
            # Add to trajectory buffer
            self._trajectory_buffer.append(
                (carla.Transform(
                    carla.Location(
                        sample_x,
                        sample_y,
                        self._waypoint_buffer[0][0].transform.location.z + 0.5
                        if self._waypoint_buffer else self._ego_pos.location.z
                    )
                ), target_speed)
            )
            
            if break_flag:
                break

    def _filter_waypoint_buffer(self):
        """
        Filter waypoints that would cause unstable vehicle dynamics
        """
        previous_waypoint = None
        buffer_copy = list(self._waypoint_buffer)
        
        for i, (waypoint, _) in enumerate(buffer_copy):
            # Only check first few waypoints
            if i >= 3:
                break
                
            # Find index in original buffer
            j = i - (len(buffer_copy) - len(self._waypoint_buffer))
            if j < 0:
                continue
                
            # Check if waypoint is behind vehicle
            _, angle = calculate_angle_distance(
                waypoint.transform.location,
                self._ego_pos.location,
                self._ego_pos.rotation.yaw
            )
            
            if angle > 90:
                # Remove waypoint if it's behind us
                del self._waypoint_buffer[j]
                continue
                
            if previous_waypoint is None:
                previous_waypoint = waypoint
                continue
                
            # Check if waypoint is on different lane and too close
            if (previous_waypoint.lane_id != waypoint.lane_id and
                    len(self._waypoint_buffer) >= 2):
                    
                distance = compute_distance(
                    waypoint.transform.location,
                    previous_waypoint.transform.location
                )
                
                if distance <= 4.5:
                    del self._waypoint_buffer[j]
                    
            previous_waypoint = waypoint

    def pop_waypoints(self, vehicle_transform):
        """
        Remove waypoints that have been reached
        
        Parameters
        ----------
        vehicle_transform : carla.Transform
            Current vehicle transform
        """
        # Find waypoints that have been reached
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoint_buffer):
            if distance_to_vehicle(waypoint, vehicle_transform) < self._min_distance:
                max_index = i
                
        # Move reached waypoints to history buffer
        if max_index >= 0:
            for i in range(max_index + 1):
                if self._waypoint_buffer:
                    if self._history_buffer:
                        prev_waypoint = self._history_buffer[-1]
                        incoming_waypoint = self._waypoint_buffer.popleft()
                        
                        # Only add to history if significantly different position
                        if (abs(prev_waypoint[0].transform.location.x - incoming_waypoint[0].transform.location.x) > 4.5 or
                                abs(prev_waypoint[0].transform.location.y - incoming_waypoint[0].transform.location.y) > 4.5):
                            self._history_buffer.append(incoming_waypoint)
                    else:
                        # First waypoint automatically goes to history
                        self._history_buffer.append(self._waypoint_buffer.popleft())
        
        # Remove reached trajectory points
        if self._trajectory_buffer:
            max_index = -1
            for i, (waypoint, _) in enumerate(self._trajectory_buffer):
                if distance_to_vehicle(waypoint, vehicle_transform) < max(self._min_distance - 1, 1):
                    max_index = i
                    
            if max_index >= 0:
                for i in range(max_index + 1):
                    if self._trajectory_buffer:
                        self._trajectory_buffer.popleft()

    def run_step(self, path_x, path_y, path_k, target_speed=None, trajectory=None, following=False):
        """
        Execute one step of trajectory planning
        
        Parameters
        ----------
        path_x : list
            X coordinates of path
        path_y : list
            Y coordinates of path
        path_k : list
            Curvatures along path
        target_speed : float, optional
            Target speed override
        trajectory : deque, optional
            Pre-generated trajectory for following
        following : bool
            Whether vehicle is following another vehicle
            
        Returns
        -------
        tuple
            (target_speed, target_location) for next step
        """
        # Set target speed
        self._target_speed = target_speed
        
        # Refill waypoint buffer if needed
        if len(self._waypoint_buffer) < self.waypoint_update_freq:
            for i in range(self._buffer_size - len(self._waypoint_buffer)):
                if self.waypoints_queue:
                    self._waypoint_buffer.append(self.waypoints_queue.popleft())
                else:
                    break
        
        # Generate or update trajectory
        if (not trajectory and 
                len(self._trajectory_buffer) < self.trajectory_update_freq and
                not following):
                
            self._trajectory_buffer.clear()
            
            # Need valid path to generate trajectory
            if not path_x:
                return 0, None
                
            self.generate_trajectory(path_x, path_y, path_k)
                
        elif trajectory:
            # Use provided trajectory
            self._trajectory_buffer = trajectory.copy()
        
        # Get target waypoint from trajectory
        if self._trajectory_buffer:
            self.target_waypoint, self._target_speed = self._trajectory_buffer[
                min(1, len(self._trajectory_buffer) - 1)
            ]
        else:
            # No trajectory available
            return 0, None
        
        # Update waypoint buffers
        self.pop_waypoints(self._ego_pos)
        
        # Draw debug visualizations if enabled
        if self.debug_trajectory and hasattr(self._vehicle, 'get_world'):
            draw_trajectory_points(
                self._vehicle.get_world(),
                self._long_plan_debug,
                color=carla.Color(0, 255, 0),
                size=0.05,
                lifetime=0.1
            )
            
        if self.debug and hasattr(self._vehicle, 'get_world'):
            draw_trajectory_points(
                self._vehicle.get_world(),
                self._waypoint_buffer,
                z=0.1,
                size=0.1,
                color=carla.Color(0, 0, 255),
                lifetime=0.2
            )
            draw_trajectory_points(
                self._vehicle.get_world(),
                self._history_buffer,
                z=0.1,
                size=0.1,
                color=carla.Color(255, 0, 255),
                lifetime=0.2
            )
        
        # Return target speed and location
        target_location = (
            self.target_waypoint.transform.location 
            if hasattr(self.target_waypoint, 'transform') 
            else self.target_waypoint.location
        )
        
        return self._target_speed, target_location
