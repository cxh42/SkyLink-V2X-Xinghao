# -*- coding: utf-8 -*-
"""
Drone planning manager module
"""

import math
import numpy as np
import carla
import airsim

from skylink_v2x.core.autonomy_stack.planning.core.collision_detection import CollisionDetector
from skylink_v2x.core.autonomy_stack.planning.core.trajectory_generation import LocalTrajectoryPlanner
from skylink_v2x.core.autonomy_stack.planning.core.route_planning import GlobalRoutePlanner
from skylink_v2x.core.autonomy_stack.planning.core.route_planning import GlobalRouteDatabase
from skylink_v2x.core.autonomy_stack.planning.visualization.debug_helper import PlanningDebugHelper
from skylink_v2x.core.autonomy_stack.planning.managers.base_manager import BasePlanningManager
from skylink_v2x.utils import dis_3d
from skylink_v2x.core.autonomy_stack.planning.core.common_utils import (
    get_vehicle_speed, compute_distance, distance_to_vehicle
)

class DroneMode:
    """Enumeration of drone operation modes"""
    HOVER = "hover"
    ESCORT = "escort"
    PATROL = "patrol"

class DronePlanner:
    
    def __init__(self, drone, config_yaml, comm_manager, attached_vehicle):
        self._drone = drone
        self._agent_id = drone.id
        self._mode = config_yaml['mode']
        self._config = config_yaml
        if self._mode not in DroneMode.__dict__.values():
            raise ValueError(f"Invalid mode: {self._mode}. Must be one of {DroneMode.__dict__.values()}")
        self._comm_manager = comm_manager
        self._attached_vehicle = attached_vehicle
        self._patrol_points = config_yaml.get('patrol_points', [])
        self._patrol_idx = 0
        
    def update_information(self, ego_pos):
        self._ego_pos = ego_pos
        
    def escort(self):
        """
        Execute an escort maneuver
        
        Returns
        -------
        target_location
        """
        vehicle_transform = self._attached_vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = vehicle_transform.rotation.yaw
        height = self._config['height']
        target_location = carla.Location(
            vehicle_location.x,
            vehicle_location.y,
            height
        )
        airsim_drone_yaw = vehicle_yaw - 90
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=airsim_drone_yaw)
        
        return target_location, yaw_mode
    
    def hover(self):
        """
        Execute a hover maneuver
        
        Returns
        -------
        target_location
        """
        height = self._config['height']
        target_location = carla.Location(
            self._ego_pos.location.x,
            self._ego_pos.location.y,
            height
        )
        return target_location, None
    
    def patrol(self):
        """
        Execute a patrol maneuver
        
        Returns
        -------
        target_location
        """
        assert self._patrol_points, "Patrol points are not defined"
        target_location = carla.Location(
            self._patrol_points[self._patrol_idx][0],
            self._patrol_points[self._patrol_idx][1],
            self._patrol_points[self._patrol_idx][2]
        )
        if dis_3d(
            self._ego_pos.location, target_location) < self._config['patrol_radius']:
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_points)
        target_location = carla.Location(
            self._patrol_points[self._patrol_idx][0],
            self._patrol_points[self._patrol_idx][1],
            self._patrol_points[self._patrol_idx][2]
        )
        return target_location, None

    
    def run_step(self):
        """
        Execute one planning step based on the current mode
        
        Parameters
        ----------
        target_speed : float, optional
            Override target speed if specified
            
        Returns
        -------
        tuple
            target_location
        """
        # Handle different modes
        if self._mode == DroneMode.HOVER:
            planning_results = self.hover()
            
        elif self._mode == DroneMode.ESCORT:
            assert self._attached_vehicle is not None, "Attached vehicle is required for escort mode"
            planning_results = self.escort()
            
        elif self._mode == DroneMode.PATROL:
            planning_results = self.patrol()
        
        else:
            raise ValueError(f"Unknown mode: {self._mode}")
        
        self._comm_manager.buffer_planning(self._agent_id, planning_results)
        
        