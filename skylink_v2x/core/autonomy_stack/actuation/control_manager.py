# -*- coding: utf-8 -*-
"""
Controller interface
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import importlib
import airsim
from skylink_v2x.utils import convert_carla_to_airsim
from skylink_v2x.core.autonomy_stack.actuation.pid_controller import Controller
from skylink_v2x.utils import convert_airsim_to_carla


class ControlManager(object):
    """
    Controller manager that is used to select
    and call different controller's functions.

    Parameters
    ----------
    control_config : dict
        The configuration dictionary of the control manager module.

    Attributes
    ----------
    controller : opencda object.
        The controller object of the OpenCDA framwork.
    """

    def __init__(self, control_config):
        self.controller = Controller(control_config['args'])

    def update_info(self, ego_pos, ego_speed):
        """
        Update ego vehicle information for controller.
        """
        self.controller.update_info(ego_pos, ego_speed)

    def run_step(self, target_speed, waypoint):
        """
        Execute current controller step.
        """
        control_command = self.controller.run_step(target_speed, waypoint)
        return control_command


class DroneControlManager(object):
    """
    Controller manager for drone.
    """

    def __init__(self, control_config, airsim_client, drone_name):
        self.speed = control_config['speed']
        self.airsim_client = airsim_client
        self.drone_name = drone_name
        
    def run_step(self, target_location, yaw_mode=None):
        """
        Execute current controller step.
        """
        target_airsim_pos = convert_carla_to_airsim(target_location)
        if yaw_mode == None:
            # print("landed:", self.airsim_client.getMultirotorState(vehicle_name=self.drone_name).landed_state)
            self.airsim_client.moveToPositionAsync(
                target_airsim_pos.x_val, 
                target_airsim_pos.y_val, 
                target_airsim_pos.z_val,
                self.speed, 
                vehicle_name=self.drone_name
                )
            # print(target_airsim_pos.x_val, target_airsim_pos.y_val, target_airsim_pos.z_val)
            drone_pos = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated
            # new_loc = convert_airsim_to_carla(drone_pos.position)
            # print(new_loc)
            
        else:
            # print("landed:", self.airsim_client.getMultirotorState(vehicle_name=self.drone_name).landed_state)
            self.airsim_client.moveToPositionAsync(
                target_airsim_pos.x_val, 
                target_airsim_pos.y_val, 
                target_airsim_pos.z_val,
                self.speed, 
                yaw_mode=yaw_mode, 
                vehicle_name=self.drone_name
                )
            # print(target_airsim_pos.x_val, target_airsim_pos.y_val, target_airsim_pos.z_val)
            drone_pos = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated
            # new_loc = convert_airsim_to_carla(drone_pos.position)
            # print(new_loc)
            
            
