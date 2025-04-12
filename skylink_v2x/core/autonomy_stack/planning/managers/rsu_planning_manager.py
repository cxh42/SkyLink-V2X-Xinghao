# -*- coding: utf-8 -*-
"""
RSU (Road Side Unit) planning manager module
"""

import carla
from skylink_v2x.core.autonomy_stack.planning.managers.base_manager import BasePlanningManager

class RSUPlanningManager(BasePlanningManager):
    """
    Planning manager for Road Side Units (RSUs).
    
    This class provides an interface for managing road-side infrastructure such as
    traffic monitoring sensors, cameras, and other stationary units.
    
    Parameters
    ----------
    rsu_id : int
        Unique identifier for the RSU
    carla_map : carla.Map
        Map of the simulation world
    config : dict
        Configuration dictionary
    """
    
    def __init__(self, rsu_id, carla_map, config):
        super().__init__(config)
        self.rsu_id = rsu_id
        self._map = carla_map
        self._position = None
        self.objects = {}
        
    def update_information(self, position, speed=0, objects=None):
        """
        Update RSU with latest information
        
        Parameters
        ----------
        position : carla.Transform
            Current position of the RSU
        speed : float
            Not applicable for RSUs, included for interface consistency
        objects : dict
            Objects detected by perception
        """
        self._position = position
        self.objects = objects if objects else {}
    
    def set_destination(self, start_location, end_location, clean=False):
        """
        Set the destination for the RSU
        
        This is primarily an interface method for consistency. RSUs are typically
        stationary, so this has no effect by default.
        
        Parameters
        ----------
        start_location : carla.Location
            Starting location
        end_location : carla.Location
            Destination location
        clean : bool
            Whether to clean existing waypoints
        """
        # RSUs are typically stationary, so this is a placeholder
        pass
    
    def run_step(self, target_speed=None):
        """
        Execute one planning step for the RSU
        
        This method would implement any sensor movements or configuration changes
        for the RSU based on detected objects or conditions.
        
        Parameters
        ----------
        target_speed : float, optional
            Not applicable for RSUs, included for interface consistency
            
        Returns
        -------
        tuple
            Planning result (0, None) as RSUs don't move
        """
        # Interface method - RSUs typically don't move
        return 0, None