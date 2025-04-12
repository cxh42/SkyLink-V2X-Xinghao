# -*- coding: utf-8 -*-
"""
Base class for planning managers
"""

import abc

class BasePlanningManager(abc.ABC):
    """
    Base class for all planning managers.
    
    This class defines the common interface and functionality that all planning managers
    should implement, providing a consistent API across different agent types.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with planning parameters
    """
    
    def __init__(self, config):
        """Initialize the base planning manager with configuration"""
        self.config = config
    
    @abc.abstractmethod
    def update_information(self, position, speed, objects):
        """
        Update planning manager with latest perception and localization data
        
        Parameters
        ----------
        position : carla.Transform
            Current position and orientation
        speed : float
            Current speed in km/h
        objects : dict
            Dictionary of detected objects by type
        """
        pass
    
    @abc.abstractmethod
    def set_destination(self, start_location, end_location, clean=False):
        """
        Set the destination for planning
        
        Parameters
        ----------
        start_location : carla.Location
            Starting location
        end_location : carla.Location
            Destination location
        clean : bool
            Whether to clean existing waypoints
        """
        pass
    
    @abc.abstractmethod
    def run_step(self, target_speed=None):
        """
        Execute one planning step
        
        Parameters
        ----------
        target_speed : float, optional
            Target speed if specified
            
        Returns
        -------
        tuple
            Planning result (speed, location)
        """
        pass