
# -*- coding: utf-8 -*-
"""
Debug visualization and statistics for planning
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from skylink_v2x.core.autonomy_stack.planning.visualization.profile_visualizer import (
    draw_velocity_profile,
    draw_acceleration_profile,
    draw_ttc_profile
)

class PlanningDebugHelper:
    """
    Helper class for debugging and evaluating planning behavior.
    
    This class collects statistics on vehicle performance and
    provides visualization tools.
    
    Parameters
    ----------
    actor_id : int
        ID of the actor being debugged
    """

    def __init__(self, actor_id):
        self.actor_id = actor_id
        
        # Data collection lists
        self.speed_list = [[]]  # List of speed values in m/s
        self.acceleration_list = [[]]  # List of acceleration values in m/s²
        self.ttc_list = [[]]  # List of time-to-collision values in seconds
        
        # Counter for filtering startup transients
        self.counter = 0

    def update(self, speed, ttc):
        """
        Update debug statistics with latest values
        
        Parameters
        ----------
        speed : float
            Current speed in km/h
        ttc : float
            Current time to collision in seconds
        """
        self.counter += 1
        
        # Skip first few frames to avoid initialization artifacts
        if self.counter > 100:
            # Store speed in m/s
            self.speed_list[0].append(speed / 3.6)
            
            # Calculate acceleration
            if len(self.speed_list[0]) <= 1:
                self.acceleration_list[0].append(0)
            else:
                # Assuming 0.05s timestep
                self.acceleration_list[0].append(
                    (self.speed_list[0][-1] - self.speed_list[0][-2]) / 0.05
                )
                
            # Store time to collision
            self.ttc_list[0].append(ttc)

    def evaluate(self):
        """
        Generate performance evaluation plots and statistics
        
        Returns
        -------
        tuple
            (figure, performance_text) with visualization and statistics summary
        """
        warnings.filterwarnings('ignore')
        
        # Create figure for plotting
        figure = plt.figure()
        
        # Plot speed profile
        plt.subplot(311)
        draw_velocity_profile(self.speed_list)
        
        # Plot acceleration profile
        plt.subplot(312)
        draw_acceleration_profile(self.acceleration_list)
        
        # Plot TTC profile
        plt.subplot(313)
        draw_ttc_profile(self.ttc_list)
        
        # Set title
        figure.suptitle(f'Planning performance for actor ID {self.actor_id}')
        
        # Calculate statistics
        spd_avg = np.mean(np.array(self.speed_list[0]))
        spd_std = np.std(np.array(self.speed_list[0]))
        
        acc_avg = np.mean(np.array(self.acceleration_list[0]))
        acc_std = np.std(np.array(self.acceleration_list[0]))
        
        ttc_array = np.array(self.ttc_list[0])
        ttc_array = ttc_array[ttc_array < 1000]  # Filter out default TTC value
        ttc_avg = np.mean(ttc_array) if len(ttc_array) > 0 else 0
        ttc_std = np.std(ttc_array) if len(ttc_array) > 0 else 0
        
        # Create performance text summary
        performance_text = (
            f'Speed average: {spd_avg:.2f} (m/s), '
            f'Speed std: {spd_std:.2f} (m/s)\n'
            f'Acceleration average: {acc_avg:.2f} (m/s²), '
            f'Acceleration std: {acc_std:.2f} (m/s²)\n'
            f'TTC average: {ttc_avg:.2f} (s), '
            f'TTC std: {ttc_std:.2f} (s)\n'
        )
        
        return figure, performance_text