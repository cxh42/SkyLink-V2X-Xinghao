# -*- coding: utf-8 -*-
"""
Visualization utilities for planning profiles
"""

import numpy as np
import matplotlib.pyplot as plt

def draw_velocity_profile(velocity_list):
    """
    Draw velocity profiles in a single plot
    
    Parameters
    ----------
    velocity_list : list
        List of velocity profiles to plot
    """
    for i, velocity in enumerate(velocity_list):
        x_values = np.arange(len(velocity)) * 0.05  # Assuming 0.05s timestep
        plt.plot(x_values, velocity)
    
    plt.ylim([0, 34])  # 0-34 m/s range
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    
    fig = plt.gcf()
    fig.set_size_inches(11, 5)

def draw_acceleration_profile(acceleration_list):
    """
    Draw acceleration profiles in a single plot
    
    Parameters
    ----------
    acceleration_list : list
        List of acceleration profiles to plot
    """
    for i, acceleration in enumerate(acceleration_list):
        x_values = np.arange(len(acceleration)) * 0.05  # Assuming 0.05s timestep
        plt.plot(x_values, acceleration)
    
    plt.ylim([-8, 8])  # -8 to 8 m/s² range
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    
    fig = plt.gcf()
    fig.set_size_inches(11, 5)

def draw_ttc_profile(ttc_list):
    """
    Draw time-to-collision profiles in a single plot
    
    Parameters
    ----------
    ttc_list : list
        List of TTC profiles to plot
    """
    for i, ttc in enumerate(ttc_list):
        x_values = np.arange(len(ttc)) * 0.05  # Assuming 0.05s timestep
        plt.plot(x_values, ttc)
    
    plt.xlabel("Time (s)")
    plt.ylabel("TTC (s)")
    plt.ylim([0, 30])
    
    fig = plt.gcf()
    fig.set_size_inches(11, 5)

def draw_time_gap_profile(gap_list):
    """
    Draw time gap profiles in a single plot
    
    Parameters
    ----------
    gap_list : list
        List of time gap profiles to plot
    """
    for i, gap in enumerate(gap_list):
        x_values = np.arange(len(gap)) * 0.05  # Assuming 0.05s timestep
        plt.plot(x_values, gap)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Time Gap (s)")
    plt.ylim([0.0, 1.8])
    
    fig = plt.gcf()
    fig.set_size_inches(11, 5)

def draw_distance_gap_profile(gap_list):
    """
    Draw distance gap profiles in a single plot
    
    Parameters
    ----------
    gap_list : list
        List of distance gap profiles to plot
    """
    for i, gap in enumerate(gap_list):
        x_values = np.arange(len(gap)) * 0.05  # Assuming 0.05s timestep
        plt.plot(x_values, gap)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Distance Gap (m)")
    plt.ylim([5, 35])
    
    fig = plt.gcf()
    fig.set_size_inches(11, 5)

def draw_combined_plot(velocity_list, acceleration_list, time_gap_list, distance_gap_list, ttc_list):
    """
    Draw combined plots of multiple metrics
    
    Parameters
    ----------
    velocity_list : list
        List of velocity profiles
    acceleration_list : list
        List of acceleration profiles
    time_gap_list : list
        List of time gap profiles
    distance_gap_list : list
        List of distance gap profiles
    ttc_list : list
        List of TTC profiles
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with combined plots
    """
    fig = plt.figure()
    
    # Velocity subplot
    plt.subplot(511)
    draw_velocity_profile(velocity_list)
    
    # Acceleration subplot
    plt.subplot(512)
    draw_acceleration_profile(acceleration_list)
    
    # Time gap subplot
    plt.subplot(513)
    draw_time_gap_profile(time_gap_list)
    
    # Distance gap subplot
    plt.subplot(514)
    draw_distance_gap_profile(distance_gap_list)
    
    # TTC subplot
    plt.subplot(515)
    draw_ttc_profile(ttc_list)
    
    # Create legend
    labels = []
    for i in range(1, len(velocity_list) + 1):
        if i == 1:
            labels.append('Leading Vehicle, id: 0')
        else:
            labels.append(f'Following Vehicle, id: {i-1}')
    
    fig.legend(labels, loc='upper right')
    
    return fig
