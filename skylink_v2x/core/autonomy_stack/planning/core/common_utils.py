# -*- coding: utf-8 -*-
"""
Common utility functions for planning
"""

import math
import numpy as np
import carla

def get_vehicle_speed(vehicle):
    """
    Calculate vehicle speed in km/h
    
    Parameters
    ----------
    vehicle : carla.Vehicle
        Vehicle to get speed for
        
    Returns
    -------
    float
        Vehicle speed in km/h
    """
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def clamp_positive(value):
    """
    Ensure value is positive or zero
    
    Parameters
    ----------
    value : float
        Input value
        
    Returns
    -------
    float
        Positive value or zero
    """
    return max(0.0, value)

def calculate_angle_distance(target_location, current_location, orientation):
    """
    Calculate distance and angle between two locations
    
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
        (distance, angle) between locations
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

def compute_distance(location_1, location_2):
    """
    Compute distance between two locations
    
    Parameters
    ----------
    location_1 : carla.Location
        First location
    location_2 : carla.Location
        Second location
        
    Returns
    -------
    float
        Distance between locations
    """
    dx = location_2.x - location_1.x
    dy = location_2.y - location_1.y
    dz = location_2.z - location_1.z
    
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def distance_to_vehicle(waypoint, vehicle_transform):
    """
    Calculate distance between waypoint and vehicle
    
    Parameters
    ----------
    waypoint : carla.Waypoint or carla.Transform
        Waypoint or transform
    vehicle_transform : carla.Transform
        Vehicle transform
        
    Returns
    -------
    float
        Distance between waypoint and vehicle
    """
    if hasattr(waypoint, 'transform'):
        location = waypoint.transform.location
    else:
        location = waypoint.location
    
    dx = location.x - vehicle_transform.location.x
    dy = location.y - vehicle_transform.location.y
    
    return math.sqrt(dx*dx + dy*dy)

def draw_trajectory_points(world, points, color=None, size=0.1, z=0.0, lifetime=0.1):
    """
    Draw trajectory points for debugging
    
    Parameters
    ----------
    world : carla.World
        Simulation world
    points : list
        List of points to draw
    color : carla.Color
        Color for points
    size : float
        Size of points
    z : float
        Z offset
    lifetime : float
        How long points should remain visible
    """
    if color is None:
        color = carla.Color(255, 0, 0)
    
    for point in points:
        if isinstance(point, tuple):
            if hasattr(point[0], 'transform'):
                location = point[0].transform.location
            else:
                location = point[0].location
        else:
            if hasattr(point, 'transform'):
                location = point.transform.location
            else:
                location = point.location
        
        world.debug.draw_point(
            carla.Location(location.x, location.y, location.z + z),
            size=size,
            color=color,
            life_time=lifetime
        )