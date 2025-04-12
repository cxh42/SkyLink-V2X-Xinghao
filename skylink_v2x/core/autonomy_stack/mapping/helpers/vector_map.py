"""
Vector map generation utilities for structured road network data.
"""
import time
import numpy as np
import json

def calculate_lane_heading(centerline):
    """Calculate heading at each point of a lane centerline."""
    headings = []
    for i in range(len(centerline) - 1):
        current, next_point = centerline[i], centerline[i + 1]
        dx, dy = next_point[0] - current[0], next_point[1] - current[1]
        heading = np.arctan2(dy, dx)
        headings.append(heading)
    
    # Copy the last heading for the last point
    if headings:
        headings.append(headings[-1])
    return headings

def calculate_lane_curvature(centerline, headings):
    """Calculate curvature at each point of a lane centerline."""
    curvatures = [0.0]  # First point has zero curvature
    
    for i in range(1, len(centerline) - 1):
        # Use heading change between consecutive segments as curvature
        angle_diff = headings[i] - headings[i-1]
        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate segment length
        segment_length = np.linalg.norm(centerline[i][:2] - centerline[i-1][:2])
        if segment_length > 0.001:  # Avoid division by zero
            curvature = angle_diff / segment_length
            curvatures.append(curvature)
        else:
            curvatures.append(0.0)
    
    # Copy the last curvature for the last point
    if len(curvatures) > 1:
        curvatures.append(curvatures[-1])
    return curvatures


