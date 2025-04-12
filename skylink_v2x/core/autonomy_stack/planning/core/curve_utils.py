# -*- coding: utf-8 -*-
"""
Curve and spline utilities for trajectory generation
"""

import math
import numpy as np
import bisect

class Spline:
    """
    Cubic spline interpolation class.
    
    Parameters
    ----------
    x : array_like
        X coordinates of points to interpolate
    y : array_like
        Y coordinates of points to interpolate
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.nx = len(x)
        h = np.diff(x)
        
        # Calculate a coefficients
        self.a = [yi for yi in y]
        
        # Calculate c coefficients
        A = self._calc_matrix_A(h)
        B = self._calc_matrix_B(h)
        self.c = np.linalg.solve(A, B)
        
        # Calculate b and d coefficients
        self.b, self.d = [], []
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            
            self.b.append(
                (self.a[i + 1] - self.a[i]) / h[i] - 
                h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            )

    def calc(self, t):
        """
        Calculate interpolated value at point t
        
        Parameters
        ----------
        t : float
            Point at which to evaluate the spline
            
        Returns
        -------
        float
            Interpolated value at t, or None if t is outside range
        """
        if t < self.x[0] or t > self.x[-1]:
            return None
            
        i = self._search_index(t)
        dx = t - self.x[i]
        
        return self.a[i] + self.b[i] * dx + self.c[i] * dx**2 + self.d[i] * dx**3

    def calc_derivative(self, t):
        """
        Calculate first derivative at point t
        
        Parameters
        ----------
        t : float
            Point at which to evaluate the derivative
            
        Returns
        -------
        float
            First derivative at t, or None if t is outside range
        """
        if t < self.x[0] or t > self.x[-1]:
            return None
            
        i = self._search_index(t)
        dx = t - self.x[i]
        
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx**2

    def calc_second_derivative(self, t):
        """
        Calculate second derivative at point t
        
        Parameters
        ----------
        t : float
            Point at which to evaluate the second derivative
            
        Returns
        -------
        float
            Second derivative at t, or None if t is outside range
        """
        if t < self.x[0] or t > self.x[-1]:
            return None
            
        i = self._search_index(t)
        dx = t - self.x[i]
        
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

    def _search_index(self, x):
        """Find segment index containing x"""
        return bisect.bisect(self.x, x) - 1

    def _calc_matrix_A(self, h):
        """Calculate matrix A for spline coefficients"""
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
            
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        
        return A

    def _calc_matrix_B(self, h):
        """Calculate matrix B for spline coefficients"""
        B = np.zeros(self.nx)
        
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - \
                       3.0 * (self.a[i + 1] - self.a[i]) / h[i]
                       
        return B


class Spline2D:
    """
    2D cubic spline class for trajectory generation.
    
    Parameters
    ----------
    x : array_like
        X coordinates of points to interpolate
    y : array_like
        Y coordinates of points to interpolate
    """

    def __init__(self, x, y):
        self.s = self._calc_cumulative_distance(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def _calc_cumulative_distance(self, x, y):
        """Calculate cumulative distance along the curve"""
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        
        s = [0]
        s.extend(np.cumsum(self.ds))
        
        return s

    def calc_position(self, s):
        """
        Calculate position at parameter s
        
        Parameters
        ----------
        s : float
            Parameter value along curve
            
        Returns
        -------
        tuple
            (x, y) coordinates at parameter s
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)
        
        return x, y

    def calc_curvature(self, s):
        """
        Calculate curvature at parameter s
        
        Parameters
        ----------
        s : float
            Parameter value along curve
            
        Returns
        -------
        float
            Curvature at parameter s
        """
        dx = self.sx.calc_derivative(s)
        ddx = self.sx.calc_second_derivative(s)
        dy = self.sy.calc_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        
        curvature = (ddy * dx - ddx * dy) / ((dx**2 + dy**2)**(3/2))
        
        return curvature

    def calc_yaw(self, s):
        """
        Calculate yaw angle at parameter s
        
        Parameters
        ----------
        s : float
            Parameter value along curve
            
        Returns
        -------
        float
            Yaw angle at parameter s
        """
        dx = self.sx.calc_derivative(s)
        dy = self.sy.calc_derivative(s)
        
        return math.atan2(dy, dx)


def generate_spline_path(x, y, ds=0.1):
    """
    Generate a smooth path from waypoints
    
    Parameters
    ----------
    x : array_like
        X coordinates of waypoints
    y : array_like
        Y coordinates of waypoints
    ds : float
        Distance between interpolated points
        
    Returns
    -------
    tuple
        (rx, ry, ryaw, rk, s) coordinates, yaw angles, curvatures, and parameters
    """
    sp = Spline2D(x, y)
    s = list(np.arange(sp.s[0], sp.s[-1], ds))
    
    rx, ry, ryaw, rk = [], [], [], []
    
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
    
    return rx, ry, ryaw, rk, s
