# -*- coding: utf-8 -*-
"""
Vehicle planning manager module
"""

import math
import random
import sys

import numpy as np
import carla

from skylink_v2x.core.autonomy_stack.planning.core.collision_detection import CollisionDetector
from skylink_v2x.core.autonomy_stack.planning.core.trajectory_generation import LocalTrajectoryPlanner
from skylink_v2x.core.autonomy_stack.planning.core.route_planning import GlobalRoutePlanner
from skylink_v2x.core.autonomy_stack.planning.core.route_planning import GlobalRouteDatabase
from skylink_v2x.core.autonomy_stack.planning.visualization.debug_helper import PlanningDebugHelper
from skylink_v2x.core.autonomy_stack.planning.managers.base_manager import BasePlanningManager
from skylink_v2x.core.autonomy_stack.planning.core.common_utils import (
    get_vehicle_speed, clamp_positive, calculate_angle_distance, 
    compute_distance, distance_to_vehicle
)
from agents.navigation.behavior_agent import BehaviorAgent


class VehiclePlanningManagerRuleBased(BasePlanningManager):
    """
    Planning manager for autonomous vehicles.
    
    This class handles route planning, behavior decision making, and trajectory generation
    for autonomous vehicles navigating complex environments.
    
    Parameters
    ----------
    vehicle : carla.Vehicle
        The vehicle to be controlled
    carla_map : carla.Map
        Map of the simulation world
    config : dict
        Configuration dictionary
    """
    
    def __init__(self, vehicle, carla_map, config):
        super().__init__(config)
        self.vehicle = vehicle
        # Position and speed from localization
        self._ego_pos = None
        self._ego_speed = 0.0
        self._map = carla_map



        # Speed parameters
        self.max_speed = config['max_speed']
        self.tailgate_speed = config['tailgate_speed']
        self.speed_limit_distance = config['speed_lim_dist']
        self.speed_decrease = config['speed_decrease']

        # Safety parameters
        self.safety_time = config['safety_time']
        self.emergency_factor = config['emergency_param']
        self.braking_distance = 0
        self.time_to_collision = 1000
        
        # Collision detection
        self._collision_detector = CollisionDetector(
            time_ahead=config['collision_time_ahead']
        )
        
        self.ignore_traffic_light = config['ignore_traffic_light']
        self.overtake_allowed = config['overtake_allowed']
        self.overtake_allowed_origin = config['overtake_allowed']
        self.overtake_counter = 0
        self.hazard_flag = False

        # Route planning
        self._global_planner = None
        self.start_waypoint = None
        self.end_waypoint = None
        self._sampling_resolution = config['sample_resolution']

        # Traffic control state
        self.light_state = "Red"
        self.light_id_to_ignore = -1
        self.stop_sign_wait_counter = 0

        # Local planning
        self._local_planner = LocalTrajectoryPlanner(
            self, carla_map, config['local_planner']
        )

        # Special behavior flags
        self.car_following_flag = False
        self.lane_change_allowed = True
        self.destination_push_flag = 0

        # Perception whitelist for cooperative agents
        self.white_list = []
        self.obstacle_vehicles = []
        self.objects = {}

        # Debugging
        self.debug_helper = PlanningDebugHelper(self.vehicle.id)
        self.debug = config.get('debug', False)
        
        # Global route storage
        self.initial_global_route = None
        

    def add_to_white_list(self, vehicle_manager):
        """
        Add vehicle manager to whitelist (will be ignored in collision checks)
        
        Parameters
        ----------
        vehicle_manager : object
            Vehicle manager to be added to whitelist
        """
        self.white_list.append(vehicle_manager)

    def _filter_white_list_vehicles(self, obstacles):
        """
        Remove whitelisted vehicles from obstacle list
        
        Parameters
        ----------
        obstacles : list
            List of detected obstacle vehicles
            
        Returns
        -------
        list
            Filtered list of obstacles (without whitelisted vehicles)
        """
        filtered_obstacles = []

        for obstacle in obstacles:
            is_whitelisted = False
            obstacle_x = obstacle.get_location().x
            obstacle_y = obstacle.get_location().y

            obstacle_waypoint = self._map.get_waypoint(obstacle.get_location())
            obstacle_lane_id = obstacle_waypoint.lane_id

            for vm in self.white_list:
                position = vm.v2x_manager.get_ego_pos()
                vm_x = position.location.x
                vm_y = position.location.y

                whitelist_waypoint = self._map.get_waypoint(position.location)
                whitelist_lane_id = whitelist_waypoint.lane_id

                # Different lane ID means different vehicle
                if obstacle_lane_id != whitelist_lane_id:
                    continue

                # Check if positions are close enough
                if abs(vm_x - obstacle_x) <= 3.0 and abs(vm_y - obstacle_y) <= 3.0:
                    is_whitelisted = True
                    break
                    
            if not is_whitelisted:
                filtered_obstacles.append(obstacle)

        return filtered_obstacles

    def set_destination(self, start_location, end_location, clean=False, 
                       end_reset=True, clean_history=False):
        """
        Set destination for route planning
        
        Parameters
        ----------
        start_location : carla.Location
            Starting location
        end_location : carla.Location
            Destination location
        clean : bool
            Whether to clean existing waypoints
        end_reset : bool
            Whether to reset the end waypoint
        clean_history : bool
            Whether to clean waypoint history
        """
        if clean:
            self._local_planner.get_waypoints_queue().clear()
            self._local_planner.get_trajectory().clear()
            self._local_planner.get_waypoint_buffer().clear()
            
        if clean_history:
            self._local_planner.get_history_buffer().clear()

        self.start_waypoint = self._map.get_waypoint(start_location)

        # Ensure start waypoint is behind the vehicle
        if self._ego_pos:
            current_loc = self._ego_pos.location
            current_yaw = self._ego_pos.rotation.yaw
            _, angle = calculate_angle_distance(
                self.start_waypoint.transform.location, current_loc, current_yaw
            )

            # Adjust waypoint until it's behind the vehicle
            while angle > 90:
                self.start_waypoint = self.start_waypoint.next(1)[0]
                _, angle = calculate_angle_distance(
                    self.start_waypoint.transform.location, current_loc, current_yaw
                )

        end_waypoint = self._map.get_waypoint(end_location)
        if end_reset:
            self.end_waypoint = end_waypoint

        # Calculate route
        route_trace = self._plan_route(self.start_waypoint, end_waypoint)

        # Store initial route if not already stored
        if self.initial_global_route is None:
            self.initial_global_route = route_trace

        # Set route in local planner
        self._local_planner.set_global_plan(route_trace, clean)

    def get_local_planner(self):
        """
        Get the local trajectory planner
        
        Returns
        -------
        LocalTrajectoryPlanner
            The local trajectory planner
        """
        return self._local_planner

    def reroute(self, spawn_points):
        """
        Replan route when approaching destination
        
        Parameters
        ----------
        spawn_points : list
            List of possible spawn points for new destination
        """
        if self.debug:
            print("Target almost reached, setting new destination...")
            
        # Shuffle spawn points and select a new destination
        random.shuffle(spawn_points)
        new_start = self._local_planner.waypoints_queue[-1][0].transform.location
        
        # Ensure new destination is different from start
        destination = spawn_points[0].location
        if spawn_points[0].location == new_start:
            destination = spawn_points[1].location
            
        if self.debug:
            print(f"New destination: {destination}")

        # Set new destination
        self.set_destination(new_start, destination)

    def _plan_route(self, start_waypoint, end_waypoint):
        """
        Plan route from start to end waypoint
        
        Parameters
        ----------
        start_waypoint : carla.Waypoint
            Starting waypoint
        end_waypoint : carla.Waypoint
            Destination waypoint
            
        Returns
        -------
        list
            List of waypoints from start to end
        """
        # Initialize global planner if needed
        if self._global_planner is None:
            world = self.vehicle.get_world()
            database = GlobalRouteDatabase(
                world.get_map(), 
                sampling_resolution=self._sampling_resolution
            )
            planner = GlobalRoutePlanner(database)
            planner.setup()
            self._global_planner = planner

        # Plan route
        route = self._global_planner.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location
        )

        return route

    def handle_traffic_light(self, waypoint):
        """
        Handle traffic lights and stop signs
        
        Parameters
        ----------
        waypoint : carla.Waypoint
            Current waypoint
            
        Returns
        -------
        int
            0 if no stop needed, 1 if stop needed
        """
        light_id = -1
        if self.vehicle.get_traffic_light() is not None:
            light_id = self.vehicle.get_traffic_light().id

        # Handle stop sign wait counter
        if 60 <= self.stop_sign_wait_counter < 240:
            self.stop_sign_wait_counter += 1
        elif self.stop_sign_wait_counter >= 240:
            self.stop_sign_wait_counter = 0

        # Handle red light state
        if self.light_state == "Red":
            # Handle stop signs (light_id == -1)
            if light_id == -1:
                # Wait at stop sign for ~1-2 seconds
                if self.stop_sign_wait_counter < 60:
                    self.stop_sign_wait_counter += 1
                    return 1  # Stop needed
                else:
                    return 0  # No stop needed
            
            # Regular traffic light handling
            if not waypoint.is_junction and (self.light_id_to_ignore != light_id or light_id == -1):
                return 1  # Stop needed
            elif waypoint.is_junction and light_id != -1:
                self.light_id_to_ignore = light_id
                
        # Reset ignored light ID when passing a different light
        if self.light_id_to_ignore != light_id:
            self.light_id_to_ignore = -1
            
        return 0  # No stop needed

    def check_collision(self, path_x, path_y, path_yaw, waypoint, adjacent_check=False):
        """
        Check for potential collisions along the path
        
        Parameters
        ----------
        path_x : list
            X coordinates of the path
        path_y : list
            Y coordinates of the path
        path_yaw : list
            Yaw angles of the path
        waypoint : carla.Waypoint
            Current waypoint
        adjacent_check : bool
            Whether checking adjacent lane
            
        Returns
        -------
        tuple
            (collision_detected, closest_vehicle, distance)
        """
        def distance_to_waypoint(vehicle):
            """Calculate distance from vehicle to waypoint"""
            return vehicle.get_location().distance(waypoint.transform.location)

        collision_detected = False
        min_distance = float('inf')
        closest_vehicle = None

        # Check each obstacle vehicle
        for vehicle in self.obstacle_vehicles:
            collision_free = self._collision_detector.collision_circle_check(
                path_x, path_y, path_yaw, vehicle, 
                self._ego_speed / 3.6, self._map,
                adjacent_check=adjacent_check
            )
            
            if not collision_free:
                collision_detected = True
                
                # Calculate distance accounting for vehicle length
                distance = max(0, distance_to_waypoint(vehicle) - 3)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle = vehicle

        return collision_detected, closest_vehicle, min_distance

    def manage_overtaking(self, obstacle_vehicle):
        """
        Determine if overtaking is possible and plan accordingly
        
        Parameters
        ----------
        obstacle_vehicle : carla.Vehicle
            The vehicle to potentially overtake
            
        Returns
        -------
        bool
            True if overtaking not possible/needed, False if overtaking initiated
        """
        # Get obstacle vehicle information
        obstacle_loc = obstacle_vehicle.get_location()
        obstacle_waypoint = self._map.get_waypoint(obstacle_loc)

        # Check if lane change is allowed
        left_turn = obstacle_waypoint.left_lane_marking.lane_change
        right_turn = obstacle_waypoint.right_lane_marking.lane_change

        # Get left and right waypoints
        left_waypoint = obstacle_waypoint.get_left_lane()
        right_waypoint = obstacle_waypoint.get_right_lane()

        # Check for left overtake possibility
        if ((left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both) and
                left_waypoint and
                obstacle_waypoint.lane_id * left_waypoint.lane_id > 0 and
                left_waypoint.lane_type == carla.LaneType.Driving):
            
            # Generate and check path for left overtake
            rx, ry, ryaw = self._collision_detector.adjacent_lane_collision_check(
                ego_loc=self._ego_pos.location, 
                target_wpt=left_waypoint,
                carla_map=self._map,
                overtake=True, 
                world=self.vehicle.get_world()
            )
            
            collision, _, _ = self.check_collision(
                rx, ry, ryaw, 
                self._map.get_waypoint(self._ego_pos.location), 
                adjacent_check=True
            )
            
            if not collision:
                print("Left overtake initiated")
                self.overtake_counter = 100
                
                # Plan the overtake maneuver
                next_waypoint_list = left_waypoint.next(self._ego_speed / 3.6 * 6)
                if not next_waypoint_list:
                    return True

                next_waypoint = next_waypoint_list[0]
                left_waypoint = left_waypoint.next(5)[0]
                
                self.set_destination(
                    left_waypoint.transform.location,
                    next_waypoint.transform.location,
                    clean=True,
                    end_reset=False
                )
                return False

        # Check for right overtake possibility
        if ((right_turn == carla.LaneChange.Right or right_turn == carla.LaneChange.Both) and
                right_waypoint and
                obstacle_waypoint.lane_id * right_waypoint.lane_id > 0 and
                right_waypoint.lane_type == carla.LaneType.Driving):
                
            # Generate and check path for right overtake
            rx, ry, ryaw = self._collision_detector.adjacent_lane_collision_check(
                ego_loc=self._ego_pos.location,
                target_wpt=right_waypoint,
                overtake=True,
                carla_map=self._map,
                world=self.vehicle.get_world()
            )
            
            collision, _, _ = self.check_collision(
                rx, ry, ryaw, 
                self._map.get_waypoint(self._ego_pos.location), 
                adjacent_check=True
            )
            
            if not collision:
                print("Right overtake initiated")
                self.overtake_counter = 100
                
                # Plan the overtake maneuver
                next_waypoint_list = right_waypoint.next(self._ego_speed / 3.6 * 6)
                if not next_waypoint_list:
                    return True

                next_waypoint = next_waypoint_list[0]
                right_waypoint = right_waypoint.next(5)[0]
                
                self.set_destination(
                    right_waypoint.transform.location,
                    next_waypoint.transform.location,
                    clean=True,
                    end_reset=False
                )
                return False

        return True  # Overtaking not possible or not initiated

    def check_lane_change_safety(self):
        """
        Check if lane change is safe
        
        Returns
        -------
        bool
            True if lane change is safe, False otherwise
        """
        ego_waypoint = self._map.get_waypoint(self._ego_pos.location)
        ego_lane_id = ego_waypoint.lane_id
        target_waypoint = None

        # Find the closest waypoint on the adjacent lane
        for waypoint in self.get_local_planner().get_waypoint_buffer():
            if waypoint[0].lane_id != ego_lane_id:
                target_waypoint = waypoint[0]
                break
                
        if not target_waypoint:
            return False

        # Generate and check path for lane change
        rx, ry, ryaw = self._collision_detector.adjacent_lane_collision_check(
            ego_loc=self._ego_pos.location,
            target_wpt=target_waypoint,
            overtake=False,
            carla_map=self._map,
            world=self.vehicle.get_world()
        )
        
        collision, _, _ = self.check_collision(
            rx, ry, ryaw, 
            self._map.get_waypoint(self._ego_pos.location), 
            adjacent_check=True
        )
        
        return not collision

    def calculate_following_speed(self, lead_vehicle, distance, target_speed=None):
        """
        Calculate appropriate following speed
        
        Parameters
        ----------
        lead_vehicle : carla.Vehicle
            The vehicle to follow
        distance : float
            Distance to the lead vehicle
        target_speed : float, optional
            Desired target speed if specified
            
        Returns
        -------
        float
            Calculated target speed
        """
        if not target_speed:
            target_speed = self.max_speed - self.speed_limit_distance

        # Get lead vehicle speed
        lead_vehicle_speed = get_vehicle_speed(lead_vehicle)

        # Calculate time to collision
        delta_v = max(1, (self._ego_speed - lead_vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        self.time_to_collision = ttc

        # Adjust speed based on safety criteria
        if 0 < ttc < self.safety_time:
            # Within safety time, slow down
            target_speed = min(
                clamp_positive(lead_vehicle_speed - self.speed_decrease),
                target_speed
            )
        else:
            # Otherwise, match lead vehicle speed with small margin
            if lead_vehicle_speed == 0:
                target_speed = 0
            else:
                target_speed = min(lead_vehicle_speed + 1, target_speed)
                
        return target_speed

    def is_near_intersection(self, objects, waypoint_buffer):
        """
        Check if vehicle is approaching an intersection
        
        Parameters
        ----------
        objects : dict
            Dictionary of detected objects
        waypoint_buffer : deque
            Buffer of upcoming waypoints
            
        Returns
        -------
        bool
            True if near intersection, False otherwise
        """
        # Check if any traffic light is near future waypoints
        for traffic_light in objects.get('traffic_lights', []):
            for waypoint, _ in waypoint_buffer:
                distance = traffic_light.get_location().distance(waypoint.transform.location)
                if distance < 20:
                    return True
        return False

    def is_close_to_destination(self):
        """
        Check if vehicle is close to destination
        
        Returns
        -------
        bool
            True if close to destination, False otherwise
        """
        if not self._ego_pos or not self.end_waypoint:
            return False
            
        return (abs(self._ego_pos.location.x - self.end_waypoint.transform.location.x) <= 10 and
                abs(self._ego_pos.location.y - self.end_waypoint.transform.location.y) <= 10)

    def validate_lane_change(self, lane_change_allowed, collision_detector_enabled, curvatures):
        """
        Determine if lane change should be allowed
        
        Parameters
        ----------
        lane_change_allowed : bool
            Previous lane change permission
        collision_detector_enabled : bool
            Whether collision detection is enabled
        curvatures : list
            List of path curvatures
            
        Returns
        -------
        bool
            True if lane change is allowed, False otherwise
        """
        # Disallow lane change on high-curvature roads
        if len(curvatures) > 2 and np.mean(np.abs(np.array(curvatures))) > 0.04:
            return False
            
        # Lane change requirements
        lane_change_check_needed = (
            collision_detector_enabled and
            self.get_local_planner().lane_id_change and
            self.get_local_planner().lane_lateral_change and
            self.overtake_counter <= 0 and
            not self.destination_push_flag
        )
        
        if lane_change_check_needed:
            lane_change_allowed = lane_change_allowed and self.check_lane_change_safety()
            if not lane_change_allowed:
                print("Lane change not allowed due to safety concerns")
                
        return lane_change_allowed

    def calculate_push_destination(self, ego_waypoint, is_intersection):
        """
        Calculate temporary destination for push operation
        
        Parameters
        ----------
        ego_waypoint : carla.Waypoint
            Current ego vehicle waypoint
        is_intersection : bool
            Whether vehicle is near intersection
            
        Returns
        -------
        carla.Waypoint
            The temporary push destination
        """
        waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
        
        # Different strategies based on intersection
        if is_intersection:
            # Use future waypoint at intersections
            reset_index = len(waypoint_buffer) // 2
            reset_target = waypoint_buffer[reset_index][0].next(
                max(self._ego_speed / 3.6, 10.0)
            )[0]
        else:
            # Use direct forward path otherwise
            reset_target = ego_waypoint.next(
                max(self._ego_speed / 3.6 * 3, 10.0)
            )[0]
            
        if self.debug:
            print(f'Vehicle id: {self.vehicle.id}: destination pushed forward due to '
                  f'potential collision, reset destination: '
                  f'{reset_target.transform.location.x}, '
                  f'{reset_target.transform.location.y}, '
                  f'{reset_target.transform.location.z}')
                  
        return reset_target
    
    def update_information(self, ego_pos, ego_speed, objects):
        """
        Update the planning manager with latest perception and localization data
        
        Parameters
        ----------
        ego_pos : carla.Transform
            Current vehicle position and orientation
        ego_speed : float
            Current speed in km/h
        objects : dict
            Dictionary of detected objects by type
        """
        # Update localization information
        self._ego_speed = ego_speed
        self._ego_pos = ego_pos
        self.braking_distance = self._ego_speed / 3.6 * self.emergency_factor
        
        # Update local planner
        self._local_planner.update_information(ego_pos, ego_speed)

        # Update perception information
        self.objects = objects
        obstacle_vehicles = objects['vehicles']
        self.obstacle_vehicles = self._filter_white_list_vehicles(obstacle_vehicles)

        # Update debugging helper
        self.debug_helper.update(ego_speed, self.time_to_collision)

        # Update traffic light state
        if self.ignore_traffic_light:
            self.light_state = "Green"
        else:
            self.light_state = str(self.vehicle.get_traffic_light_state())
            
    
    def run_step(self, 
                 target_speed=None, 
                 localization_data=None,
                 perception_data=None,
                 mapping_data=None,
                 collision_detector_enabled=True, 
                 lane_change_allowed=True):
        
        #TODO Rule-based planning has not been tested yet
        
        # Get current position
        ego_location = self._ego_pos.location
        ego_waypoint = self._map.get_waypoint(ego_location)
        waypoint_buffer = self.get_local_planner().get_waypoint_buffer()
        
        # Reset time to collision
        self.time_to_collision = 1000
        
        # Manage overtake counter
        if self.overtake_counter > 0:
            self.overtake_counter -= 1
            
        # Manage destination push flag
        if self.destination_push_flag > 0:
            self.destination_push_flag -= 1
            
        # Check for intersections
        is_intersection = self.is_near_intersection(self.objects, waypoint_buffer)

        # Check destination reached
        if self.is_close_to_destination():
            print('Destination reached, simulation complete')
            sys.exit(0)

        # Handle traffic lights and stop signs
        if self.handle_traffic_light(ego_waypoint) != 0:
            return 0, None

        # Reset destination when route complete
        if (len(self.get_local_planner().get_waypoints_queue()) == 0 and
                len(self.get_local_planner().get_waypoint_buffer()) <= 2):
            if self.debug:
                print('Destination reset!')
                
            # Reset flags
            self.overtake_allowed = self.overtake_allowed_origin
            self.lane_change_allowed = True
            self.destination_push_flag = 0
            
            # Set destination back to original end
            self.set_destination(
                ego_location,
                self.end_waypoint.transform.location,
                clean=True,
                clean_history=True
            )

        # Disable overtaking at intersections
        if is_intersection:
            self.overtake_allowed = False
        else:
            self.overtake_allowed = self.overtake_allowed_origin

        # Generate path
        path_x, path_y, path_curvature, path_yaw = self._local_planner.generate_path()

        # Check lane change permission
        self.lane_change_allowed = self.validate_lane_change(
            lane_change_allowed, 
            collision_detector_enabled,
            path_curvature
        )

        # Collision detection
        hazard_detected = False
        if collision_detector_enabled:
            hazard_detected, obstacle_vehicle, distance = self.check_collision(
                path_x, path_y, path_yaw, ego_waypoint
            )
            
        car_following_needed = False

        # Reset hazard flag if no hazard detected
        if not hazard_detected:
            self.hazard_flag = False

        # Handle lane change safety issues (push case)
        if (not self.lane_change_allowed and
                self._local_planner.potential_curved_road and
                not self.destination_push_flag and
                self.overtake_counter <= 0):
                
            # Disable overtaking during push
            self.overtake_allowed = False
            
            # Calculate push destination
            reset_target = self.calculate_push_destination(ego_waypoint, is_intersection)
            
            # Set push timeout
            self.destination_push_flag = 90
            
            # Set temporary destination
            self.set_destination(
                ego_location,
                reset_target.transform.location,
                clean=True,
                end_reset=False
            )
            
            # Regenerate path with new destination
            path_x, path_y, path_curvature, path_yaw = self._local_planner.generate_path()

        # Handle vehicle blocking ahead - car following case
        elif hazard_detected and (
                not self.overtake_allowed or
                self.overtake_counter > 0 or
                self._local_planner.potential_curved_road):
            car_following_needed = True
            
        # Handle overtaking case
        elif hazard_detected and self.overtake_allowed and self.overtake_counter <= 0:
            obstacle_speed = get_vehicle_speed(obstacle_vehicle)
            obstacle_lane = self._map.get_waypoint(obstacle_vehicle.get_location()).lane_id
            ego_lane = self._map.get_waypoint(self._ego_pos.location).lane_id
            
            # Only consider overtaking when in same lane
            if ego_lane == obstacle_lane:
                # Set hazard flag for potential joining transitions
                self.hazard_flag = hazard_detected
                
                # Only overtake if we're faster than obstacle
                if self._ego_speed >= obstacle_speed - 5:
                    car_following_needed = self.manage_overtaking(obstacle_vehicle)
                else:
                    car_following_needed = True

        # Car following behavior
        if car_following_needed:
            # Emergency stop if too close
            if distance < max(self.braking_distance, 3):
                return 0, None

            # Calculate car following speed
            target_speed = self.calculate_following_speed(
                obstacle_vehicle, distance, target_speed
            )
            
            # Run local planner with adjusted speed
            target_speed, target_loc = self._local_planner.run_step(
                path_x, path_y, path_curvature, target_speed=target_speed
            )
            
            return target_speed, target_loc

        # Normal driving behavior
        default_speed = self.max_speed - self.speed_limit_distance if not target_speed else target_speed
        target_speed, target_loc = self._local_planner.run_step(
            path_x, path_y, path_curvature, target_speed=default_speed
        )
        
        return target_speed, target_loc
        
    


    
class VehiclePlanningManagerCarla:
    
    def __init__(self, vehicle, carla_map, config):
        
        self.planner = BehaviorAgent(vehicle=vehicle, map_inst=carla_map)
    
    def update_information(self):
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
        self.planner.update_information()
    
    def set_destination(self, start_location, end_location):
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
        self.planner.set_destination(start_location=start_location, end_location=end_location)
    
    def run_step(self):
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
        control = self.planner.run_step()
        return control


        
class VehiclePlanningManager:
    
    def __init__(self, vehicle, carla_map, config, comm_manager, autonomy_option='carla'):
        self.autonomy_option = autonomy_option
        self.agent_id = vehicle.id
        if autonomy_option == 'carla':
            self.planner = VehiclePlanningManagerCarla(vehicle, carla_map, config)
        elif autonomy_option == 'rule_based':
            raise NotImplementedError("Rule-based planning is not yet tested")
            self.planner = VehiclePlanningManagerRuleBased(vehicle, carla_map, config)
        else:
            raise ValueError(f"Invalid planning option: {autonomy_option}")
        self._comm_manager = comm_manager
        
    
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
        if self.autonomy_option == 'carla':
            self.planner.update_information()
        elif self.autonomy_option == 'rule_based':
            self.planner.update_information(position, speed, objects)
        else:
            raise ValueError(f"Invalid planning option: {self.autonomy_option}")
    
    def set_destination(self, start_location, end_location, clean=False, end_reset=True):
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
        if self.autonomy_option == 'carla':
            self.planner.set_destination(start_location=start_location, end_location=end_location)
        elif self.autonomy_option == 'rule_based':
            self.planner.set_destination(start_location=start_location, end_location=end_location, clean=clean, end_reset=end_reset)
        else:
            raise ValueError(f"Invalid planning option: {self.autonomy_option}")
    
    def run_step(self, 
                 target_speed=None,
                 localization_data=None,
                 perception_data=None,
                 mapping_data=None,):
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
        if self.autonomy_option == 'carla':
            control = self.planner.run_step()
            planning_outputs = control
        elif self.autonomy_option == 'rule_based':
            waypoints = self.planner.run_step(target_speed=target_speed)
            planning_outputs = waypoints
        else:
            raise ValueError(f"Invalid planning option: {self.autonomy_option}")
        
        self._comm_manager.buffer_planning(self.agent_id, planning_outputs)
        return planning_outputs        
        
        

        
