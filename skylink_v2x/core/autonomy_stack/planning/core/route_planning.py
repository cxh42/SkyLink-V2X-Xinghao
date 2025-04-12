# -*- coding: utf-8 -*-
"""
Global route planning utilities
"""

import math
import numpy as np
import networkx as nx
import carla

class RoadOption:
    """Road option enumeration for route planning"""
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class GlobalRouteDatabase:
    """
    Database for retrieving road network information.
    
    Parameters
    ----------
    world_map : carla.Map
        Carla map of the simulation world
    sampling_resolution : float
        Resolution for sampling waypoints
    """

    def __init__(self, world_map, sampling_resolution=1.0):
        self._sampling_resolution = sampling_resolution
        self._map = world_map

    def get_topology(self):
        """
        Get topology of the road network.
        
        Returns
        -------
        list
            List of road segments with their waypoints and connections
        """
        topology = []
        
        # Get topology from map
        for segment in self._map.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            
            # Round coordinates to avoid floating point issues
            x1, y1, z1 = round(l1.x, 0), round(l1.y, 0), round(l1.z, 0)
            x2, y2, z2 = round(l2.x, 0), round(l2.y, 0), round(l2.z, 0)
            
            # Store segment information
            segment_dict = {
                'entry': wp1,
                'exit': wp2,
                'entryxyz': (x1, y1, z1),
                'exitxyz': (x2, y2, z2),
                'path': []
            }
            
            # Sample points along the segment
            end_loc = wp2.transform.location
            if wp1.transform.location.distance(end_loc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(end_loc) > self._sampling_resolution:
                    segment_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                segment_dict['path'].append(wp1.next(self._sampling_resolution)[0])
                
            topology.append(segment_dict)
            
        return topology

    def get_waypoint(self, location):
        """
        Get waypoint at specified location.
        
        Parameters
        ----------
        location : carla.Location
            Location to find waypoint at
            
        Returns
        -------
        carla.Waypoint
            Waypoint at the location
        """
        return self._map.get_waypoint(location)

    def get_resolution(self):
        """
        Get sampling resolution.
        
        Returns
        -------
        float
            Sampling resolution
        """
        return self._sampling_resolution


class GlobalRoutePlanner:
    """
    Path planning in global road network.
    
    Parameters
    ----------
    database : GlobalRouteDatabase
        Database of road network
    """

    def __init__(self, database):
        self._db = database
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None
        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

    def setup(self):
        """
        Build the graph representation of the world map.
        """
        self._topology = self._db.get_topology()
        self._graph, self._id_map, self._road_id_to_edge = self._build_graph()
        self._find_loose_ends()
        self._add_lane_changes()

    def _build_graph(self):
        """
        Build networkx graph from topology.
        
        Returns
        -------
        tuple
            (graph, id_map, road_id_to_edge) graph representation and mappings
        """
        graph = nx.DiGraph()
        id_map = {}  # Maps (x,y,z) to node id
        road_id_to_edge = {}  # Maps road_id to section/lane edges
        
        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id
            
            # Add nodes if they don't exist
            for vertex in [entry_xyz, exit_xyz]:
                if vertex not in id_map:
                    new_id = len(id_map)
                    id_map[vertex] = new_id
                    graph.add_node(new_id, vertex=vertex)
            
            # Map node IDs
            n1, n2 = id_map[entry_xyz], id_map[exit_xyz]
            
            # Create road ID mapping structure
            if road_id not in road_id_to_edge:
                road_id_to_edge[road_id] = {}
            if section_id not in road_id_to_edge[road_id]:
                road_id_to_edge[road_id][section_id] = {}
            road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
            
            # Get entry and exit vectors
            entry_vector = np.array([
                entry_wp.transform.rotation.get_forward_vector().x,
                entry_wp.transform.rotation.get_forward_vector().y,
                entry_wp.transform.rotation.get_forward_vector().z
            ])
            
            exit_vector = np.array([
                exit_wp.transform.rotation.get_forward_vector().x,
                exit_wp.transform.rotation.get_forward_vector().y,
                exit_wp.transform.rotation.get_forward_vector().z
            ])
            
            # Calculate net vector
            net_vector = self._vector_between_locations(
                entry_wp.transform.location,
                exit_wp.transform.location
            )
            
            # Add edge with attributes
            graph.add_edge(
                n1, n2,
                length=len(path) + 1,
                path=path,
                entry_waypoint=entry_wp,
                exit_waypoint=exit_wp,
                entry_vector=entry_vector,
                exit_vector=exit_vector,
                net_vector=net_vector,
                intersection=intersection,
                type=RoadOption.LANEFOLLOW
            )
        
        return graph, id_map, road_id_to_edge

    def _find_loose_ends(self):
        """
        Find and connect loose ends in the road network.
        """
        count_loose_ends = 0
        resolution = self._db.get_resolution()
        
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            
            # Check if this road has already been processed
            if (road_id in self._road_id_to_edge and
                section_id in self._road_id_to_edge[road_id] and
                lane_id in self._road_id_to_edge[road_id][section_id]):
                continue
                
            # Add loose end to graph
            count_loose_ends += 1
            
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = {}
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = {}
            
            n1 = self._id_map[exit_xyz]
            n2 = -1 * count_loose_ends
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
            
            # Follow road to get path
            next_wp = end_wp.next(resolution)
            path = []
            
            while next_wp:
                if (len(next_wp) == 0 or
                    next_wp[0].road_id != road_id or
                    next_wp[0].section_id != section_id or
                    next_wp[0].lane_id != lane_id):
                    break
                path.append(next_wp[0])
                next_wp = next_wp[0].next(resolution)
            
            if path:
                # Add to graph
                n2_xyz = (path[-1].transform.location.x,
                          path[-1].transform.location.y,
                          path[-1].transform.location.z)
                          
                self._graph.add_node(n2, vertex=n2_xyz)
                self._graph.add_edge(
                    n1, n2,
                    length=len(path) + 1,
                    path=path,
                    entry_waypoint=end_wp,
                    exit_waypoint=path[-1],
                    entry_vector=None,
                    exit_vector=None,
                    net_vector=None,
                    intersection=end_wp.is_junction,
                    type=RoadOption.LANEFOLLOW
                )

    def _add_lane_changes(self):
        """
        Add lane change connections to the graph.
        """
        for segment in self._topology:
            left_found, right_found = False, False
            
            for waypoint in segment['path']:
                if segment['entry'].is_junction:
                    continue
                    
                # Check for right lane change
                if (waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and
                        not right_found):
                    right_lane = waypoint.get_right_lane()
                    
                    if (right_lane and
                            right_lane.lane_type == carla.LaneType.Driving and
                            waypoint.road_id == right_lane.road_id):
                        right_wp = right_lane
                        right_road_option = RoadOption.CHANGELANERIGHT
                        right_loc = self._locate_waypoint(right_wp)
                        
                        if right_loc:
                            self._graph.add_edge(
                                self._id_map[segment['entryxyz']],
                                right_loc[0],
                                entry_waypoint=waypoint,
                                exit_waypoint=right_wp,
                                intersection=False,
                                exit_vector=None,
                                path=[],
                                length=100,
                                type=right_road_option,
                                change_waypoint=right_wp
                            )
                            right_found = True
                
                # Check for left lane change
                if (waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and
                        not left_found):
                    left_lane = waypoint.get_left_lane()
                    
                    if (left_lane and
                            left_lane.lane_type == carla.LaneType.Driving and
                            waypoint.road_id == left_lane.road_id):
                        left_wp = left_lane
                        left_road_option = RoadOption.CHANGELANELEFT
                        left_loc = self._locate_waypoint(left_wp)
                        
                        if left_loc:
                            self._graph.add_edge(
                                self._id_map[segment['entryxyz']],
                                left_loc[0],
                                entry_waypoint=waypoint,
                                exit_waypoint=left_wp,
                                intersection=False,
                                exit_vector=None,
                                path=[],
                                length=100,
                                type=left_road_option,
                                change_waypoint=left_wp
                            )
                            left_found = True
                
                if left_found and right_found:
                    break

    def _locate_waypoint(self, waypoint):
        """
        Find the edge corresponding to a waypoint.
        
        Parameters
        ----------
        waypoint : carla.Waypoint
            Waypoint to locate
            
        Returns
        -------
        tuple
            Edge in graph corresponding to waypoint
        """
        road_id = waypoint.road_id
        section_id = waypoint.section_id
        lane_id = waypoint.lane_id
        
        if (road_id in self._road_id_to_edge and
                section_id in self._road_id_to_edge[road_id] and
                lane_id in self._road_id_to_edge[road_id][section_id]):
            return self._road_id_to_edge[road_id][section_id][lane_id]
        
        return None

    def _vector_between_locations(self, loc1, loc2):
        """
        Calculate vector between two locations.
        
        Parameters
        ----------
        loc1 : carla.Location
            First location
        loc2 : carla.Location
            Second location
            
        Returns
        -------
        numpy.ndarray
            Vector from loc1 to loc2
        """
        x = loc2.x - loc1.x
        y = loc2.y - loc1.y
        z = loc2.z - loc1.z
        norm = np.linalg.norm([x, y, z]) + 1e-10
        
        return np.array([x/norm, y/norm, z/norm])

    def _distance_heuristic(self, n1, n2):
        """
        Distance heuristic for A* path finding.
        
        Parameters
        ----------
        n1 : int
            First node ID
        n2 : int
            Second node ID
            
        Returns
        -------
        float
            Euclidean distance between nodes
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])
        l2 = np.array(self._graph.nodes[n2]['vertex'])
        
        return np.linalg.norm(l1 - l2)

    def _find_route(self, origin, destination):
        """
        Find route between origin and destination using A*.
        
        Parameters
        ----------
        origin : carla.Location
            Origin location
        destination : carla.Location
            Destination location
            
        Returns
        -------
        list
            List of node IDs forming the route
        """
        start_edge = self._locate_waypoint(self._db.get_waypoint(origin))
        end_edge = self._locate_waypoint(self._db.get_waypoint(destination))
        
        if not start_edge or not end_edge:
            return None
            
        route = nx.astar_path(
            self._graph,
            source=start_edge[0],
            target=end_edge[0],
            heuristic=self._distance_heuristic,
            weight='length'
        )
        
        route.append(end_edge[1])
        
        return route

    def _analyze_junction(self, index, route, threshold=math.radians(35)):
        """
        Analyze a junction to determine turn direction.
        
        Parameters
        ----------
        index : int
            Index in route
        route : list
            Route as list of node IDs
        threshold : float
            Angle threshold for turn classification
            
        Returns
        -------
        RoadOption
            Turn direction at junction
        """
        decision = None
        
        if index == 0:
            return RoadOption.LANEFOLLOW
            
        previous_node = route[index - 1]
        current_node = route[index]
        next_node = route[index + 1]
        
        next_edge = self._graph.edges[current_node, next_node]
        
        if (self._previous_decision != RoadOption.VOID and
                self._intersection_end_node > 0 and
                self._intersection_end_node != previous_node and
                next_edge['type'] == RoadOption.LANEFOLLOW and
                next_edge['intersection']):
            decision = self._previous_decision
        else:
            self._intersection_end_node = -1
            current_edge = self._graph.edges[previous_node, current_node]
            
            calculate_turn = (
                current_edge['type'] == RoadOption.LANEFOLLOW and
                not current_edge['intersection'] and
                next_edge['type'] == RoadOption.LANEFOLLOW and
                next_edge['intersection']
            )
            
            if calculate_turn:
                # Find end of intersection
                last_node, tail_edge = self._get_last_intersection_edge(index, route)
                self._intersection_end_node = last_node
                
                if tail_edge:
                    next_edge = tail_edge
                
                # Get vectors
                cv = current_edge['exit_vector']
                nv = next_edge['exit_vector']
                
                if cv is None or nv is None:
                    return next_edge['type']
                
                # Check all possible turns
                cross_list = []
                for neighbor in self._graph.successors(current_node):
                    select_edge = self._graph.edges[current_node, neighbor]
                    if select_edge['type'] == RoadOption.LANEFOLLOW:
                        if neighbor != route[index + 1]:
                            sv = select_edge['net_vector']
                            cross_list.append(np.cross(cv, sv)[2])
                
                # Determine turn direction
                next_cross = np.cross(cv, nv)[2]
                deviation = math.acos(np.clip(
                    np.dot(cv, nv) / (np.linalg.norm(cv) * np.linalg.norm(nv)),
                    -1.0, 1.0
                ))
                
                if not cross_list:
                    cross_list.append(0)
                
                if deviation < threshold:
                    decision = RoadOption.STRAIGHT
                elif cross_list and next_cross < min(cross_list):
                    decision = RoadOption.LEFT
                elif cross_list and next_cross > max(cross_list):
                    decision = RoadOption.RIGHT
                elif next_cross < 0:
                    decision = RoadOption.LEFT
                elif next_cross > 0:
                    decision = RoadOption.RIGHT
                else:
                    decision = RoadOption.STRAIGHT
            else:
                decision = next_edge['type']
        
        self._previous_decision = decision
        return decision

    def _get_last_intersection_edge(self, index, route):
        """
        Get the last edge in an intersection.
        
        Parameters
        ----------
        index : int
            Starting index in route
        route : list
            Route as list of node IDs
            
        Returns
        -------
        tuple
            (last_node, last_edge) at the end of intersection
        """
        last_intersection_edge = None
        last_node = None
        
        for i in range(index, len(route) - 1):
            node1, node2 = route[i], route[i + 1]
            edge = self._graph.edges[node1, node2]
            
            if node1 == route[index]:
                last_intersection_edge = edge
                
            if edge['type'] == RoadOption.LANEFOLLOW and edge['intersection']:
                last_intersection_edge = edge
                last_node = node2
            else:
                break
                
        return last_node, last_intersection_edge

    def _find_closest_in_list(self, target_wp, waypoint_list):
        """
        Find the closest waypoint in a list.
        
        Parameters
        ----------
        target_wp : carla.Waypoint
            Target waypoint
        waypoint_list : list
            List of waypoints to search
            
        Returns
        -------
        int
            Index of closest waypoint
        """
        min_distance = float('inf')
        closest_index = -1
        
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(target_wp.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                
        return closest_index

    def trace_route(self, origin, destination):
        """
        Calculate a route between origin and destination.
        
        Parameters
        ----------
        origin : carla.Location
            Origin location
        destination : carla.Location
            Destination location
            
        Returns
        -------
        list
            List of (waypoint, road_option) tuples forming the route
        """
        route_trace = []
        
        # Find route
        route = self._find_route(origin, destination)
        if not route:
            return route_trace
            
        # Get waypoints
        current_waypoint = self._db.get_waypoint(origin)
        destination_waypoint = self._db.get_waypoint(destination)
        resolution = self._db.get_resolution()
        
        # Process each edge in the route
        for i in range(len(route) - 1):
            road_option = self._analyze_junction(i, route)
            edge = self._graph.edges[route[i], route[i + 1]]
            path = []
            
            if edge['type'] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                route_trace.append((current_waypoint, road_option))
                
                # Get next edge
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._locate_waypoint(exit_wp)
                next_edge = self._graph.edges[n1, n2]
                
                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path']) - 1, closest_index + 5)
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                    
                route_trace.append((current_waypoint, road_option))
                
            else:
                # Follow the path for this edge
                path = [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self._find_closest_in_list(current_waypoint, path)
                
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    
                    # Check if we're near destination
                    if len(route) - i <= 2:
                        if waypoint.transform.location.distance(destination) < 2 * resolution:
                            break
                        
                        # Check if we're in the destination lane
                        if (current_waypoint.road_id == destination_waypoint.road_id and
                                current_waypoint.section_id == destination_waypoint.section_id and
                                current_waypoint.lane_id == destination_waypoint.lane_id):
                            
                            # Check if we've passed the destination
                            destination_index = self._find_closest_in_list(destination_waypoint, path)
                            if closest_index > destination_index:
                                break
        
        return route_trace
    