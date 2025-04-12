# Author: Fengze Yang <fred.yang@utah.edu>
# License:

from .agent_manager import AgentManager
# from .data_logger import DataLogger
# Assume MLManager is defined elsewhere (stub provided as needed)
from .ml_manager import MLManager


class SimScenario:
    def __init__(self, ml_manager: MLManager=None):
        """
        Initialize the simulation scenario.
        """
        # Containers for dynamic and static managers
        self.dynamic_managers = {
            'vehicle_managers': {},         # Dict of dynamic vehicle managers, {id: manager}
            'drone_managers': {},           # Dict of dynamic drone managers, {id: manager}
            'ego_vehicle_manager_id': None,    # Stores the ID of the ego vehicle manager
            'platooning_dict': {}              # Dict for platooning managers, {pmid: platooning_manager}
        }
        self.static_managers = {
            'rsu_manager_dict': {}             # Dict for RSU managers
        }
        # Load MLManager
        self.ml_manager = ml_manager

        # Simulation step counter
        self.sim_step = 0

        # For co-simulation: mapping of SUMO to Carla vehicle IDs
        self.sumo2carla_ids = {}

        # For co-simulation: mapping of AirSim to Carla drone IDs
        self.airsim2carla_ids = {}

        # Data logger for simulation data
        # TODO: 1. Each agent should have their on logger. 2. Defined by not used.
        # self.data_logger = DataLogger()


    # def update_platooning(self, platooning_manger):
    #     """
    #     Add created platooning.

    #     Parameters
    #     ----------
    #     platooning_manger : opencda object
    #         The platooning manager class.
    #     """
    #     self.dynamic_managers['platooning_dict'].update(
    #         {platooning_manger.pmid: platooning_manger})


    def update_agent_manager(self, agent_manager: AgentManager):
        """
        Add or update an agent manager in the scenario.

        Parameters
        ----------
        agent_manager : AgentManager
            An instance of AgentManager (or its subclass) to be added.
        """
        # Update the combined dictionary
        self.agent_managers[agent_manager.id] = agent_manager

        # Categorize based on type or available attributes
        if hasattr(agent_manager, 'dynamic_agent'):
            if agent_manager.type == 'drone':
                self.dynamic_managers['drone_managers'].update(
                    {agent_manager.id: agent_manager})
            else:
                self.dynamic_managers['vehicle_managers'].update(
                    {agent_manager.id: agent_manager})

                # Set the ego vehicle manager based on a policy, e.g., smallest vehicle ID
                if (self.dynamic_managers['ego_vehicle_manager_id'] is None or 
                    agent_manager.vehicle.id < self.dynamic_managers['ego_vehicle_manager_id']):
                    self.dynamic_managers['ego_vehicle_manager_id'] = agent_manager.vehicle.id
        else:
            # For non-dynamic agents, assume RSU or similar static manager
            self.static_managers['rsu_manager_dict'][agent_manager.id] = agent_manager


    def update_sumo_vehicles(self, sumo2carla_ids: dict):
        """
        Update the SUMO-to-Carla vehicle ID mapping.

        Parameters
        ----------
        sumo2carla_ids : dict
            Key: SUMO vehicle ID
            Value: Carla vehicle ID
        """
        self.sumo2carla_ids.update(sumo2carla_ids)

    
    def update_airsim_drones(self, airsim2carla_ids: dict):
        """
        Update the AirSim-to-Carla drone ID mapping.

        Parameters
        ----------
        sumo2carla_ids : dict
            Key: SUMO vehicle ID
            Value: Carla vehicle ID
        """
        self.airsim2carla_ids.update(airsim2carla_ids)


    def get_agent_managers(self) -> dict:
        """
        Return the combined dictionary of all agent managers.

        Returns
        -------
        self.agent_managers
            A dictionary of all agent managers, indexed by their IDs.
        """
        return self.agent_managers
    

    def locate_agent_manager(self, loc) -> dict:
        """
        Locate an agent manager based on a given location.

        Parameters
        ----------
        loc : carla.Location
            An object representing the location, have attributes x, y, and z.

        Returns
        -------
        target_manager : dict
            A dictionary containing the found agent managers, if any.
        """
        def find_manager(managers, key):
            """
            Helper function to find a manager at a given location.

            Parameters
            ----------
            managers : set
                A set of agent managers to search.
            key : str
                A string key to store the found manager in the target dictionary.

            Raises
            ------
            ValueError
                If multiple managers are found at the same location.
            """
            for manager in managers:
                pos = manager.localizer.get_ego_pos().location

                if target_manager[key] is not None:
                    raise ValueError(f"Multiple {key} found at the same location")
                
                if pos.x == loc.x and pos.y == loc.y:
                    target_manager[key] = manager
                    

        target_manager = {'vehicle_manager': None, 
                          'drone_manager': None, 
                          'rsu_manager': None}

        find_manager(self.dynamic_managers['vehicle_managers'], 'vehicle_manager')
        find_manager(self.dynamic_managers['drone_managers'], 'drone_manager')
        find_manager(self.static_managers['rsu_manager_dict'], 'rsu_manager')

        return target_manager


    def get_ego_vehicle_manager(self) -> object:
        """
        Retrieve the ego vehicle manager (i.e., the primary vehicle).

        Returns
        -------
        AgentManager or None
            The manager corresponding to the ego vehicle, determined as the one 
            with the smallest vehicle ID.
        """
        return self.dynamic_managers['vehicle_managers'][
            self.dynamic_managers['ego_vehicle_manager_id']]


    def count(self):
        """
        Increment the simulation step and update simulation state if needed.
        """
        # TODO: Implement data logging.
        
        self.sim_step += 1


    def remove(self):
        """
        Clean up and remove resources associated with the scenario.
        """
        # Iterate over all agent managers and perform their cleanup routines
        for manager in self.agent_managers.values():
            manager.remove()
        
        # Clear all variables
        self.dynamic_managers.clear()
        self.static_managers.clear()
        self.ml_manager = None
        self.sim_step = 0
        self.sumo2carla_ids.clear()
        self.airsim2carla_ids.clear()
        self.data_logger = None
