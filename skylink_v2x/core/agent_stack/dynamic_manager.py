# Author: Fengze Yang <fred.yang@utah.edu>
# License:

from abc import ABC, abstractmethod
from .agent_manager import AgentManager

class DynamicManager(AgentManager, ABC):
    """
    Abstract class for dynamic manager that defines the basic structure of a dynamic manager.

    Attributes
    ----------
    localizer: Localizer
        The localizer of the dynamic manager.
    sim_map: Map
        The map of the dynamic manager.
    dynamic_agent: Vehicle or Drone
        The dynamic agent of the dynamic manager.
    map_manager: MapManager
        The map manager of the dynamic manager.
    type: str
        The type of the dynamic manager.

    Methods
    ----------
    set_destination(self, start_location, end_location, clean: bool, end_reset: bool)
        Set the destination of the dynamic agent.
    """
    def __init__(self, id: int, carla_world=None, config=None):
        super().__init__(id, carla_world=carla_world)
        self.sim_map = None  # Could be a carla.Map or skylink.Map instance
        self.dynamic_agent = None  # Could be a carla.Vehicle or skylink.Drone instance
        self.type = None  # "vehicle" or "drone"


    @abstractmethod
    def set_destination(self, start_location, end_location, clean: bool, end_reset: bool):
        raise NotImplementedError("Subclasses must implement this method")
    