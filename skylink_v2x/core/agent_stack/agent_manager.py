# Author: Fengze Yang <fred.yang@utah.edu>
# License:


import uuid
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np


class AgentManager(ABC):
    """
    Abstract class for agent manager that defines the basic structure of an agent manager.
    
    Attributes
    ----------
    id: str
        The unique identifier of the agent manager.
    perception_manager: PerceptionManager
        The perception manager of the agent manager.
    data_logger: DataLogger
        The data logger of the agent manager.
    v2x_manager: V2XManager
        The V2X manager of the agent manager.
    curr_time: datetime
        The current time of the agent manager.
        
    Methods
    ----------
    update(self)
        Update the agent manager.
    run(self)
        Run the agent manager.
    remove(self)
        Remove the agent manager.
    """
    def __init__(self, id: int, carla_world=None, config=None):
        self.type = ''
        self.agent_id = id
        self.manager_id = int(uuid.uuid1())
        self.world = carla_world
        self.localizer = None
        self.perception_manager = None
        self.curr_time = datetime.now()
        v2x_config = getattr(config, 'v2x', None)
        self._comm_range = getattr(v2x_config, 'comm_range', np.inf)


    # @abstractmethod
    # def update(self):
    #     raise NotImplementedError("Subclasses must implement this method")


    # @abstractmethod
    # def run(self):
    #     raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def remove(self):
        raise NotImplementedError("Subclasses must implement this method")
    

    def get_id(self):
        return self.agent_id
        
        
    def get_comm_range(self):
        return self._comm_range
    