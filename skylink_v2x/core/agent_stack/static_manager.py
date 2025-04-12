# Author: Fengze Yang <fred.yang@utah.edu>
# License:

from abc import ABC

from .agent_manager import AgentManager

class StaticManager(AgentManager, ABC):
    """
    Class for managing static agents. Static agents are agents that do not move.
    """
    def __init__(self, id: int):
        super().__init__(id)
        