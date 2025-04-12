from collections import deque
from copy import deepcopy
from abc import ABC
from skylink_v2x.utils import velocity_to_speed_3d

class LocalizationManager(ABC):
    """
    Base class for managing localization of an actor within a cooperative autonomous vehicle (CAV) world.
    Stores the position, velocity, and timestamp for localization purposes.
    """
    def __init__(self, 
                 actor, 
                 config, 
                 comm_manager=None):
        """
        Initialize the localization manager.
        
        :param cav_world: The CAV world instance providing timestamp information.
        :param actor: The actor (vehicle, RSU, or drone) being localized.
        :param config: Configuration object containing optional parameters.
        """
        self._actor = actor
        self._actor_id = actor.id
        self._comm_manager = comm_manager
        
        self._position = None
        self._velocity = None
        self._timestamp = None

    def localize(self):
        self._position = self._actor.get_transform()
        self._velocity = self._actor.get_velocity()
        self._timestamp = self._comm_manager.get_timestamp()
        
        localization_results =  {
            'position': self._position,
            'velocity': self._velocity,
            'speed_mps': velocity_to_speed_3d(self._velocity, meters=True),
            'speed_kph': velocity_to_speed_3d(self._velocity),
            'timestamp': self._timestamp
        }

        self._comm_manager.buffer_localization(self._actor_id, localization_results)
        

    def destroy(self):
        """
        Clears the historical data buffers.
        """
        self._position = None
        self._velocity = None
        self._timestamp = None


class VehicleLocalizationManager(LocalizationManager):
    """
    Localization manager for vehicles. Inherits functionality from LocalizationManager.
    """
    pass


class RSULocalizationManager(LocalizationManager):
    """
    Localization manager for Roadside Units (RSUs).
    RSUs do not have velocity data, so the localization method is overridden.
    """
    def localize(self):
        self._position = self._actor.get_transform()
        self._timestamp = self._comm_manager.get_timestamp()
        
        localization_results =  {
            'position': self._position,
            'timestamp': self._timestamp
        }

        self._comm_manager.buffer_localization(self._actor_id, localization_results)

class DroneLocalizationManager(LocalizationManager):
    """
    Localization manager for drones. Inherits functionality from LocalizationManager.
    """
    def localize(self):
        self._position = self._actor.get_transform()
        self._velocity = self._actor.get_velocity()
        self._timestamp = self._comm_manager.get_timestamp()
        
        localization_results =  {
            'position': self._position,
            'velocity': self._velocity,
            'speed_mps': velocity_to_speed_3d(self._velocity, meters=True),
            'speed_kph': velocity_to_speed_3d(self._velocity),
            'timestamp': self._timestamp
        }

        self._comm_manager.buffer_localization(self._actor_id, localization_results)
        
