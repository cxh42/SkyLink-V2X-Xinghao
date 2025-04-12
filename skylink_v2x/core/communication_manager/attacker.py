from abc import ABC, abstractmethod

class Attacker(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def attack(self, data):
        return data
    
class PerceptionAttacker(Attacker):
    def __init__(self, config):
        super(PerceptionAttacker, self).__init__(config)

    def attack(self, data):
        return data
    
class LocalizationAttacker(Attacker):
    def __init__(self, config):
        super(LocalizationAttacker, self).__init__(config)

    def attack(self, data):
        return data
    
class MappingAttacker(Attacker):
    def __init__(self, config):
        super(MappingAttacker, self).__init__(config)

    def attack(self, data):
        return data
    
class PlanningAttacker(Attacker):
    def __init__(self, config):
        super(PlanningAttacker, self).__init__(config)

    def attack(self, data):
        return data
    
class ControlAttacker(Attacker):
    def __init__(self, config):
        super(ControlAttacker, self).__init__(config)
        
    def attack(self, data):
        return data
    
class SafetyAttacker(Attacker):
    def __init__(self, config):
        super(SafetyAttacker, self).__init__(config)

    def attack(self, data):
        return data
    