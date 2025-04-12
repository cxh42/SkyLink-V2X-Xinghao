from abc import ABC, abstractmethod
import random

class MessageLossCreator(ABC):
    def __init__(self, config):
        self._config = config
        # self.loss_rate = config.get("loss_rate", 0.1)  

    @abstractmethod
    def apply_loss(self, data):
        return data

class PerceptionMessageLoss(MessageLossCreator):
    def __init__(self, config):
        super(PerceptionMessageLoss, self).__init__(config)

    def apply_loss(self, data):
        return data
        # return [msg for msg in data if random.random() >= self.loss_rate]

class LocalizationMessageLoss(MessageLossCreator):
    def __init__(self, config):
        super(LocalizationMessageLoss, self).__init__(config)

    def apply_loss(self, data):
        return data
        # return [msg for msg in data if random.random() >= self.loss_rate]

class MappingMessageLoss(MessageLossCreator):
    def __init__(self, config):
        super(MappingMessageLoss, self).__init__(config)

    def apply_loss(self, data):
        return data
        # return [msg for msg in data if random.random() >= self.loss_rate]

class PlanningMessageLoss(MessageLossCreator):
    def __init__(self, config):
        super(PlanningMessageLoss, self).__init__(config)

    def apply_loss(self, data):
        return data
        # return [msg for msg in data if random.random() >= self.loss_rate]

class ControlMessageLoss(MessageLossCreator):
    def __init__(self, config):
        super(ControlMessageLoss, self).__init__(config)

    def apply_loss(self, data):
        return data
        # return [msg for msg in data if random.random() >= self.loss_rate]

class SafetyMessageLoss(MessageLossCreator):
    def __init__(self, config):
        super(SafetyMessageLoss, self).__init__(config)

    def apply_loss(self, data):
        return data
        # return [msg for msg in data if random.random() >= self.loss_rate]

