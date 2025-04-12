from abc import ABC, abstractmethod

class ErrorCreator(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def apply_error(self, data):
        return data

class LocalizationError(ErrorCreator):
    def __init__(self, config):
        super(LocalizationError, self).__init__(config)

    def apply_error(self, data):
        return data