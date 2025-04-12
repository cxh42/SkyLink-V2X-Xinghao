from collections import deque

class LatencyCreator:
    def __init__(self, config):
        self._config = config

    def apply_latency(self, data_list: deque):
        return data_list[-1]
    
    