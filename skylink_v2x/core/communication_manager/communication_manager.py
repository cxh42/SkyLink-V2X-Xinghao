from collections import deque
import weakref
import carla
import numpy as np
from datetime import datetime
import gc  # 导入垃圾回收模块
from copy import deepcopy, copy
from skylink_v2x.core.communication_manager.attacker import (
    PerceptionAttacker, LocalizationAttacker, MappingAttacker, 
    PlanningAttacker, ControlAttacker
)
from skylink_v2x.core.communication_manager.message_loss_creator import (
    PerceptionMessageLoss, LocalizationMessageLoss, MappingMessageLoss, 
    PlanningMessageLoss, ControlMessageLoss
)
from skylink_v2x.core.communication_manager.error_creator import LocalizationError
from skylink_v2x.core.communication_manager.latency_creator import LatencyCreator
from skylink_v2x.core.communication_manager.data_logger import DataLogger

class CommAgentHelper:
    
    def __init__(self, comm_config, agent_id, position=None, velocity=None, comm_range=np.inf, agent_type=None):
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._position = position
        self._velocity = velocity
        self._comm_range = comm_range
        
        # 减小数据缓冲区大小，从默认的100减少到10
        data_buffer_size = getattr(comm_config, 'data_buffer_size', 10)
        
        # Recent data transmissions with a configurable maximum size to simulate latency
        self._history_perception = deque(maxlen=data_buffer_size)
        self._history_localization = deque(maxlen=data_buffer_size)
        self._history_mapping = deque(maxlen=data_buffer_size)
        self._history_planning = deque(maxlen=data_buffer_size)
        self._history_control = deque(maxlen=data_buffer_size)
        
        # Data buffer to store the most recent data. It is used for synchronization of all agents.
        self._perception_buffer = None
        self._localization_buffer = None
        self._mapping_buffer = None
        self._planning_buffer = None
        self._control_buffer = None
        
    def update_info(self, position, velocity):
        self._position = position
        self._velocity = velocity
        
    def get_agent_type(self):
        """Returns the type of the agent."""
        return self._agent_type
        
    def get_position(self):
        """Returns the current position of the agent."""
        return self._position

    def get_velocity(self):
        """Returns the current velocity of the agent."""
        return self._velocity

    def sync_perception(self):
        self._history_perception.append(self._perception_buffer)
        self._perception_buffer = None
        
    def sync_localization(self):
        self._history_localization.append(self._localization_buffer)
        self._localization_buffer = None
        
    def sync_mapping(self):
        self._history_mapping.append(self._mapping_buffer)
        self._mapping_buffer = None
        
    def sync_planning(self):
        self._history_planning.append(self._planning_buffer)
        self._planning_buffer = None
        
    def sync_control(self):
        self._history_control.append(self._control_buffer)
        self._control_buffer = None
        
    def get_perception_data(self):
        return self._history_perception
    
    def get_localization_data(self):
        return self._history_localization
    
    def get_mapping_data(self):
        return self._history_mapping
    
    def get_planning_data(self):
        return self._history_planning
    
    def get_control_data(self):
        return self._history_control
    
    def buffer_perception(self, data):
        self._perception_buffer = data
        
    def buffer_localization(self, data):
        self._localization_buffer = data
        
    def buffer_mapping(self, data):
        self._mapping_buffer = data
        
    def buffer_planning(self, data):
        self._planning_buffer = data
        
    def buffer_control(self, data):
        self._control_buffer = data
    
    def clear_resources(self):
        """清理资源，释放内存"""
        # 清空所有缓冲区
        self._history_perception.clear()
        self._history_localization.clear()
        self._history_mapping.clear()
        self._history_planning.clear()
        self._history_control.clear()
        
        # 删除当前的缓冲数据
        self._perception_buffer = None
        self._localization_buffer = None
        self._mapping_buffer = None
        self._planning_buffer = None
        self._control_buffer = None


class CommunicationManager:
    """
    CommunicationManager manages vehicle-to-everything (V2X) communication within a cooperative autonomous vehicle (CAV) environment.
    It handles data exchange, applies latency effects, simulates message loss, and injects adversarial attacks.
    """

    def __init__(self, comm_config):
        """
        Initializes the CommunicationManager.

        :param cav_world: Reference to the cooperative autonomous vehicle world.
        :param comm_config: Configuration dictionary containing V2X communication settings.
        :param agent_id: Unique identifier for the agent (vehicle).
        """
        self._comm_config = comm_config  # Communication configuration settings
        # self._comm_range = getattr(self._comm_config, '_comm_range', np.inf)  # Communication range settings
        self._connected_agents = dict()  # Set of connected agents
        self._comm_agent_helpers = dict()  # Set of communication agent helpers
        self._timestamp = 0
        
        self.perception_attacker = PerceptionAttacker(getattr(self._comm_config, 'perception_attacker', None))
        self.localization_attacker = LocalizationAttacker(getattr(self._comm_config, 'localization_attacker', None))
        self.mapping_attacker = MappingAttacker(getattr(self._comm_config, 'mapping_attacker', None))
        self.planning_attacker = PlanningAttacker(getattr(self._comm_config, 'planning_attacker', None))
        self.control_attacker = ControlAttacker(getattr(self._comm_config, 'control_attacker', None))
        
        self.perception_message_loss = PerceptionMessageLoss(getattr(self._comm_config, 'perception_message_loss', None))
        self.localization_message_loss = LocalizationMessageLoss(getattr(self._comm_config, 'localization_message_loss', None))
        self.mapping_message_loss = MappingMessageLoss(getattr(self._comm_config, 'mapping_message_loss', None))
        self.planning_message_loss = PlanningMessageLoss(getattr(self._comm_config, 'planning_message_loss', None))
        self.control_message_loss = ControlMessageLoss(getattr(self._comm_config, 'control_message_loss', None))
        
        self.localization_error_creator = LocalizationError(getattr(self._comm_config, 'localization_error_creator', None))
        
        # Latency simulation
        self.latency_creator = LatencyCreator(getattr(self._comm_config, 'latency', 0))
        data_logger_config = getattr(comm_config, 'data_logger', None)
        self._log_data = data_logger_config is not None
        if self._log_data:
            self._data_logger = DataLogger(data_logger_config)
            self.start_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        
                
    def register(self, agent_manager):
        
        agent_id = agent_manager.get_id()
        self._connected_agents[agent_id] = agent_manager
        self._comm_agent_helpers[agent_id] = CommAgentHelper(comm_config=self._comm_config, 
                                                             agent_id=agent_id,
                                                             comm_range=agent_manager.get_comm_range(),
                                                             agent_type=agent_manager.type)

    def buffer_perception(self, agent_id, data):
        # TODO: deepcopy is not enabled (RuntimeError: Pickling of "carla.libcarla.Vehicle" instances is not enabled (http://www.boost.org/libs/python/doc/v2/pickle.html))
        self._comm_agent_helpers[agent_id].buffer_perception(data)
    
    def buffer_localization(self, agent_id, data):
        # TODO: deepcopy is not enabled (RuntimeError: Pickling of "carla.libcarla.Transform" instances is not enabled (http://www.boost.org/libs/python/doc/v2/pickle.html))
        self._comm_agent_helpers[agent_id].buffer_localization(data)
        
    def buffer_mapping(self, agent_id, data):
        # 避免对大型数据结构使用深拷贝，这可能会占用大量内存
        # 如果数据有自己的浅拷贝方法，尽量使用
        if hasattr(data, 'copy') and callable(getattr(data, 'copy')):
            copied_data = data.copy()
        else:
            # 对于不能修改的情况，仍保留deepcopy以维持功能
            copied_data = deepcopy(data)
            
        self._comm_agent_helpers[agent_id].buffer_mapping(copied_data)
        
    def buffer_planning(self, agent_id, data):
        # TODO: deepcopy is not enabled (RuntimeError: Pickling of "carla.libcarla.VehicleControl" instances is not enabled (http://www.boost.org/libs/python/doc/v2/pickle.html))
        self._comm_agent_helpers[agent_id].buffer_planning(data)
        
    def buffer_control(self, agent_id, data):
        # 使用和mapping数据相同的逻辑
        if hasattr(data, 'copy') and callable(getattr(data, 'copy')):
            copied_data = data.copy()
        else:
            copied_data = deepcopy(data)
            
        self._comm_agent_helpers[agent_id].buffer_control(copied_data)
        
    def sync_perception(self):
        for agent_id in self._connected_agents.keys():
            self._comm_agent_helpers[agent_id].sync_perception()
            
    def sync_localization(self):
        for agent_id in self._connected_agents.keys():
            self._comm_agent_helpers[agent_id].sync_localization()
            
    def sync_mapping(self):
        for agent_id in self._connected_agents.keys():
            self._comm_agent_helpers[agent_id].sync_mapping()
            
    def sync_planning(self):
        for agent_id in self._connected_agents.keys():
            self._comm_agent_helpers[agent_id].sync_planning()
            
    def sync_control(self):
        for agent_id in self._connected_agents.keys():
            self._comm_agent_helpers[agent_id].sync_control()
            
    def get_perception_data(self):
        data_dict = dict()
        for agent_id in self._connected_agents.keys():
            history_data = self._comm_agent_helpers[agent_id].get_perception_data()
            data = self.latency_creator.apply_latency(history_data)
            data = self.perception_attacker.attack(data)
            data = self.perception_message_loss.apply_loss(data)
            data_dict[agent_id] = data
        # TODO: Fuse perception data from all agents (This should actually done by the perception manager)
        return data_dict
    
    def get_localization_data(self):
        data_dict = dict()
        for agent_id in self._connected_agents.keys():
            history_data = self._comm_agent_helpers[agent_id].get_localization_data()
            data = self.latency_creator.apply_latency(history_data)
            data = self.localization_error_creator.apply_error(data)
            data = self.localization_attacker.attack(data)
            data = self.localization_message_loss.apply_loss(data)
            data_dict[agent_id] = data
        return data_dict
    
    def get_mapping_data(self):
        data_dict = dict()
        for agent_id in self._connected_agents.keys():
            history_data = self._comm_agent_helpers[agent_id].get_mapping_data()
            data = self.latency_creator.apply_latency(history_data)
            data = self.mapping_attacker.attack(data)
            data = self.mapping_message_loss.apply_loss(data)
            data_dict[agent_id] = data
        return data_dict
    
    def get_planning_data(self):
        data_dict = dict()
        for agent_id in self._connected_agents.keys():
            history_data = self._comm_agent_helpers[agent_id].get_planning_data()
            data = self.latency_creator.apply_latency(history_data)
            data = self.planning_attacker.attack(data)
            data = self.planning_message_loss.apply_loss(data)
            data_dict[agent_id] = data
        return data_dict
    
    def get_control_data(self):
        data_dict = dict()
        for agent_id in self._connected_agents.keys():
            history_data = self._comm_agent_helpers[agent_id].get_control_data()
            data = self.latency_creator.apply_latency(history_data)
            data = self.control_attacker.attack(data)
            data = self.control_message_loss.apply_loss(data)
            data_dict[agent_id] = data
        return data_dict
    
    def log_data(self):
        for agent_id in self._connected_agents:
            self._data_logger.set_path(self.start_time, agent_id, self._timestamp)
            agent_comm_helper = self._comm_agent_helpers[agent_id]
            agent_manager = self._connected_agents[agent_id]
            self._data_logger.save_data(agent_comm_helper=agent_comm_helper, 
                                        agent_manager=agent_manager, 
                                        agent_id=agent_id)
            # self._connected_agents[agent_id].perception_manager.visualize_sensors()
            
            
        # Load perception objects from any of the object (here we use the last one)
        detected_objects = agent_comm_helper.get_perception_data()[-1]['detected_objects']
        self._data_logger.save_objects(detected_objects=detected_objects)
    
    def clean_agent_history(self, max_history=2):
        """
        保留最近的历史数据，清理旧数据
        """
        for agent_id in self._comm_agent_helpers:
            agent_helper = self._comm_agent_helpers[agent_id]
            
            # 只保留最近的max_history条历史数据
            if len(agent_helper._history_perception) > max_history:
                while len(agent_helper._history_perception) > max_history:
                    agent_helper._history_perception.popleft()
                    
            if len(agent_helper._history_localization) > max_history:
                while len(agent_helper._history_localization) > max_history:
                    agent_helper._history_localization.popleft()
                    
            if len(agent_helper._history_mapping) > max_history:
                while len(agent_helper._history_mapping) > max_history:
                    agent_helper._history_mapping.popleft()
                    
            if len(agent_helper._history_planning) > max_history:
                while len(agent_helper._history_planning) > max_history:
                    agent_helper._history_planning.popleft()
                    
            if len(agent_helper._history_control) > max_history:
                while len(agent_helper._history_control) > max_history:
                    agent_helper._history_control.popleft()
    
    def clear_resources(self):
        """清理所有资源，释放内存"""
        for agent_id in list(self._comm_agent_helpers.keys()):
            if hasattr(self._comm_agent_helpers[agent_id], 'clear_resources'):
                self._comm_agent_helpers[agent_id].clear_resources()
        
        self._comm_agent_helpers.clear()
        self._connected_agents.clear()
        
        # 清除其他引用
        if hasattr(self, '_data_logger'):
            del self._data_logger

    def tick(self):
        """
        模拟通信管理器一个时间步。
        """
        if self._log_data:
            self.log_data()
            
        # 每10个时间步清理一次历史数据
        if self._timestamp % 10 == 0:
            self.clean_agent_history()
            
        # 每30个时间步执行一次垃圾回收
        if self._timestamp % 30 == 0:
            gc.collect()
            
        self._timestamp += 1
    
    def get_timestamp(self):
        return self._timestamp