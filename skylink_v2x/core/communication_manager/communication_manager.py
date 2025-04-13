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
        if self._perception_buffer is not None:
            self._history_perception.append(self._perception_buffer)
            self._perception_buffer = None
        
    def sync_localization(self):
        if self._localization_buffer is not None:
            self._history_localization.append(self._localization_buffer)
            self._localization_buffer = None
        
    def sync_mapping(self):
        if self._mapping_buffer is not None:
            self._history_mapping.append(self._mapping_buffer)
            self._mapping_buffer = None
        
    def sync_planning(self):
        if self._planning_buffer is not None:
            self._history_planning.append(self._planning_buffer)
            self._planning_buffer = None
        
    def sync_control(self):
        if self._control_buffer is not None:
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
        if agent_id in self._comm_agent_helpers:
            self._comm_agent_helpers[agent_id].buffer_perception(data)
    
    def buffer_localization(self, agent_id, data):
        # TODO: deepcopy is not enabled (RuntimeError: Pickling of "carla.libcarla.Transform" instances is not enabled (http://www.boost.org/libs/python/doc/v2/pickle.html))
        if agent_id in self._comm_agent_helpers:
            self._comm_agent_helpers[agent_id].buffer_localization(data)
        
    def buffer_mapping(self, agent_id, data):
        # 避免对大型数据结构使用深拷贝，这可能会占用大量内存
        if agent_id not in self._comm_agent_helpers:
            return
            
        # 如果数据有自己的浅拷贝方法，尽量使用
        if hasattr(data, 'copy') and callable(getattr(data, 'copy')):
            try:
                copied_data = data.copy()
            except Exception as e:
                print(f"复制数据失败: {str(e)}, 使用原始数据")
                copied_data = data
        else:
            # 对于不能修改的情况，仍保留deepcopy以维持功能
            try:
                copied_data = deepcopy(data)
            except Exception as e:
                # 如果deepcopy失败，尝试使用原始数据
                print(f"deepcopy失败: {str(e)}, 使用原始数据")
                copied_data = data
            
        self._comm_agent_helpers[agent_id].buffer_mapping(copied_data)
        
    def buffer_planning(self, agent_id, data):
        # TODO: deepcopy is not enabled (RuntimeError: Pickling of "carla.libcarla.VehicleControl" instances is not enabled (http://www.boost.org/libs/python/doc/v2/pickle.html))
        if agent_id in self._comm_agent_helpers:
            self._comm_agent_helpers[agent_id].buffer_planning(data)
        
    def buffer_control(self, agent_id, data):
        # 使用和mapping数据相同的逻辑
        if agent_id not in self._comm_agent_helpers:
            return
            
        if hasattr(data, 'copy') and callable(getattr(data, 'copy')):
            try:
                copied_data = data.copy()
            except Exception as e:
                print(f"复制数据失败: {str(e)}, 使用原始数据")
                copied_data = data
        else:
            try:
                copied_data = deepcopy(data)
            except Exception as e:
                print(f"deepcopy失败: {str(e)}, 使用原始数据")
                copied_data = data
            
        self._comm_agent_helpers[agent_id].buffer_control(copied_data)
        
    def sync_perception(self):
        for agent_id in list(self._connected_agents.keys()):
            if agent_id in self._comm_agent_helpers:
                self._comm_agent_helpers[agent_id].sync_perception()
            
    def sync_localization(self):
        for agent_id in list(self._connected_agents.keys()):
            if agent_id in self._comm_agent_helpers:
                self._comm_agent_helpers[agent_id].sync_localization()
            
    def sync_mapping(self):
        for agent_id in list(self._connected_agents.keys()):
            if agent_id in self._comm_agent_helpers:
                self._comm_agent_helpers[agent_id].sync_mapping()
            
    def sync_planning(self):
        for agent_id in list(self._connected_agents.keys()):
            if agent_id in self._comm_agent_helpers:
                self._comm_agent_helpers[agent_id].sync_planning()
            
    def sync_control(self):
        for agent_id in list(self._connected_agents.keys()):
            if agent_id in self._comm_agent_helpers:
                self._comm_agent_helpers[agent_id].sync_control()
            
    def get_perception_data(self):
        data_dict = dict()
        for agent_id in list(self._connected_agents.keys()):
            if agent_id not in self._comm_agent_helpers:
                continue
                
            history_data = self._comm_agent_helpers[agent_id].get_perception_data()
            
            # 安全检查：确保历史数据非空
            if not history_data or len(history_data) == 0:
                data_dict[agent_id] = None
                continue
                
            try:
                data = self.latency_creator.apply_latency(history_data)
                data = self.perception_attacker.attack(data)
                data = self.perception_message_loss.apply_loss(data)
                data_dict[agent_id] = data
            except Exception as e:
                print(f"处理感知数据时出错: {str(e)}")
                data_dict[agent_id] = None
                
        return data_dict
    
    def get_localization_data(self):
        data_dict = dict()
        for agent_id in list(self._connected_agents.keys()):
            if agent_id not in self._comm_agent_helpers:
                continue
                
            history_data = self._comm_agent_helpers[agent_id].get_localization_data()
            
            # 安全检查：确保历史数据非空
            if not history_data or len(history_data) == 0:
                data_dict[agent_id] = None
                continue
                
            try:
                data = self.latency_creator.apply_latency(history_data)
                data = self.localization_error_creator.apply_error(data)
                data = self.localization_attacker.attack(data)
                data = self.localization_message_loss.apply_loss(data)
                data_dict[agent_id] = data
            except Exception as e:
                print(f"处理定位数据时出错: {str(e)}")
                data_dict[agent_id] = None
                
        return data_dict
    
    def get_mapping_data(self):
        data_dict = dict()
        for agent_id in list(self._connected_agents.keys()):
            if agent_id not in self._comm_agent_helpers:
                continue
                
            history_data = self._comm_agent_helpers[agent_id].get_mapping_data()
            
            # 安全检查：确保历史数据非空
            if not history_data or len(history_data) == 0:
                data_dict[agent_id] = None
                continue
                
            try:
                data = self.latency_creator.apply_latency(history_data)
                data = self.mapping_attacker.attack(data)
                data = self.mapping_message_loss.apply_loss(data)
                data_dict[agent_id] = data
            except Exception as e:
                print(f"处理地图数据时出错: {str(e)}")
                data_dict[agent_id] = None
                
        return data_dict
    
    def get_planning_data(self):
        data_dict = dict()
        for agent_id in list(self._connected_agents.keys()):
            if agent_id not in self._comm_agent_helpers:
                continue
                
            history_data = self._comm_agent_helpers[agent_id].get_planning_data()
            
            # 安全检查：确保历史数据非空
            if not history_data or len(history_data) == 0:
                data_dict[agent_id] = None
                continue
                
            try:
                data = self.latency_creator.apply_latency(history_data)
                data = self.planning_attacker.attack(data)
                data = self.planning_message_loss.apply_loss(data)
                data_dict[agent_id] = data
            except Exception as e:
                print(f"处理规划数据时出错: {str(e)}")
                data_dict[agent_id] = None
                
        return data_dict
    
    def get_control_data(self):
        data_dict = dict()
        for agent_id in list(self._connected_agents.keys()):
            if agent_id not in self._comm_agent_helpers:
                continue
                
            history_data = self._comm_agent_helpers[agent_id].get_control_data()
            
            # 安全检查：确保历史数据非空
            if not history_data or len(history_data) == 0:
                data_dict[agent_id] = None
                continue
                
            try:
                data = self.latency_creator.apply_latency(history_data)
                data = self.control_attacker.attack(data)
                data = self.control_message_loss.apply_loss(data)
                data_dict[agent_id] = data
            except Exception as e:
                print(f"处理控制数据时出错: {str(e)}")
                data_dict[agent_id] = None
                
        return data_dict
    
    def log_data(self):
        # 完全重写日志记录功能，避免索引错误
        try:
            if not self._connected_agents or not hasattr(self, '_data_logger'):
                return  # 没有代理或没有数据记录器，直接返回
                
            agent_ids = list(self._connected_agents.keys())
            
            # 为每个代理记录数据
            for agent_id in agent_ids:
                if agent_id not in self._comm_agent_helpers or agent_id not in self._connected_agents:
                    continue
                    
                try:
                    self._data_logger.set_path(self.start_time, agent_id, self._timestamp)
                    agent_comm_helper = self._comm_agent_helpers[agent_id]
                    agent_manager = self._connected_agents[agent_id]
                    self._data_logger.save_data(agent_comm_helper=agent_comm_helper, 
                                              agent_manager=agent_manager, 
                                              agent_id=agent_id)
                except Exception as e:
                    print(f"保存代理 {agent_id} 数据时出错: {str(e)}")
            
            # 尝试安全地保存检测到的对象
            try:
                # 使用第一个代理的感知数据
                if agent_ids and agent_ids[0] in self._comm_agent_helpers:
                    agent_comm_helper = self._comm_agent_helpers[agent_ids[0]]
                    perception_queue = agent_comm_helper.get_perception_data()
                    
                    # 确保队列不为空且有至少一个元素
                    if perception_queue and len(perception_queue) > 0:
                        try:
                            # 获取最后一个元素
                            last_perception = perception_queue[-1]
                            
                            # 检查是否是字典并且包含detected_objects键
                            if isinstance(last_perception, dict) and 'detected_objects' in last_perception:
                                try:
                                    detected_objects = last_perception['detected_objects']
                                    self._data_logger.save_objects(detected_objects=detected_objects)
                                except Exception as object_save_error:
                                    print(f"保存检测对象时出错: {str(object_save_error)}")
                        except IndexError:
                            # 安全处理索引错误
                            print("感知队列索引错误，跳过保存检测对象")
            except Exception as perception_error:
                print(f"处理感知数据时出错: {str(perception_error)}")
                
        except Exception as general_error:
            print(f"日志记录过程中发生一般错误: {str(general_error)}")
    
    def clean_agent_history(self, max_history=2):
        """
        保留最近的历史数据，清理旧数据
        """
        try:
            for agent_id in list(self._comm_agent_helpers.keys()):
                if agent_id not in self._comm_agent_helpers:
                    continue
                    
                agent_helper = self._comm_agent_helpers[agent_id]
                
                # 安全清理历史数据
                try:
                    # 对每个队列应用安全清理
                    for history_queue in [
                        agent_helper._history_perception,
                        agent_helper._history_localization,
                        agent_helper._history_mapping,
                        agent_helper._history_planning,
                        agent_helper._history_control
                    ]:
                        # 只有当队列长度超过最大历史记录数时才进行清理
                        try:
                            while len(history_queue) > max_history:
                                history_queue.popleft()
                        except Exception as queue_error:
                            print(f"清理队列时出错: {str(queue_error)}")
                except Exception as helper_error:
                    print(f"清理代理 {agent_id} 历史数据时出错: {str(helper_error)}")
        except Exception as e:
            print(f"清理代理历史数据的过程中发生一般错误: {str(e)}")
    
    def clear_resources(self):
        """清理所有资源，释放内存"""
        try:
            # 使用列表复制以避免在迭代过程中修改字典
            agent_ids = list(self._comm_agent_helpers.keys())
            
            for agent_id in agent_ids:
                if agent_id in self._comm_agent_helpers:
                    try:
                        if hasattr(self._comm_agent_helpers[agent_id], 'clear_resources'):
                            self._comm_agent_helpers[agent_id].clear_resources()
                    except Exception as e:
                        print(f"清理代理 {agent_id} 资源时出错: {str(e)}")
            
            # 清空所有字典
            self._comm_agent_helpers.clear()
            self._connected_agents.clear()
            
            # 清除数据记录器
            if hasattr(self, '_data_logger'):
                try:
                    del self._data_logger
                except:
                    pass
        except Exception as e:
            print(f"清理所有资源时出错: {str(e)}")

    def tick(self):
        """
        模拟通信管理器一个时间步。
        """
        try:
            # 执行日志记录（如果启用）
            if self._log_data and hasattr(self, '_data_logger'):
                try:
                    self.log_data()
                except Exception as log_error:
                    print(f"记录数据时出错: {str(log_error)}")
            
            # 每10个时间步清理一次历史数据
            if self._timestamp % 10 == 0:
                try:
                    self.clean_agent_history()
                except Exception as clean_error:
                    print(f"清理历史数据时出错: {str(clean_error)}")
            
            # 每30个时间步执行一次垃圾回收
            if self._timestamp % 30 == 0:
                try:
                    gc.collect()
                except Exception as gc_error:
                    print(f"执行垃圾回收时出错: {str(gc_error)}")
                    
        except Exception as tick_error:
            print(f"执行时间步时出错: {str(tick_error)}")
        
        # 增加时间戳
        self._timestamp += 1
    
    def get_timestamp(self):
        return self._timestamp