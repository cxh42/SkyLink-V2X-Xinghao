# -*- coding: utf-8 -*-
"""
Scenario testing: Single vehicle driving in the customized 2 lane highway map.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import traceback
import carla
import gc

from skylink_v2x.scenario_manager.scenario_manager import ScenarioManager
from skylink_v2x.core.agent_stack.sim_scenario import SimScenario
from skylink_v2x.scenario_manager.yaml_utils import add_current_time


def run_scenario(opt, scenario_params):
    # 初始化为空，确保在任何情况下都有定义
    agent_dict = {}
    bg_veh_list = []
    scenario_manager = None
    
    try:
        scenario_params = add_current_time(scenario_params)
        
        # 创建CAV世界
        cav_world = SimScenario(opt.apply_ml)
        
        # 创建场景管理器
        scenario_manager = ScenarioManager(scenario_params,
                                           opt.apply_ml,
                                           opt.version,
                                           cav_world=cav_world)

        if opt.record:
            try:
                scenario_manager.client.start_recorder(f"{scenario_params['scenario_name']}.log", True)
                print(f"开始记录到文件: {scenario_params['scenario_name']}.log")
            except Exception as e:
                print(f"启动记录器失败: {str(e)}")

        try:
            # 尝试创建代理管理器
            agent_dict = scenario_manager.create_agent_managers()
            
            # 检查是否成功创建至少一个代理
            if not agent_dict:
                print("警告: 没有成功创建任何代理，退出场景")
                return
                
            # 创建背景交通
            traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()
        except Exception as e:
            print(f"场景初始化失败: {str(e)}")
            traceback.print_exc()
            # 提前结束
            return

        spectator = scenario_manager.carla_world.get_spectator()
        
        # 添加帧数计数器
        frame_count = 0
        
        # 添加最大帧数检查
        max_frames = scenario_params.get('max_frames', None)
        if max_frames:
            print(f"设置最大帧数限制: {max_frames}")
            
        try:
            while True:
                try:
                    scenario_manager.tick()
                except Exception as e:
                    print(f"场景tick失败: {str(e)}")
                    traceback.print_exc()
                    break
                
                # 检查cav1是否存在
                if 'cav1' not in agent_dict:
                    print("警告: cav1不存在，无法设置观察者位置")
                else:
                    # 设置观察者位置
                    try:
                        transform = agent_dict['cav1'].vehicle.get_transform()
                        spectator.set_transform(carla.Transform(
                            transform.location + carla.Location(z=70),
                            carla.Rotation(pitch=-90)))
                    except Exception as e:
                        print(f"设置观察者位置失败: {str(e)}")
                
                # 执行场景步骤，使用异常处理
                try:
                    scenario_manager.run_step()
                except Exception as e:
                    print(f"场景步骤执行失败: {str(e)}")
                    traceback.print_exc()
                    break  # 执行失败时退出循环
                
                # 增加帧数计数
                frame_count += 1
                
                # 每100帧执行一次垃圾回收
                if frame_count % 100 == 0:
                    try:
                        gc.collect()
                        print(f"当前帧: {frame_count}, 执行了垃圾回收")
                    except Exception as e:
                        print(f"垃圾回收失败: {str(e)}")
                
                # 检查是否达到最大帧数
                if max_frames and frame_count >= max_frames:
                    print(f"已达到最大帧数限制: {max_frames}")
                    break
        except KeyboardInterrupt:
            print("用户中断执行")
        except Exception as e:
            print(f"场景执行出错: {str(e)}")
            traceback.print_exc()

    except Exception as e:
        print(f"初始化场景过程中出错: {str(e)}")
        traceback.print_exc()
        
    finally:
        print("正在清理场景资源...")
        
        # 首先停止记录
        if opt.record and scenario_manager:
            try:
                scenario_manager.client.stop_recorder()
                print("已停止记录器")
            except Exception as e:
                print(f"停止记录器失败: {str(e)}")
        
        # 关闭场景管理器
        if scenario_manager:
            try:
                scenario_manager.close()
                print("已关闭场景管理器")
            except Exception as e:
                print(f"关闭场景管理器失败: {str(e)}")
        
        # 清理代理资源
        for k, v in list(agent_dict.items()):
            try:
                v.remove()
                print(f"已移除代理: {k} ({v.type}, ID: {v.agent_id})")
            except Exception as e:
                print(f"移除代理 {k} 失败: {str(e)}")
        agent_dict.clear()
        
        # 清理背景车辆
        for i, v in enumerate(bg_veh_list):
            try:
                v.destroy()
            except Exception as e:
                print(f"移除背景车辆 {i} 失败: {str(e)}")
                
        # 强制垃圾回收
        try:
            gc.collect()
            print("已执行最终垃圾回收")
        except Exception as e:
            print(f"最终垃圾回收失败: {str(e)}")
            
        print("场景资源清理完成")