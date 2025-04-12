# -*- coding: utf-8 -*-
"""
Scenario testing: Single vehicle dring in the customized 2 lane highway map.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import carla

from skylink_v2x.scenario_manager.scenario_manager import ScenarioManager

from skylink_v2x.core.agent_stack.sim_scenario import SimScenario
# from skylink_v2x.scenario_testing.evaluations.evaluate_manager import \
#     EvaluationManager
from skylink_v2x.scenario_manager.yaml_utils import \
    add_current_time


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)
        # current_path = os.path.dirname(os.path.realpath(__file__))
        # xodr_path = os.path.abspath(os.path.join(current_path, '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr'))

        # create CAV world
        cav_world = SimScenario(opt.apply_ml)
        # create scenario manager
        scenario_manager = ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder(f"{scenario_params['scenario_name']}.log", True)

        

        
            
        

        agent_dict = scenario_manager.create_agent_managers()
        # create background traffic in carla
        traffic_manager, bg_veh_list = scenario_manager.create_traffic_carla()
        

        # create evaluation manager
        # eval_manager = \
        #     EvaluationManager(scenario_manager.cav_world,
        #                       script_name='single_2lanefree_carla',
        #                       current_time=scenario_params['current_time'])

        spectator = scenario_manager.carla_world.get_spectator()

        # 在无限循环之前添加计数器
        frame_count = 0

        # run steps
        while True:
            try:
                scenario_manager.tick()
                scenario_manager.run_step()
                
                # 周期性检查内存
                frame_count += 1
                if frame_count % 100 == 0:
                    # 如果有psutil库，可以添加内存监控
                    pass
                    
            except KeyboardInterrupt:
                print('用户终止程序')
                break
            except Exception as e:
                print(f"运行时发生错误: {e}")
                break

    except Exception as e:
        print(f"主循环外发生错误: {e}")
    finally:
        print("清理资源中...")
        
        # 停止录制（如果启用）
        if opt.record:
            try:
                scenario_manager.client.stop_recorder()
            except:
                pass

        # 安全关闭场景管理器
        try:
            scenario_manager.close()
        except Exception as e:
            print(f"关闭场景管理器时出错: {e}")

        # 安全移除所有代理
        for v in agent_dict.values():
            try:
                v.remove()
            except Exception as e:
                print(f"移除代理时出错: {e}")
                
        agent_dict.clear()
        
        # 安全移除背景车辆
        for v in bg_veh_list:
            try:
                v.remove()
            except:
                pass
                
        # 强制垃圾回收
        import gc
        gc.collect()