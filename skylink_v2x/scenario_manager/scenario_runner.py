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
        # run steps
        while True:
            scenario_manager.tick()
            # Since a scene always have a vehicle, we can use the first one as the ego vehicle
            if 'cav1' not in agent_dict:
                raise ValueError("The first vehicle must be cav1 for correctly spawn the spectator.")
            transform = agent_dict['cav1'].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=70),
                carla.Rotation(
                    pitch=-
                    90)))
            
            # The following commented code is for ego vehicle visualization. Consider removing it or move it somewhere else.
            # spectator.set_transform(carla.Transform(
            #     transform.location +
            #     carla.Location(
            #         x=-3,
            #         y=-5,
            #         z=4),
            #     carla.Rotation(
            #         yaw=65,
            #         pitch=-
            #         30)))
            
            # spectator.set_transform(carla.Transform(
            #     transform.location +
            #     carla.Location(
            #         x=0,
            #         y=-5,
            #         z=1.3),
            #     carla.Rotation(
            #         yaw=90,
            #         pitch=-
            #         0)))

            # for i, single_cav in enumerate(agent_dict):
            #     single_cav.update_info()
            #     control = single_cav.run_step()
            #     single_cav.vehicle.apply_control(control)
            scenario_manager.run_step()

    finally:
        # eval_manager.evaluate() # TODO: Implement evaluation later

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        for v in agent_dict.values():
            v.remove()
        agent_dict.clear()
        for v in bg_veh_list:
            v.remove()
