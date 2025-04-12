# SkyLink-V2X：A Unified Air-Ground V2X Collaboration Framework



## Installation
Please install airsim refering to [Airsim Installation](doc/Airsim_installation.md)

Please install carla refering to [Carla Installation](doc/Carla_installation.md)

Please install python dependencies by
```bash
conda create -n skylink_v2x python=3.7
conda activate skylink_v2x
python -n pip install -r requirements.txt
python setup.py develop # 
```

## Run Scenarios

Run 
```bash
python skylink_v2x.py -t [scenario_name]
```

For example
```bash
python skylink_v2x.py -t town05_demo
```

You may find scenario names under `skylink_v2x/skylink_configs`



## Customize your Scenarios

Please refer to `skylink_v2x/skylink_configs/town05_demo.yaml` and `skylink_v2x/skylink_configs/default.yaml` for details. More instruction will be release later.

### Some useful tools for scenario design


## Customize your agent autonomy

Agent autonomy includes **Perception**, **Localization**, **Mapping**, **Planning**, and **Control**. Each portion support flexibly inferface for both rule-based and deep-learning based algorithms

For instruction will be release later.

## Customize your communication

Skylink support multiple types simulation regarding communincation robustness, including **package loss**, **communication latency**, **adversarial attack**, **agent disconnection**, **sensor drop**, etc.

More instruction will be release later.
