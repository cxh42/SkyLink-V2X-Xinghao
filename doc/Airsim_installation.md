Got it! Here’s the improved version in proper Markdown format:

# Guidance for AirSim Installation for Skylink

## AirSim Binaries

- Download the **Blocks.zip** binary from the [AirSim Releases](https://github.com/Microsoft/AirSim/releases) page instead of building from source.

## Running the Binary

- Ensure the binary is executable with the following command:

```bash
./Blocks.sh -windowed -ResX=1280 -ResY=720
```
After running AirSim, the path to the settings.json file will be shown in the execution window.
It’s typically located at: `~/Documents/AirSim/settings.json`



Then, please peplace the contents of the file with the following configuration to support 11 UAVs:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "RpcPort": 41451,
  "RpcEnabled": true,
  "LocalHostIp": "127.0.0.1",
  "Vehicles": {
    "UAV1":  { "VehicleType": "SimpleFlight", "X": -20,   "Y": 0,   "Z": -10 },
    "UAV2":  { "VehicleType": "SimpleFlight", "X": 100,   "Y": 20,  "Z": -10 },
    "UAV3":  { "VehicleType": "SimpleFlight", "X": 50,    "Y": 50,  "Z": -10 },
    "UAV4":  { "VehicleType": "SimpleFlight", "X": -120,  "Y": 20,  "Z": -10 },
    "UAV5":  { "VehicleType": "SimpleFlight", "X": 200,   "Y": 50,  "Z": -10 },
    "UAV6":  { "VehicleType": "SimpleFlight", "X": 150,   "Y": 20,  "Z": -10 },
    "UAV7":  { "VehicleType": "SimpleFlight", "X": -220,  "Y": 50,  "Z": -10 },
    "UAV8":  { "VehicleType": "SimpleFlight", "X": 300,   "Y": 20,  "Z": -10 },
    "UAV9":  { "VehicleType": "SimpleFlight", "X": 250,   "Y": 50,  "Z": -10 },
    "UAV10": { "VehicleType": "SimpleFlight", "X": -320,  "Y": 20,  "Z": -10 },
    "UAV11": { "VehicleType": "SimpleFlight", "X": 400,   "Y": 50,  "Z": -10 }
  }
}
```


- This configuration enables a maximum of 11 UAVs to run simultaneously.
You can add more if needed, but ensure the UAVs are spaced far enough apart and at -10m high to avoid collisions.

Recommended Run Mode
- To reduce resource usage and minimize the chance of accidental collisions, run in non-rendered mode:

```bash
./Blocks.sh -windowed -ResX=1280 -ResY=720 -nullrhi -nosound -nopause
```