import yaml
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict
from skylink_v2x.utils import read_yaml, CONFIG_PATH

@dataclass
class ClassConfig:
    class_idx: int
    name: str
    motion_type: str
    blueprints: List[str]

class MotionType(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"

class ObjectConfig:
    _instance = None

    def __new__(cls, yaml_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(yaml_path)
        return cls._instance


    def _initialize(self, yaml_path: str):
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
        except FileNotFoundError:
            raise ValueError(f"YAML file not found at: {yaml_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {yaml_path}: {str(e)}")

        self._class_idx_dict: Dict[int, ClassConfig] = {
            idx: ClassConfig(idx, entry["name"], entry["motion_type"], entry["blueprints"])
            for idx, entry in data["class_idx"].items()
        }
        self.vehicle_types = getattr(data, "vehicle_types", ['car', 'van', 'truck', 'bus'])
        
    def get_class_idx(self) -> List[int]:
        """Return a list of all class indices."""
        return list(self._class_idx_dict.keys())

    def get_by_name(self, name: str) -> ClassConfig:
        """Retrieve a class configuration by its name."""
        for config in self._class_idx_dict.values():
            if config.name == name:
                return config
        raise ValueError(f"Class with name '{name}' not found.")
    
    def get_by_names(self, names: List[str]) -> List[ClassConfig]:
        """Retrieve a list of class configurations by their names."""
        config_list = []
        for config in self._class_idx_dict.values():
            if config.name in names:
                config_list.append(config)
        return config_list         
    
    def get_all_vehicles(self) -> List[ClassConfig]:
        """Retrieve all class configurations for vehicles."""
        return self.get_by_names(self.vehicle_types)

    def get_by_type(self, actor_type: MotionType) -> List[ClassConfig]:
        """Retrieve all class configurations matching the given motion type."""
        return [config for config in self._class_idx_dict.values() if config.motion_type == actor_type.value]
    
    def get_dynamic_classes(self) -> List[ClassConfig]:
        """Retrieve all class configurations with dynamic motion type."""
        return self.get_by_type(MotionType.DYNAMIC)
    
    def get_static_classes(self) -> List[ClassConfig]:
        """Retrieve all class configurations with static motion type."""
        return self.get_by_type(MotionType.STATIC)
    
    def is_dynamic(self, class_idx: int) -> bool:
        """Check if the class configuration has dynamic motion type."""
        return self._class_idx_dict[class_idx].motion_type == MotionType.DYNAMIC.value
    
    def is_static(self, class_idx: int) -> bool:
        """Check if the class configuration has static motion type."""
        return self._class_idx_dict[class_idx].motion_type == MotionType.STATIC.value
    
    def is_traffic_light(self, class_idx: int) -> bool:
        """Check if the class configuration corresponds to a traffic light."""
        return self._class_idx_dict[class_idx].name == "traffic_light"    
    
    def get_by_blueprint(self, blueprint: str) -> ClassConfig:
        """Retrieve a class configuration by matching a blueprint."""
        if blueprint == 'spectator' or blueprint.startswith('sensor.'):
            return None
        
        for config in self._class_idx_dict.values():
            for bp in config.blueprints:
                if bp.endswith(".*"):
                    prefix = bp[:-2]
                    if blueprint == prefix or blueprint.startswith(prefix + "."):
                        return config
                elif bp.endswith("*"):
                    prefix = bp[:-1]
                    if blueprint.startswith(prefix):
                        return config
                elif bp == blueprint:
                    return config
                
        if blueprint == 'traffic.unknown':
            # TODO: Manully remove the warning flag for traffic.unknown. We should consider whether to add it to the class configuration.
            return None    
        print(f"Warning: Blueprint {blueprint} not found in class configurations.")
        return None
    
    def is_blueprint_dynamic(self, blueprint: str) -> bool:
        """Check if the blueprint corresponds to a dynamic class configuration."""
        config = self.get_by_blueprint(blueprint)
        return config and self.is_dynamic(config.class_idx)
    
    def is_blueprint_static(self, blueprint: str) -> bool:
        """Check if the blueprint corresponds to a static class configuration."""
        config = self.get_by_blueprint(blueprint)
        return config and self.is_static(config.class_idx)
    
    def is_blueprint_traffic_light(self, blueprint: str) -> bool:
        """Check if the blueprint corresponds to a traffic light class configuration."""
        config = self.get_by_blueprint(blueprint)
        return config and self.is_traffic_light(config.class_idx)
    
    def is_blueprint_object(self, blueprint: str) -> bool:
        """Check if the blueprint corresponds to an object class configuration."""
        return self.get_by_blueprint(blueprint) is not None
    



object_config_path = read_yaml(CONFIG_PATH)["object_config_path"]
OBJECT_CONFIG = ObjectConfig(object_config_path)

