
from abc import ABC, abstractmethod
import carla
import weakref
import numpy as np
from skylink_v2x.utils import *

class PerceptionSensor(ABC):
    def __init__(self):
        """Initialize sensor with no data."""
        self._sensor = None
        self._data = None
        self._timestamp = None
        self._frame = None

    def spawn(self, world, attach_to, sensor_config):
        """
        Spawn a sensor in CARLA.
        
        Args:
            world (carla.World): The CARLA world instance.
            attach_to (carla.Actor): The actor to attach this sensor to.
            sensor_config: Configuration object/struct for sensor attributes.
        """
        blueprint = self.get_blueprint(world, sensor_config)
        carla_location = carla.Location(
            x=sensor_config.pos[0], 
            y=sensor_config.pos[1],
            z=sensor_config.pos[2]
        )
        carla_rotation = carla.Rotation(
            pitch=sensor_config.pos[3],
            yaw=sensor_config.pos[4],
            roll=sensor_config.pos[5]
        )
        self._transform = carla.Transform(carla_location, carla_rotation)
        self._sensor = world.spawn_actor(blueprint, self._transform, attach_to=attach_to)
        self.listen()
    
    @abstractmethod
    def get_info(self):
        """Retrieve sensor data (abstract)."""
        pass
        
    @abstractmethod
    def listen(self):
        """Listen to incoming sensor data (abstract)."""
        pass
        
    @abstractmethod
    def get_blueprint(self, world, sensor_config):
        """
        Get the blueprint for the sensor and set its attributes (abstract).
        
        Returns:
            carla.Blueprint: The blueprint object properly configured.
        """
        pass

    @abstractmethod
    def on_data_event(event):
        """Handle incoming sensor data (abstract)."""
        pass
    
    def destroy(self):
        """Destroy the sensor to free resources."""
        if self._sensor is not None:
            self._sensor.destroy()
            self._sensor = None
        self._data = None
        self._timestamp = None
        self._frame = None


class CameraSensor(PerceptionSensor):
    def get_blueprint(self, world, sensor_config):
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', str(sensor_config.fov))
        # TODO: make sure is this the correct way to set image size
        # or should we manually resize the image?
        blueprint.set_attribute('image_size_x', str(sensor_config.resolution[0]))
        blueprint.set_attribute('image_size_y', str(sensor_config.resolution[1]))
        return blueprint

    def listen(self):
        """Register a callback to listen for camera images."""
        weak_self = weakref.ref(self)
        self._sensor.listen(
            lambda event: (weak_self() and weak_self().on_data_event(event))
        )

    def on_data_event(self, event):
        image_data = np.array(event.raw_data)
        # TODO: If needed, reshape to (height, width, 4):
        image_data = image_data.reshape((event.height, event.width, 4))
        
        # Take only the first 3 channels (RGB)
        
        self._data = image_data[:, :, :3]
        self._timestamp = event.timestamp
        self._frame = event.frame

    def get_info(self):
        """Return the latest camera data and metadata."""
        
        return {
            'image': self._data,
            'timestamp': self._timestamp,
            'frame': self._frame,
            'sensor_type': 'CameraSensor',
            'trans': self._transform,
            'image_size_x': self._sensor.attributes['image_size_x'],
            'image_size_y': self._sensor.attributes['image_size_y'],
            'fov': self._sensor.attributes['fov'],
        }


class LidarSensor(PerceptionSensor):
    def get_blueprint(self, world, sensor_config):
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('upper_fov', str(sensor_config['upper_fov']))
        blueprint.set_attribute('lower_fov', str(sensor_config['lower_fov']))
        blueprint.set_attribute('channels', str(sensor_config['channels']))
        blueprint.set_attribute('range', str(sensor_config['range']))
        blueprint.set_attribute('points_per_second', str(sensor_config['points_per_second']))
        blueprint.set_attribute('rotation_frequency', str(sensor_config['rotation_frequency']))
        blueprint.set_attribute('dropoff_general_rate', str(sensor_config['dropoff_general_rate']))
        blueprint.set_attribute('dropoff_intensity_limit', str(sensor_config['dropoff_intensity_limit']))
        blueprint.set_attribute('dropoff_zero_intensity', str(sensor_config['dropoff_zero_intensity']))
        blueprint.set_attribute('noise_stddev', str(sensor_config['noise_stddev']))
        return blueprint

    def listen(self):
        """Register a callback to listen for Lidar data."""
        weak_self = weakref.ref(self)
        self._sensor.listen(
            lambda event: (weak_self() and weak_self().on_data_event(event))
        )

    def on_data_event(self, event):
        data = np.frombuffer(event.raw_data, dtype=np.float32)
        data = data.reshape((int(data.shape[0] / 4), 4))
        self._data = data
        self._timestamp = event.timestamp
        self._frame = event.frame

    def get_info(self):
        """Return the latest lidar data and metadata."""
        return {
            'points': self._data,
            'timestamp': self._timestamp,
            'frame': self._frame,
            'trans': self._transform,
            'sensor_type': 'LidarSensor',
        }


class SemanticLidarSensor(PerceptionSensor):
    def __init__(self):
        super().__init__()
        self._points = None
        self._obj_tag = None
        self._obj_idx = None
    
    def get_blueprint(self, world, sensor_config):
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        blueprint.set_attribute('upper_fov', str(sensor_config['upper_fov']))
        blueprint.set_attribute('lower_fov', str(sensor_config['lower_fov']))
        blueprint.set_attribute('channels', str(sensor_config['channels']))
        blueprint.set_attribute('range', str(sensor_config['range']))
        blueprint.set_attribute('points_per_second', str(sensor_config['points_per_second']))
        blueprint.set_attribute('rotation_frequency', str(sensor_config['rotation_frequency']))
        return blueprint
    
    def listen(self):
        """Register a callback to listen for semantic Lidar data."""
        weak_self = weakref.ref(self)
        self._sensor.listen(
            lambda event: (weak_self() and weak_self().on_data_event(event))
        )

    def on_data_event(self, event):
        data = np.frombuffer(
            event.raw_data, 
            dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('CosAngle', np.float32), ('ObjIdx', np.uint32),
                ('ObjTag', np.uint32)
            ])
        )

        self._points = np.vstack([data['x'], data['y'], data['z']]).T
        self._obj_tag = data['ObjTag'].copy()
        self._obj_idx = data['ObjIdx'].copy()
        self._data = data
        self._timestamp = event.timestamp
        self._frame = event.frame

    def get_info(self):
        """Return the latest semantic lidar data and metadata."""
        return {
            'points': self._points,
            'obj_tag': self._obj_tag,
            'obj_idx': self._obj_idx,
            'raw_data': self._data,
            'timestamp': self._timestamp,
            'frame': self._frame,
            'trans': self._transform,
            'sensor_type': 'SemanticLidarSensor',
        }

    def destroy(self):
        """Destroy the semantic lidar sensor and clear local references."""
        super().destroy()
        self._points = None
        self._obj_tag = None
        self._obj_idx = None

