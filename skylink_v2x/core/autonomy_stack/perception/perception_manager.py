from abc import ABC, abstractmethod
from typing import List, Dict, Any
import carla
import json
import numpy as np
import cv2
import time
import math

from skylink_v2x.core.autonomy_stack.perception.sensors import (
    CameraSensor, LidarSensor, SemanticLidarSensor, PerceptionSensor
)
from skylink_v2x.carla_helper import OBJECT_CONFIG

class PerceptionManager(ABC):
    def __init__(
        self, 
        carla_world: carla.World, 
        agent_id: int, 
        config: Any,
        comm_manager
    ):
        """
        Initialize the perception manager.
        
        Args:
            carla_world (carla.World): Simulation environment (CARLA world).
            carla_map (carla.Map): Map of the environment.
            agent_id (int): Identifier of the agent.
            config (Any): Configuration settings (e.g., sensor config, detection model, etc.).
        """
        self._carla_world = carla_world
        self._agent_id = agent_id
        self._config = config
        self._use_gt = (self._config.detection == 'gt')
        self._comm_manager = comm_manager
        
        # TODO: Load detection model based on model name
        self._model = None if self._use_gt else self._config.detection
        assert self._model is None, "Current implementation only supports ground truth detection."
        
        # A dictionary to store sensor_name -> sensor_instance
        self._sensors: Dict[str, PerceptionSensor] = {}
        # Get the actor (vehicle, RSU, or drone) by agent_id
        self._actor = self._carla_world.get_actor(self._agent_id)
        
        # Verify that the actor is valid
        self.actor_check()
        self.spawn_sensors()

    @abstractmethod
    def actor_check(self):
        """
        Abstract method to ensure the correct actor (vehicle/RSU/drone) is found.
        Must be implemented by child classes.
        """
        pass

    def spawn_sensors(self):
        """
        Spawn all sensors (cameras, lidars, semantic lidars) and attach them to the actor.
        Iterates over the sensor configurations defined in self._config.
        """
        # Spawn camera sensors
        for cam_config in self._config.sensors.camera:
            # CameraSensor should be a class that can accept (world, actor, config)
            camera_sensor = CameraSensor()
            camera_sensor.spawn(self._carla_world, self._actor, cam_config)
            # Assume each cam_config has a 'name' attribute to be used as dictionary key
            self._sensors[cam_config.name] = camera_sensor

        # Spawn lidar sensors
        if 'lidar' in self._config.sensors:
            for lidar_config in self._config.sensors.lidar:
                lidar_sensor = LidarSensor()
                lidar_sensor.spawn(self._carla_world, self._actor, lidar_config)
                self._sensors[lidar_config.name] = lidar_sensor

        # Spawn semantic lidar sensors
        if 'semantic_lidar' in self._config.sensors:
            for sem_lidar_config in self._config.sensors.semantic_lidar:
                semantic_lidar_sensor = SemanticLidarSensor()
                semantic_lidar_sensor.spawn(self._carla_world, self._actor, sem_lidar_config)
                self._sensors[sem_lidar_config.name] = semantic_lidar_sensor

    def detect_objects(self, v2x_perception_data: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect objects in the environment.
        
        If self._use_gt is True, use CARLA built-in ground truth (all actors).
        Otherwise, fallback to a detection model (placeholder).
        
        Returns:
            list: A list of detected objects containing their ID, class, position, velocity, etc.
        """
        # TODO: Only detect objects within sensor range
        detected_objects = []
        if self._use_gt:
            all_actors = self._carla_world.get_actors()
            
            for actor in all_actors:
                # Exclude the ego vehicle itself
                if actor.id != self._agent_id:
                    # TODO: Need a class mapping for actor.type_id
                    obj_class = actor.type_id
                    if not OBJECT_CONFIG.is_blueprint_object(obj_class):
                        continue
                    obj_position = actor.get_location()
                    obj_velocity = actor.get_velocity()
                    detected_objects.append({
                        'actor': actor, #TODO: Here we include the actor object itself, this is for accommodating the planning module, which should & will be refactored. 
                        'is_dynamic': OBJECT_CONFIG.is_blueprint_dynamic(obj_class),
                        'is_traffic_light': OBJECT_CONFIG.is_blueprint_traffic_light(obj_class),
                        'id': actor.id,
                        'class': obj_class,
                        'position': (obj_position.x, obj_position.y, obj_position.z),
                        'velocity': (obj_velocity.x, obj_velocity.y, obj_velocity.z)
                    })
        else:
            # Placeholder for sensor/model-based detection
            v2x_perception_data = v2x_perception_data
            print(f"Using detection model: {self._model.name if self._model else 'Unknown'}")
        
        data = {
            'raw_data': self.get_sensor_data(),
            'detected_objects': detected_objects
        }
        
        self._comm_manager.buffer_perception(self._agent_id, data)
        
    
    def get_sensors(self) -> Dict[str, PerceptionSensor]:
        """
        Get all spawned sensors.
        
        Returns:
            dict: A dictionary mapping sensor_name -> sensor_instance
        """
        return self._sensors
        
        
    def get_agent_id(self) -> int:
        """
        Get the agent ID.
        
        Returns:
            int: The agent ID.
        """
        return self._agent_id

    def get_sensor_data(self):
        """
        Retrieve the latest data from all spawned sensors.
        
        Returns:
            dict: A dictionary mapping sensor_name -> latest_sensor_data
        """
        sensor_data = {}
        for sensor_name, sensor_instance in self._sensors.items():
            sensor_data[sensor_name] = sensor_instance.get_info()
        return sensor_data



    def capture_spectator_view(self, image_list):
        spectator = self._carla_world.get_spectator()
        transform = spectator.get_transform()

        bp_lib = self._carla_world.get_blueprint_library()
        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '3840')
        cam_bp.set_attribute('image_size_y', '2160')
        cam_bp.set_attribute('fov', '90')
        cam_bp.set_attribute('sensor_tick', '0.033')

        camera = self._carla_world.spawn_actor(cam_bp, transform)

        image_ready = {'flag': False}
        def callback(image):
            if not image_ready['flag']:
                image_list.append(image)
                image_ready['flag'] = True

        camera.listen(callback)
        settings = self._carla_world.get_settings()
        print("Synchronous mode:", settings.synchronous_mode)

        self._carla_world.tick()
        self._carla_world.tick()  # 第二次 tick 才能真正获取图像

        timeout = time.time() + 1
        while not image_ready['flag'] and time.time() < timeout:
            time.sleep(0.01)

        if not image_ready['flag']:
            raise TimeoutError("Camera image capture timed out.")

        camera.stop()
        camera.destroy()


    def visualize_sensors(self):
        # 捕获spectator视角的图像
        image_list = []
        self.capture_spectator_view(image_list)
        if not image_list:
            raise ValueError("Failed to capture spectator view image")
        
        spectator_image = image_list[0]
        
        # 转换为OpenCV格式的图像
        array = np.frombuffer(spectator_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (spectator_image.height, spectator_image.width, 4))
        img = array[:, :, :3].copy()  # 只保留RGB通道并创建副本
        
        # 获取相机参数
        width = spectator_image.width
        height = spectator_image.height
        fov = 90  # 使用capture_spectator_view中设置的fov值
        
        # 获取spectator变换
        spectator = self._carla_world.get_spectator()
        spectator_transform = spectator.get_transform()
        
        # 定义世界坐标到相机坐标的转换
        def world_to_camera(world_point, camera_transform):
            """将世界坐标转换为相机坐标系"""
            # 计算相对位置向量
            location = camera_transform.location
            dx = world_point.x - location.x
            dy = world_point.y - location.y
            dz = world_point.z - location.z
            
            # 获取相机的坐标系基向量
            forward = camera_transform.get_forward_vector()
            right = camera_transform.get_right_vector()
            up = camera_transform.get_up_vector()
            
            # 将相对位置向量投影到相机坐标系
            x = right.x * dx + right.y * dy + right.z * dz
            y = up.x * dx + up.y * dy + up.z * dz
            z = forward.x * dx + forward.y * dy + forward.z * dz
            
            return np.array([x, y, z])
        
        # 定义相机坐标到图像坐标的转换
        def camera_to_image(camera_point, fov=90.0, width=width, height=height):
            """将相机坐标系中的点投影到图像平面"""
            # 如果点在相机后方，返回None
            if camera_point[2] <= 0:
                return None
            
            # 计算焦距
            f = width / (2.0 * np.tan(np.radians(fov) / 2.0))
            
            # 投影到图像平面
            x = f * camera_point[0] / camera_point[2] + width / 2.0
            y = f * -camera_point[1] / camera_point[2] + height / 2.0  # 注意Y轴反向
            
            # 检查是否在图像范围内
            if 0 <= x < width and 0 <= y < height:
                return (int(x), int(y))
            else:
                return None
        
        # 定义世界坐标到图像坐标的转换
        def world_to_image(world_point, camera_transform, fov=90.0):
            """将世界坐标点投影到图像平面"""
            camera_point = world_to_camera(world_point, camera_transform)
            return camera_to_image(camera_point, fov)
        
        # 绘制车辆位置
        vehicle_location = self._actor.get_location()
        vehicle_pixel = world_to_image(vehicle_location, spectator_transform)
        # if vehicle_pixel:
        #     cv2.circle(img, vehicle_pixel, 10, (0, 0, 255), -1)
        
        # 绘制车辆方向
        veh_transform = self._actor.get_transform()
        forward_vector = veh_transform.get_forward_vector()
        forward_location = carla.Location(
            vehicle_location.x + forward_vector.x * 2,
            vehicle_location.y + forward_vector.y * 2,
            vehicle_location.z + forward_vector.z * 2
        )
        forward_pixel = world_to_image(forward_location, spectator_transform)
        # if vehicle_pixel and forward_pixel:
        #     cv2.arrowedLine(img, vehicle_pixel, forward_pixel, (255, 0, 0), 2)
        
        # 绘制传感器位置
        for name, sensor in self._sensors.items():
            info = sensor.get_info()
            sensor_transform = info['trans']
            sensor_location = sensor_transform.location
            sensor_location = carla.Location(
                x=sensor_location.x + vehicle_location.x,
                y=sensor_location.y + vehicle_location.y,
                z=sensor_location.z + vehicle_location.z
            )
            # 将传感器位置投影到图像平面
            sensor_pixel = world_to_image(sensor_location, spectator_transform)
            if sensor_pixel:
                # 绘制传感器标记和标签
                # cv2.circle(img, sensor_pixel, 5, (0, 255, 0), -1)
                # cv2.putText(img, name, (sensor_pixel[0] + 5, sensor_pixel[1] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 如果是相机传感器，绘制FOV半透明扇形
                if info.get('sensor_type') == 'CameraSensor':
                    try:
                        sensor_fov = float(info['fov'])
                    except Exception:
                        sensor_fov = 90.0  # 默认FOV
                    
                    # 扇形参数
                    fov_distance = 5.0  # 扇形半径
                    angle_samples = 30  # 角度采样数
                    depth_samples = 5  # 深度采样数（用于创建逐渐透明的效果）
                    
                    # 获取相机的方向向量
                    forward = sensor_transform.get_forward_vector()
                    right = sensor_transform.get_right_vector()
                    
                    # 为了创建深度渐变效果，我们从近到远绘制多个扇形层
                    for depth_step in range(depth_samples):
                        # 当前深度层的内外半径
                        inner_radius = fov_distance * depth_step / depth_samples
                        outer_radius = fov_distance * (depth_step + 1) / depth_samples
                        
                        # 创建这层的多边形点
                        fan_points = []
                        
                        # 绘制扇形的内弧（从左到右）
                        for i in range(angle_samples + 1):
                            angle = np.radians(sensor_fov) * (i / angle_samples - 0.5)  # -fov/2 到 fov/2
                            
                            # 计算旋转后的方向向量
                            rotated_dir = carla.Vector3D(
                                forward.x * np.cos(angle) + right.x * np.sin(angle),
                                forward.y * np.cos(angle) + right.y * np.sin(angle),
                                forward.z * np.cos(angle) + right.z * np.sin(angle)
                            )
                            
                            # 归一化
                            length = np.sqrt(rotated_dir.x**2 + rotated_dir.y**2 + rotated_dir.z**2)
                            rotated_dir.x /= length
                            rotated_dir.y /= length
                            rotated_dir.z /= length
                            
                            # 内弧上的点
                            inner_point = carla.Location(
                                sensor_location.x + rotated_dir.x * inner_radius,
                                sensor_location.y + rotated_dir.y * inner_radius,
                                sensor_location.z + rotated_dir.z * inner_radius
                            )
                            
                            # 将点投影到图像
                            inner_pixel = world_to_image(inner_point, spectator_transform)
                            if inner_pixel:
                                fan_points.append(inner_pixel)
                        
                        # 绘制扇形的外弧（从右到左）
                        for i in range(angle_samples, -1, -1):
                            angle = np.radians(sensor_fov) * (i / angle_samples - 0.5)  # -fov/2 到 fov/2
                            
                            # 计算旋转后的方向向量
                            rotated_dir = carla.Vector3D(
                                forward.x * np.cos(angle) + right.x * np.sin(angle),
                                forward.y * np.cos(angle) + right.y * np.sin(angle),
                                forward.z * np.cos(angle) + right.z * np.sin(angle)
                            )
                            
                            # 归一化
                            length = np.sqrt(rotated_dir.x**2 + rotated_dir.y**2 + rotated_dir.z**2)
                            rotated_dir.x /= length
                            rotated_dir.y /= length
                            rotated_dir.z /= length
                            
                            # 外弧上的点
                            outer_point = carla.Location(
                                sensor_location.x + rotated_dir.x * outer_radius,
                                sensor_location.y + rotated_dir.y * outer_radius,
                                sensor_location.z + rotated_dir.z * outer_radius
                            )
                            
                            # 将点投影到图像
                            outer_pixel = world_to_image(outer_point, spectator_transform)
                            if outer_pixel:
                                fan_points.append(outer_pixel)
                        
                        # 如果有足够的点形成多边形，绘制半透明填充
                        if len(fan_points) > 3:  # 至少需要3个点才能形成多边形
                            # 创建覆盖层
                            overlay = img.copy()
                            # 创建多边形
                            fan_polygon = np.array([fan_points], dtype=np.int32)
                            
                            # 根据深度设置渐变透明度（离相机越远越透明）
                            base_alpha = 0.5  # 基础透明度
                            alpha = base_alpha * (1 - depth_step * 0.5 / depth_samples)
                            
                            # 填充多边形
                            cv2.fillPoly(overlay, fan_polygon, (70, 180, 70))
                            
                            # 混合图像
                            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # 保存图像
        cv2.imwrite("debug/sensors_on_spectator.png", img)
        import pdb; pdb.set_trace()



    def destroy(self) -> None:
        """
        Destroy all spawned sensors and clean up resources.
        """
        # Safely destroy each sensor actor
        for sensor_instance in self._sensors.values():
            sensor_instance.destroy()
        self._sensors.clear()  # or self._sensors = {}
            
            

        
class VehiclePerceptionManager(PerceptionManager):
    def actor_check(self):
        """
        Ensures the vehicle actor exists in the environment.
        Raises ValueError if not found.
        """
        if not self._actor:
            raise ValueError(f"Vehicle agent with ID {self._agent_id} not found in CARLA world.")


class RSUPerceptionManager(PerceptionManager):
    def actor_check(self):
        """
        Ensures the RSU actor exists in the environment.
        Raises ValueError if not found.
        """
        if not self._actor:
            raise ValueError(f"RSU agent with ID {self._agent_id} not found in CARLA world.")


class DronePerceptionManager(PerceptionManager):
    def actor_check(self):
        """
        Ensures the drone actor exists in the environment.
        Raises ValueError if not found.
        """
        if not self._actor:
            raise ValueError(f"Drone agent with ID {self._agent_id} not found in AirSim client.")





def get_matrix(transform):
    """
    根据 CARLA Transform 构造 4x4 齐次变换矩阵。
    """
    # 将角度转为弧度
    roll = math.radians(transform.rotation.roll)
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    
    # 分别构造绕各轴旋转矩阵
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    R_pitch = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    R_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    t = np.array([transform.location.x, transform.location.y, transform.location.z]).reshape((3,1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t[:,0]
    return T

def world_to_image(cam_transform, world_location, K):
    """
    将世界坐标转换到摄像机图像平面（像素坐标）。
    cam_transform: spectator 的 Transform
    world_location: 包含 x, y, z 属性（传感器位置）
    K: 内参矩阵
    """
    # 得到摄像机外参矩阵：从世界坐标到摄像机坐标系
    T = get_matrix(cam_transform)
    T_inv = np.linalg.inv(T)
    # 世界坐标齐次化
    point_w = np.array([world_location.x, world_location.y, world_location.z, 1.0])
    # 转换到摄像机坐标
    point_cam = T_inv.dot(point_w)
    # 如果在摄像机后面，则返回 None
    if point_cam[2] <= 0.001:
        return None
    # 归一化
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]
    # 投影到像素平面
    u = K[0,0] * x + K[0,2]
    v = K[1,1] * y + K[1,2]
    return int(u), int(v)

def carla_image_to_array(image):
    """
    将 CARLA sensor.image 转换为 numpy 数组（RGB）。
    注意：CARLA 图像数据通常包含 4 通道（BGRA），这里只取前三个通道。
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    # 转换 BGR 到 RGB 如有需要，也可直接使用 BGR 显示
    array = array[:, :, :3]
    return array