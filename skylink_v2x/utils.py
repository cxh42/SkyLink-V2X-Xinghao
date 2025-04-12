from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from carla.libcarla import Vector3D
from scipy.spatial.transform import Rotation as R
import math
import carla
import airsim

CONFIG_PATH = 'skylink_v2x/skylink_configs/config.yaml'

def read_yaml(yaml_path: str) -> DictConfig:
    try:
        return OmegaConf.load(yaml_path)
    except FileNotFoundError:
        raise ValueError(f"YAML file not found at: {yaml_path}")
    except Exception as e:
        raise ValueError(f"Error loading YAML with OmegaConf from {yaml_path}: {str(e)}")
    

def velocity_to_speed_2d(velocity: Vector3D, meters=False) -> float:
    """
    Convert a 3D velocity vector to 2D speed.
    
    Args:
        velocity (carla.libcarla.Vector3D): The 3D velocity vector.
        
    Returns:
        float: The 2D speed.
    """
    speed = math.sqrt(velocity.x**2 + velocity.y**2)
    return speed if meters else 3.6 * speed

def dis_3d(p1: Vector3D, p2: Vector3D) -> float:
    """
    Calculate the 3D distance between two points.
    
    Args:
        p1 (carla.libcarla.Vector3D): First point.
        p2 (carla.libcarla.Vector3D): Second point.
        
    Returns:
        float: The 3D distance.
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def dis_2d(p1: Vector3D, p2: Vector3D) -> float:
    """
    Calculate the 2D distance between two points.
    
    Args:
        p1 (carla.libcarla.Vector3D): First point.
        p2 (carla.libcarla.Vector3D): Second point.
        
    Returns:
        float: The 2D distance.
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def velocity_to_speed_3d(velocity: Vector3D, meters=False) -> float:
    """
    Convert a 3D velocity vector to 3D speed.
    
    Args:
        velocity (carla.libcarla.Vector3D): The 3D velocity vector.
        
    Returns:
        float: The 3D speed.
    """
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed if meters else 3.6 * speed
    
def convert_carla_to_airsim(carla_location: carla.Location) -> airsim.Vector3r:
    """
    Convert a CARLA location (ENU) to an AirSim Vector3r (NED) with an added hover offset.
    
    Mapping:
      AirSim x = CARLA y
      AirSim y = -CARLA x   (to fix left/right inversion)
      AirSim z = - (CARLA z + hover_offset)
    
    Parameters:
        carla_location (carla.Location): Location in CARLA.
        hover_offset (float): Additional offset for z.
        
    Returns:
        airsim.Vector3r: The corresponding location in AirSim.
    """
    x = carla_location.y
    y = -carla_location.x
    z = -carla_location.z
    return airsim.Vector3r(x, y, z)

def convert_airsim_to_carla(airsim_vector: airsim.Vector3r) -> carla.Location:
    """
    Inverse conversion from AirSim (NED) to CARLA (ENU).
    
    Mapping:
      CARLA x = - AirSim y
      CARLA y = AirSim x
      CARLA z = - AirSim z
      
    Parameters:
        airsim_vector (airsim.Vector3r): The location in AirSim.
        
    Returns:
        carla.Location: The corresponding location in CARLA.
    """
    return carla.Location(x=-airsim_vector.y_val, y=airsim_vector.x_val, z=-airsim_vector.z_val)


def convert_carla_rotation_to_airsim_quaternion(carla_rotation) -> airsim.Quaternionr:
    """
    Convert CARLA's Euler rotation (in degrees, ENU) to AirSim's quaternion (NED).

    Mapping:
        - CARLA uses ENU (East-North-Up) coordinate system
        - AirSim uses NED (North-East-Down), so axes need to be adjusted
        - Yaw is inverted due to axis flip (Z up vs Z down)

    Parameters:
        carla_rotation (carla.Rotation): Rotation in degrees (roll, pitch, yaw)

    Returns:
        airsim.Quaternionr: Quaternion compatible with AirSim
    """
    # Convert degrees to radians
    roll_rad = math.radians(carla_rotation.roll)
    pitch_rad = math.radians(carla_rotation.pitch)
    yaw_rad = math.radians(carla_rotation.yaw)

    # Adjust for coordinate system differences (invert angles)
    # AirSim's to_quaternion takes (pitch, roll, yaw)
    return airsim.to_quaternion(-pitch_rad, -roll_rad, -yaw_rad)





def convert_airsim_quaternion_to_carla_rotation(quaternion: airsim.Quaternionr, force_bev=False) -> carla.Rotation:
    """
    Convert AirSim quaternion (NED) to CARLA's Euler rotation (ENU) in degrees.

    Mapping:
        - Convert quaternion to Euler angles (pitch, roll, yaw)
        - Negate them to match ENU system of CARLA
        - CARLA expects rotation in degrees

    Parameters:
        quaternion (airsim.Quaternionr): Orientation in AirSim (NED)

    Returns:
        carla.Rotation: Euler angles in degrees (roll, pitch, yaw)
    """
    # Extract quaternion values
    q = quaternion
    r = R.from_quat([q.x_val, q.y_val, q.z_val, q.w_val])  # scipy uses [x, y, z, w]

    # Convert to Euler angles (radians), AirSim uses pitch, roll, yaw order
    pitch_rad, roll_rad, yaw_rad = r.as_euler('xyz', degrees=False)

    # Invert angles to map NED ➜ ENU
    roll_deg = -math.degrees(roll_rad)
    pitch_deg = -math.degrees(pitch_rad)
    yaw_deg = -math.degrees(yaw_rad)
    if force_bev:
        pitch_deg = roll_deg = 0
    return carla.Rotation(roll=roll_deg, pitch=pitch_deg, yaw=yaw_deg)


def load_customized_world(xodr_path, client):
    """
    Load .xodr file and return the carla world object

    Parameters
    ----------
    xodr_path : str
        path to the xodr file

    client : carla.client
        The created CARLA simulation client.
    """
    if os.path.exists(xodr_path):
        with open(xodr_path) as od_file:
            try:
                data = od_file.read()
            except OSError:
                print('file could not be readed.')
                sys.exit()
        print('load opendrive map %r.' % os.path.basename(xodr_path))
        vertex_distance = 2.0  # in meters
        max_road_length = 500.0  # in meters
        wall_height = 1.0  # in meters
        extra_width = 0.6  # in meters
        world = client.generate_opendrive_world(
            data, carla.OpendriveGenerationParameters(
                vertex_distance=vertex_distance,
                max_road_length=max_road_length,
                wall_height=wall_height,
                additional_width=extra_width,
                smooth_junctions=False,
                enable_mesh_visibility=True))
        return world
    else:
        print('file not found.')
        return None
