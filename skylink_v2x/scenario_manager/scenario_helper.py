import carla
import random
import math
from typing import List, Tuple

try:
    from shapely.geometry import box, Point
except ImportError:
    raise ImportError("This script requires the 'shapely' package. Install via `pip install shapely`. ")

# -----------------------------
# CONFIGURABLE PARAMETERS
# -----------------------------
HOST = "localhost"
PORT = 2000
TIMEOUT = 10.0
TOWN = "Town05"          # change to your desired map

# Distance thresholds (in meters)
DS = 30.0                # max pair_wise distance between all N start points
DE = 30.0                # max pair_wise distance between all N end points
DT = 50.0                # min distance between any two selected traffic lights
MIN_SE_DIST = 10.0       # min distance between a start–end point of the same trajectory

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def _euclidean(loc1: carla.Location, loc2: carla.Location) -> float:
    """3_D Euclidean distance between two CARLA Locations."""
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2 + (loc1.z - loc2.z) ** 2)


def _cluster_points(points: List[carla.Transform], center: carla.Location, radius: float) -> List[carla.Transform]:
    """Return all points within *radius* of *center*."""
    return [p for p in points if _euclidean(p.location, center) <= radius]


def _choose_cluster(points: List[carla.Transform], n: int, radius: float, max_trials: int = 200) -> List[carla.Transform]:
    """Randomly pick a cluster center until >= n points fall inside *radius*."""
    for _ in range(max_trials):
        center = random.choice(points).location
        cluster = _cluster_points(points, center, radius)
        if len(cluster) >= n:
            return random.sample(cluster, n)
    raise RuntimeError(f"Unable to find {n} clustered points within {radius} m after {max_trials} trials.")


def _find_far_cluster(points: List[carla.Transform], n: int, near_center: carla.Location, min_center_dist: float,
                      radius: float, max_trials: int = 400) -> List[carla.Transform]:
    """Find *n* points forming a cluster whose centroid is at least *min_center_dist* from *near_center*."""
    for _ in range(max_trials):
        center = random.choice(points).location
        if _euclidean(center, near_center) < min_center_dist:
            continue
        cluster = _cluster_points(points, center, radius)
        if len(cluster) >= n:
            return random.sample(cluster, n)
    raise RuntimeError("Failed to find a distant endpoint cluster meeting the constraints.")


def _pair_points(starts: List[carla.Transform], ends: List[carla.Transform]) -> List[Tuple[carla.Transform, carla.Transform]]:
    """Greedy pairing: for each start, pick the farthest remaining end."""
    remaining_ends = ends.copy()
    pairs = []
    for s in starts:
        far_end = max(remaining_ends, key=lambda e: _euclidean(s.location, e.location))
        if _euclidean(s.location, far_end.location) < MIN_SE_DIST:
            raise RuntimeError("Start–end distance < MIN_SE_DIST; relax constraints or pick different clusters.")
        pairs.append((s, far_end))
        remaining_ends.remove(far_end)
    return pairs


def _rect_from_pair(pair: Tuple[carla.Transform, carla.Transform]):
    """Axis_aligned bounding rectangle (shapely Polygon) covering a start–end pair."""
    s, e = pair
    min_x, max_x = sorted([s.location.x, e.location.x])
    min_y, max_y = sorted([s.location.y, e.location.y])
    # Inflate by 5 m margin for safety
    margin = 5.0
    return box(min_x - margin, min_y - margin, max_x + margin, max_y + margin)


def _filter_lights_in_region(lights, region_poly):
    return [l for l in lights if region_poly.contains(Point(l.get_transform().location.x, l.get_transform().location.y))]


def _select_k_lights(lights: List[carla.TrafficLight], k: int, min_dist: float) -> List[carla.TrafficLight]:
    """Greedy max_spacing selection of K lights."""
    if k > len(lights):
        raise ValueError("Requested more traffic lights than available in region.")
    selected = [random.choice(lights)]
    while len(selected) < k:
        candidates = [l for l in lights if l not in selected and
                      all(_euclidean(l.get_transform().location, s.get_transform().location) >= min_dist for s in selected)]
        if not candidates:
            raise RuntimeError("Cannot satisfy min_dist constraint with given k; decrease k or min_dist.")
        selected.append(random.choice(candidates))
    return selected

# -----------------------------
# MAIN PIPELINE
# -----------------------------

def generate_routes_and_lights(n: int, k: int,
                               ds: float = DS, de: float = DE, dt: float = DT,
                               town: str = TOWN, host: str = HOST, port: int = PORT) -> Tuple[
                                   List[Tuple[carla.Transform, carla.Transform]],  # N start–end pairs
                                   List[box],                                      # N rectangles
                                   List[carla.TrafficLight]                        # K traffic lights
                               ]:
    """High_level helper that puts everything together."""

    # Connect & load world
    client = carla.Client(host, port)
    client.set_timeout(TIMEOUT)
    world = client.load_world(town)

    # Retrieve all candidate spawn points
    spawn_points: List[carla.Transform] = world.get_map().get_spawn_points()
    if len(spawn_points) < n * 2:
        raise RuntimeError("Not enough spawn points for requested N.")

    # 1) Pick clustered starts and clustered ends
    starts = _choose_cluster(spawn_points, n, ds)
    start_center = carla.Location(x=sum(p.location.x for p in starts) / n,
                                  y=sum(p.location.y for p in starts) / n,
                                  z=0.0)

    ends = _find_far_cluster(spawn_points, n, start_center, min_center_dist=200.0, radius=de)  # 200 m is heuristic

    # 2) Pair them to maximise per_vehicle distance
    pairs = _pair_points(starts, ends)

    # 3) Build rectangles and their union
    rectangles = [_rect_from_pair(p) for p in pairs]
    union_region = rectangles[0]
    for r in rectangles[1:]:
        union_region = union_region.union(r)

    # 4) Traffic lights inside region
    all_lights = world.get_actors().filter("traffic.traffic_light*")
    region_lights = _filter_lights_in_region(all_lights, union_region)

    # 5) Select K well_spaced lights
    selected_lights = _select_k_lights(region_lights, k, dt)

    return pairs, rectangles, selected_lights


if __name__ == "__main__":
    N = 5   # number of vehicles / trajectories
    K = 4   # number of traffic lights desired

    pairs, rects, lights = generate_routes_and_lights(N, K)

    print("Start–End Pairs:")
    for i, (s, e) in enumerate(pairs):
        print(f"  Vehicle {i}: start=({s.location.x:.1f}, {s.location.y:.1f}), end=({e.location.x:.1f}, {e.location.y:.1f})")

    print("\nBounding Rectangles (minx, miny, maxx, maxy):")
    for i, r in enumerate(rects):
        minx, miny, maxx, maxy = r.bounds
        print(f"  Trajectory {i}: {minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f}")

    print("\nSelected Traffic Lights:")
    for i, tl in enumerate(lights):
        loc = tl.get_transform().location
        print(f"  TL {i}: id={tl.id}, loc=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
