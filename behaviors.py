import numpy as np

def rotate_xy_vector(xy: np.ndarray, theta: float):
    cT = np.cos(theta)
    sT = np.sin(theta)
    return np.matmul([[cT, -sT], [sT, cT]], xy)

def move_to_goal(target_xy: np.ndarray, current_xyT: np.ndarray):
    x, y, theta = current_xyT
    goal_vector = target_xy - np.array([x, y])
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])
    vec = R @ goal_vector
    return vec

def follow_path(path: np.ndarray, current_xyT: np.ndarray):
    """
    Follow a given path by projecting robot position to nearest segment.
    path: N x 2 array of (x, y) global path points
    current_xyT: current robot pose (x, y, theta) in global frame
    Returns a robot-local direction vector
    """
    if path is None or len(path) < 2:
        return np.zeros(2, dtype=float)

    robot_pos = current_xyT[:2]
    dists = np.linalg.norm(path - robot_pos, axis=1)
    closest_idx = np.argmin(dists)
    follow_idx = min(closest_idx + 1, len(path) - 1)
    target_point = path[follow_idx]
    direction_global = target_point - robot_pos
    return rotate_xy_vector(direction_global, -current_xyT[2])

def avoid_obstacles(obstacle_xy: np.ndarray):
    SPHERE_OF_INFLUENCE = 1.0
    SAFETY_MARGIN = 0.5
    vec = np.zeros(2, dtype=float)
    for obs in obstacle_xy:
        distance = np.linalg.norm(obs)
        if distance < SPHERE_OF_INFLUENCE:
            repulsion = (1 / (distance + 1e-6) - 1 / SPHERE_OF_INFLUENCE) * (obs / distance)
            vec -= repulsion
    return vec

def swirl_obstacles(obstacle_xy: np.ndarray):
    SPHERE_OF_INFLUENCE = 1.0
    vec = np.zeros(2, dtype=float)
    for obs in obstacle_xy:
        distance = np.linalg.norm(obs)
        if distance < SPHERE_OF_INFLUENCE:
            perpendicular_vec = np.array([-obs[1], obs[0]]) / (distance + 1e-6)
            strength = (1 / (distance + 1e-6) - 1 / SPHERE_OF_INFLUENCE)
            vec += strength * perpendicular_vec
    return vec

def random_motion():
    dist_angle = np.random.rand(1, 2)[0]
    dist_angle[1] *= 2 * np.pi
    return np.array([
        dist_angle[0] * np.cos(dist_angle[1]),
        dist_angle[0] * np.sin(dist_angle[1])
    ])

def behavioral_coordination(path: np.ndarray,
                            current_xyT: np.ndarray,
                            obstacle_xy: list):
    """
    Combines path following, obstacle avoidance, and swirl behavior
    path: N x 2 array of global path waypoints
    current_xyT: robot pose [x, y, theta]
    obstacle_xy: list of obstacle points in robot-local frame
    Returns: 2D robot-local motion vector
    """
    path_follow = follow_path(path, current_xyT)
    avoid = avoid_obstacles(obstacle_xy)
    swirl = swirl_obstacles(obstacle_xy)
    rand = random_motion()

    # Weights can be tuned based on environment behavior
    w_path, w_avoid, w_swirl, w_rand = 1.2, 1.5, 0.8, 0.2
    vec = w_path * path_follow + w_avoid * avoid + w_swirl * swirl + w_rand * rand
    return vec