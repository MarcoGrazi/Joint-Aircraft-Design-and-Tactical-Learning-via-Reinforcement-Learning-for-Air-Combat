### SCRIPT FOR TESTING CORRECT FUNCTIONING OF KEY ENVIRONMENT METHODS
import numpy as np


### TEST orientation_to_target ###
def orientation_to_target(position, target_position):
    direction = target_position - position
    norm_dir = direction / np.linalg.norm(direction)  # Normalize

    dx, dy, dz = norm_dir

    # Yaw (rotation around Z axis, horizontal heading)
    yaw = np.arctan2(dy, dx)

    # Pitch (rotation around Y axis, vertical), invert dz because z+ is down
    pitch = np.arcsin(-dz)

    # Roll is not needed here
    roll = 0.0

    return np.array([roll, pitch, yaw])

# === Test cases ===
test_cases = [
    {
        "position": np.array([0.0, 0.0, 0.0]),
        "target_position": np.array([1.0, 0.0, 0.0]),
        "expected": "Facing +X (yaw=0, pitch=0)"
    },
    {
        "position": np.array([0.0, 0.0, 0.0]),
        "target_position": np.array([0.0, 1.0, 0.0]),
        "expected": "Facing +Y (yaw=90°, pitch=0)"
    },
    {
        "position": np.array([0.0, 0.0, 0.0]),
        "target_position": np.array([0.0, 0.0, 1.0]),
        "expected": "Facing downward (pitch=-90°)"
    },
    {
        "position": np.array([0.0, 0.0, 0.0]),
        "target_position": np.array([0.0, 0.0, -1.0]),
        "expected": "Facing upward (pitch=+90°)"
    },
    {
        "position": np.array([0.0, 0.0, 0.0]),
        "target_position": np.array([1.0, 1.0, 0.0]),
        "expected": "Facing diagonal (yaw=45°, pitch=0)"
    },
]

# === Run tests ===
for i, case in enumerate(test_cases):
    result = orientation_to_target(case["position"], case["target_position"])
    roll, pitch, yaw = np.rad2deg(result)

    print(f"Test {i+1}: {case['expected']}")
    print(f"→ Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°\n")



### TEST relative pos ###
def relative_pos(i, j, type):
    rel_pos = []
    if type == "aircraft":
        rel_pos = np.array(Aircrafts[j].get_pos()) - np.array(Aircrafts[i].get_pos())
    elif type == "base":
        rel_pos = np.array(bases[j]) - np.array(Aircrafts[i].get_pos())
    rel_pos = np.where(np.abs(rel_pos) < 1e-6, 0.0, rel_pos)
    return rel_pos

# === GLOBAL MOCK DATA ===
Aircrafts = [
    type("Aircraft", (), {"get_pos": lambda self: np.array([100.0, 200.0, -500.0])})(),
    type("Aircraft", (), {"get_pos": lambda self: np.array([130.0, 180.0, -490.0])})()
]

bases = [
    np.array([500.0, 500.0, 0.0]),
    np.array([50.0, 50.0, 0.0])
]

def relative_pos(i, j, type):
    rel_pos = []
    if type == "aircraft":
        rel_pos = np.array(Aircrafts[j].get_pos()) - np.array(Aircrafts[i].get_pos())
    elif type == "base":
        rel_pos = np.array(bases[j]) - np.array(Aircrafts[i].get_pos())
    rel_pos = np.where(np.abs(rel_pos) < 1e-6, 0.0, rel_pos)
    return rel_pos

# === TEST CASES ===

print("=== Test 1: relative_pos (aircraft) ===")
rp_aircraft = relative_pos(0, 1, "aircraft")
print("Aircraft 0 Position:", Aircrafts[0].get_pos())
print("Aircraft 1 Position:", Aircrafts[1].get_pos())
print("Relative Position:", rp_aircraft.round(2))

# Expected: [30, -20, 10] (A1 - A0)

print("\n=== Test 2: relative_pos (base) ===")
rp_base = relative_pos(0, 0, "base")
print("Aircraft 0 Position:", Aircrafts[0].get_pos())
print("Base 0 Position:", bases[0])
print("Relative Position:", rp_base.round(2))

# Expected: [400, 300, 500] (base - aircraft)



### TEST relative polar pos ###
def relative_polar_pos(self, i, j, type):
    rel_pos = self.relative_pos(i, j, type)
    x, y, z = rel_pos

    r = np.linalg.norm(rel_pos)  # sqrt(x^2 + y^2 + z^2)
    
    if r < 1e-6:
        theta = 0.0
        phi = 0.0
    else:
        # θ: Elevation from forward axis (X)
        theta = np.arccos(z / r)  # 0 = straight ahead, π = directly behind

        # φ: Azimuth around forward axis (in YZ plane)
        phi = np.arctan2(y, x)    # yaw-like rotation around X

    return [r/self.max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

# === Global mock setup ===
max_size = 1000.0

Aircrafts = [
    type("Aircraft", (), {"get_pos": lambda self: np.array([0.0, 0.0, -500.0])})(),
    type("Aircraft", (), {"get_pos": lambda self: np.array([0.0, 100.0, -600.0])})()
]

# === Helper function: relative position ===
def relative_pos(i, j, target_type):
    return Aircrafts[j].get_pos() - Aircrafts[i].get_pos()

# === Function to test: relative polar position ===
def relative_polar_pos(i, j, target_type):
    rel_pos = relative_pos(i, j, target_type)
    x, y, z = rel_pos
    r = np.linalg.norm(rel_pos)

    if r < 1e-6:
        theta = 0.0
        phi = 0.0
    else:
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)

    return [r / max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

# === Run test ===
print("\n--- Testing relative_polar_pos ---")

result = relative_polar_pos(0, 1, "aircraft")
print("Computed:")
print(f"  r/max_size = {result[0]:.4f}")
print(f"  cos(theta) = {result[1]:.4f}, sin(theta) = {result[2]:.4f}")
print(f"  cos(phi)   = {result[3]:.4f}, sin(phi)   = {result[4]:.4f}")

# Expected calculation
rel = np.array([0, 100, -100])
r = np.linalg.norm(rel)
theta = np.arccos(rel[2] / r)
phi = np.arctan2(rel[1], rel[0])
expected = [r / max_size, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

print("Expected:")
print(f"  r/max_size = {expected[0]:.4f}")
print(f"  cos(theta) = {expected[1]:.4f}, sin(theta) = {expected[2]:.4f}")
print(f"  cos(phi)   = {expected[3]:.4f}, sin(phi)   = {expected[4]:.4f}")



### TEST relative vel ###
def relative_vel(self, i, j, type):
    rel_vel = []
    if type == "aircraft":
        rel_vel = np.array(self.Aircrafts[j].get_absolute_vel())- np.array(self.Aircrafts[i].get_absolute_vel())
    elif type == "base":
        rel_vel = np.zeros(3) - np.array(self.Aircrafts[i].get_absolute_vel())

    # Clamp very small values to avoid instability
    rel_vel = np.where(np.abs(rel_vel) < 1e-6, 0.0, rel_vel)
    return rel_vel

### TEST relative polar vel ###
def relative_polar_vel(self, i, j, type):
    rel_vel = self.relative_vel(i, j, type)
    x, y, z = rel_vel

    r = np.linalg.norm(rel_vel)  # Speed (magnitude)

    if r == 0:
        theta = 0.0
        phi = 0.0
    else:
        theta = np.arccos(z / r)        # Inclination from z-axis
        phi = np.arctan2(y, x)          # Azimuth in xy-plane

    return [r/343, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)]

### TEST check missile tone ###
def check_missile_tone(self, agent_index):
    new_missile_tone_attack, new_missile_target = self.Aircrafts[agent_index].get_missile_tone_attack()
    new_missile_tone_defence = self.Aircrafts[agent_index].get_missile_tone_defence()
    attack_cone = self.Aircrafts[agent_index].get_cone()
    attack_pos = self.Aircrafts[agent_index].get_pos()
    attack_vel = self.Aircrafts[agent_index].get_absolute_vel()
    team = self.Aircrafts[agent_index].get_team()
    possible_targets = []
    base_target = -1
    max_defence_tone = 0

    if new_missile_tone_defence < 0.5:
        for i, aircraft in enumerate(self.Aircrafts):
            if i != agent_index and aircraft.is_alive() and aircraft.get_team() != team:
                defence_cone = aircraft.get_cone()
                defence_pos = aircraft.get_pos()
                defence_vel = aircraft.get_absolute_vel()
                intersect = self.check_intersect_cones(attack_cone, attack_pos, attack_vel,
                                                        defence_cone, defence_pos, defence_vel)

                intersected = self.check_intersect_cones(defence_cone, defence_pos, defence_vel,
                                                            attack_cone, attack_pos, attack_vel)
                if intersect:
                    possible_targets.append(self.possible_agents[i])

                defence_tone, _ = aircraft.get_missile_tone_attack()
                if intersected and defence_tone > max_defence_tone:
                    max_defence_tone = defence_tone

        for i, base in enumerate(self.bases):
            if i != team:
                attack_angle, attack_min_dist, attack_max_dist = attack_cone
                is_in_cone = self.is_within_cone(attack_pos, attack_vel, base,
                                                    attack_angle, attack_min_dist, attack_max_dist)

                dist = np.linalg.norm(self.relative_pos(agent_index, i, 'base'))
                is_in_vuln = dist < self.bases_vulnerability_distance

                if is_in_vuln and is_in_cone:
                    base_target = i

    if base_target != -1:
        # targeting bases overwrite
        if new_missile_target == "base":
            new_missile_tone_attack += self.stepwise_tone_increment
        elif new_missile_target == 'none':
            new_missile_target = 'base'
            new_missile_tone_attack = self.stepwise_tone_increment

    elif len(possible_targets)>0:
        # targeting aircraft
        if new_missile_target in possible_targets:
            new_missile_tone_attack = np.clip(new_missile_tone_attack +
                                                self.stepwise_tone_increment, 0, 1)
        else:
            new_missile_target = np.random.choice(possible_targets)
            new_missile_tone_attack = self.stepwise_tone_increment

    else:
        new_missile_target = "none"
        new_missile_tone_attack = 0

    new_missile_tone_defence = max_defence_tone

    return new_missile_tone_attack, new_missile_tone_defence, new_missile_target

### TEST is_within_cone ###
def is_within_cone(self,cone_origin, cone_direction, target_position, half_angle_rad, min_dist, max_dist):
    vector_to_target = np.array(target_position) - np.array(cone_origin)
    distance = np.linalg.norm(vector_to_target)

    if distance < min_dist or distance > max_dist:
        return False

    direction_to_target = vector_to_target / distance

    cone_direction = cone_direction / np.linalg.norm(cone_direction)
    dot = np.dot(cone_direction, direction_to_target)

    return dot >= np.cos(np.deg2rad(half_angle_rad))

### TEST check_intersect_cones ###
def check_intersect_cones(self, attack_cone, attack_pos, attack_vel, defence_cone, defence_pos, defence_vel):
    attack_angle, attack_min_dist, attack_max_dist = attack_cone
    defence_angle, defence_min_dist, defence_max_dist = defence_cone

    # Check if defender is in attacker's forward cone
    attacker_check = self.is_within_cone(attack_pos, attack_vel, defence_pos,
                                    attack_angle, attack_min_dist, attack_max_dist)
    # Check if attacker is in defender's rearward vulnerability cone
    defender_check = self.is_within_cone(defence_pos, -defence_vel, attack_pos,
                                    defence_angle, defence_min_dist, defence_max_dist)
    return attacker_check and defender_check


### TEST action_translation_layer ###
def Action_Translation_Layer(action):
    """
    Translates a high-level action (target velocity vector in body frame)
    into AoA, roll, and speed setpoints. All outputs are normalized in [-1, 1].

    Parameters:
        action (np.ndarray): [up_angle_norm, side_angle_norm, speed_norm]
            - up_angle_norm ∈ [-1, 1]: vertical angle relative to forward
            - side_angle_norm ∈ [-1, 1]: lateral angle relative to forward
            - speed_norm ∈ [-1, 1]: desired airspeed (normalized)

    Returns:
        list: [AoA_norm, sideslip_norm (0), roll_norm, speed_norm]
    """

    # === 1. Decode normalized inputs ===
    max_angle_rad = np.deg2rad(45)  # max pitch/yaw angle in radians
    v_up   = action[0] * max_angle_rad
    v_side = action[1] * max_angle_rad
    v_speed = action[2]  # Already normalized

    # === 2. Construct target direction vector in body frame (Z+ down) ===
    vx = np.cos(v_up) * np.cos(v_side)
    vy = np.cos(v_up) * np.sin(v_side)
    vz = np.sin(v_up)  # positive Z is down

    threshold_rad = np.deg2rad(5)
    if abs(v_up) > threshold_rad or abs(v_side) > threshold_rad:
        AoA_rad = v_up
        sideslip_rad = v_side 
        roll_rad = 0
    else:
        AoA_rad = np.acos(vx)
        sideslip_rad = 0
        roll_rad = np.atan2(vy, vz)  # angle from +Z toward +Y (CCW)

    AoA_norm = AoA_rad /np.pi
    sideslip_norm = sideslip_rad / np.pi
    roll_norm = roll_rad / np.pi

    return [AoA_norm, sideslip_norm, roll_norm, v_speed]

def test_Action_Translation_Layer():
    test_cases = [
        {"name": "Forward", "action": np.array([0.0, 0.0, 0.0])},
        {"name": "Climb", "action": np.array([1.0, 0.0, 0.0])},
        {"name": "Turn Right", "action": np.array([0.0, 1.0, 0.0])},
        {"name": "Climb Right", "action": np.array([1.0, 1.0, 0.0])},
        {"name": "Dive Left", "action": np.array([-1.0, -1.0, 0.0])},
    ]
    print()
    for case in test_cases:
        result = Action_Translation_Layer(case["action"])
        print(f"Test: {case['name']}")
        print(f"  Input Action: {case['action']}")
        print(f"  Output => AoA: {result[0]:.3f}, Sideslip: {result[1]:.3f}, Roll: {result[2]:.3f}, Speed: {result[3]:.3f}")
        print()

test_Action_Translation_Layer()

