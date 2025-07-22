def get_individual_reward_Flight_V1(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 3
        self.actions.append(action[:-1])

        # Flight Related Rewards
        # envelope:

        #Energy  and Speed
        mid_E = 0.3
        a_E = 10
        specific_kinetic_energy = 0.5 * np.linalg.norm(telemetry["velocity"][-1])**2
        specific_potential_energy = altitude * 9.8
        weighted_normalized_energy = (specific_kinetic_energy + specific_potential_energy*0.1) / 88224 

        reward['Energy'] = ((2/(1 + np.exp(-a_E * (weighted_normalized_energy - mid_E))))-1) * 0.3
        
        
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            std_surfaces = (std_1+std_2+std_3)/3
            a_CE = 20
            mid_CE = 0.2
            reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (std_surfaces - mid_CE)))) * 0.2
        else:
            reward['Control_erratic'] = 0.0

        a_A = 20
        mid_A = 0.3
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.3

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / self.max_size, 1.0) * 0.2

        reward['Stall Speed'] = -((vel[0]<100) * 3)

        normalized_reward = sum(reward.values()) + 0.1


        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V2(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 5
        self.actions.append(action[:-1])

        # Flight Related Rewards
        # envelope:

        #Energy  and Speed
        mid_E = 0.3
        a_E = 10
        specific_kinetic_energy = 0.5 * np.linalg.norm(telemetry["velocity"][-1])**2
        specific_potential_energy = altitude * 9.8
        weighted_normalized_energy = (specific_kinetic_energy + specific_potential_energy*0.1) / 88224 

        reward['Energy'] = ((2/(1 + np.exp(-a_E * (weighted_normalized_energy - mid_E))))-1) * 0.2
        
        
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            std_surfaces = (std_1+std_2+std_3)/3
            a_CE = 30
            mid_CE = 0.2
            reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (std_surfaces - mid_CE)))) * 0.2
        else:
            reward['Control_erratic'] = 0.0

        a_A = 20
        mid_A = 0.3
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.1

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / self.max_size, 1.0) * 0.1

        a_S = 20
        mid_S = 0.3
        abs_speed = abs(180-vel[0]) / 343
        reward['Cruise Speed'] = (1-1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.2

        a_SF = 20
        mid_SF = 0.3
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = abs(roll - 5) + abs(pitch - 10)
        reward['Stable Flight'] = (1-1/(1 + np.exp(-a_SF * (abs_attitude- mid_SF)))) * 0.2


        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.1


        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V3(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 8
        self.actions.append(action[:-1])

        # Flight Related Rewards
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            std_surfaces = (std_1+std_2+std_3)/3
            a_CE = 30
            mid_CE = 0.2
            reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (std_surfaces - mid_CE)))) * 0.2
        else:
            reward['Control_erratic'] = 0.0

        a_A = 20
        mid_A = 0.3
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.3

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / self.max_size, 1.0) * 0.1

        a_S = 20
        mid_S = 0.3
        abs_speed = abs(200-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.3

        a_SF = 20
        mid_SF = 0.3
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/4, 0) + max(abs(pitch)-np.pi/8, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.1

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.1

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V3_1(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 20
        self.actions.append(action[:-1])

        # Flight Related Rewards
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            std_surfaces = (std_1+std_2+std_3)/3
            a_CE = 30
            mid_CE = 0.2
            reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (std_surfaces - mid_CE)))) * 0.2
        else:
            reward['Control_erratic'] = 0.0

        a_A = 20
        mid_A = 0.3
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.2

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.2

        a_S = 20
        mid_S = 0.3
        abs_speed = abs(200-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.2

        a_SF = 20
        mid_SF = 0.3
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/10, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.3

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.1

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V3_2(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 20
        self.actions.append(action[:-1])

        # Flight Related Rewards
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            std_surfaces = (std_1+std_2+std_3)/3
            a_CE = 30
            mid_CE = 0.2
            reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (std_surfaces - mid_CE)))) * 0.2
        else:
            reward['Control_erratic'] = 0.0

        a_A = 40
        mid_A = 0.1
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.1

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.1

        a_S = 40
        mid_S = 0.1
        abs_speed = abs(200-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.3

        a_SF = 30
        mid_SF = 0.2
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/10, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.3

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.1

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V3_3(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 20
        self.actions.append(action[:-1])

        # Flight Related Rewards
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            std_surfaces = (std_1+std_2+std_3+std_4)/4
            a_CE = 30
            mid_CE = 0.2
            reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (std_surfaces - mid_CE)))) * 0.1
        else:
            reward['Control_erratic'] = 0.0

        a_A = 40
        mid_A = 0.1
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.3

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.2

        a_S = 40
        mid_S = 0.1
        abs_speed = abs(200-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.3

        a_SF = 30
        mid_SF = 0.2
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/10, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.2

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.1

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V4(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
    terminated = False
    truncated = False
    reward = {}
    aircraft = self.Aircrafts[agent_index]
    team = aircraft.get_team()
    telemetry = aircraft.get_agent_telemetry()
    vel = telemetry['velocity'][-1]
    altitude = -telemetry['position'][-1][2]
    agent_pos = telemetry['position'][-1]
    actions = telemetry['commands']

    # Flight Related Rewards

    #we want to push the model into making delta actions small enough that the PID can follow them:
    # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
    # thus we want a step of max 2/20 = 0.1
    d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
    d_surfaces = ((d_1+d_2+d_3+d_4)/4)
    a_CE = 30
    mid_CE = 0.1
    reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.2

    a_A = 30
    mid_A = 0.2
    abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
    reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.3

    center_dist = aircraft.get_distance_from_centroid(self.bases)
    reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.1

    a_S = 30
    mid_S = 0.2
    abs_speed = abs(250-vel[0]) / 343
    reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.2

    a_SF = 30
    mid_SF = 0.2
    roll, pitch, _ = telemetry['orientation'][-1]  # radians
    abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
    reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.2

    reward['Stall Speed'] = -((vel[0]<100) * 3)
    
    normalized_reward = sum(reward.values()) + 0.2

    acc=0
    if len(telemetry['acceleration']) > 5:
        acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

    #check collision or over-g
    if (self.check_collision(agent_index) 
        or acc >= (20*9.81) 
        or vel[0]<0 
        or altitude>self.env_size[2]
        or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
        self.Aircrafts[agent_index].kill()
        terminated = True
        normalized_reward = -100

    self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
    return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V1_SAC(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        actions = telemetry['commands']

        # Flight Related Rewards

        #we want to push the model into making delta actions small enough that the PID can follow them:
        # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
        # thus we want a step of max 2/20 = 0.1
        d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
        d_surfaces = ((d_1+d_2+d_3+d_4)/4)
        a_CE = 30
        mid_CE = 0.1
        reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.3

        a_A = 30
        mid_A = 0.2
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.4

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.1

        a_S = 30
        mid_S = 0.2
        abs_speed = abs(250-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.1

        a_SF = 30
        mid_SF = 0.2
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.2

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.2

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V1_2_SAC(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        actions = telemetry['commands']

        # Flight Related Rewards

        #we want to push the model into making delta actions small enough that the PID can follow them:
        # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
        # thus we want a step of max 2/20 = 0.1
        d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
        d_surfaces = ((d_1+d_2+d_3+d_4)/4)
        a_CE = 30
        mid_CE = 0.1
        reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.1

        a_A = 30
        mid_A = 0.2
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.4

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.2

        a_S = 30
        mid_S = 0.2
        abs_speed = abs(250-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.2

        a_SF = 30
        mid_SF = 0.2
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.1

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.2

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V1_3_SAC(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        actions = telemetry['commands']

        # Flight Related Rewards

        #we want to push the model into making delta actions small enough that the PID can follow them:
        # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
        # thus we want a step of max 2/20 = 0.1
        d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
        d_surfaces = ((d_1+d_2+d_3+d_4)/4)
        a_CE = 30
        mid_CE = 0.1
        reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.05

        a_A = 30
        mid_A = 0.15
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.4

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.2

        a_S = 40
        mid_S = 0.1
        abs_speed = abs(200-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.2

        a_SF = 30
        mid_SF = 0.15
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.15

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.2

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Flight_V1_4_SAC(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        actions = telemetry['commands']

        # Flight Related Rewards

        #we want to push the model into making delta actions small enough that the PID can follow them:
        # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
        # thus we want a step of max 2/20 = 0.1
        d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
        d_surfaces = ((d_1+d_2+d_3+d_4)/4)
        a_CE = 30
        mid_CE = 0.2
        reward['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.05

        a_A = 30
        mid_A = 0.2
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.2

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.4

        a_S = 30
        mid_S = 0.2
        abs_speed = abs(200-vel[0]) / 343
        reward['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.15

        a_SF = 30
        mid_SF = 0.2
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
        reward['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.2

        reward['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward = sum(reward.values()) + 0.4

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_reward_Pursuit(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        smoothing_window = 3
        self.actions.append(action[:-1])

        # Flight Related Rewards
        # envelope:

        #Energy  and Speed
        mid_E = 0.3
        a_E = 10
        specific_kinetic_energy = 0.5 * np.linalg.norm(telemetry["velocity"][-1])**2
        specific_potential_energy = altitude * 9.8
        weighted_normalized_energy = (specific_kinetic_energy + specific_potential_energy*0.1) / 88224 

        reward['Energy'] = ((2/(1 + np.exp(-a_E * (weighted_normalized_energy - mid_E))))-1) * 0.3
        
        
        if len(self.actions) > smoothing_window:
            window = np.array(self.actions)[-smoothing_window:]
            stds = np.std(np.array(window), axis=0)
            std_1, std_2, std_3, std_4 = stds
            reward['Control_erratic'] = -((((std_1+std_2+std_3+std_4)/4)>0.2) * 0.1)
        else:
            reward['Control_erratic'] = 0.0

        reward['Stall Speed'] = -((vel[0]<100) * 3)


       # MISSION RELATED REWARDS
        #direction
        closest_enemy_plane = None
        c = 0
        dist = 1000000
        for i, enemy_aircraft in enumerate(self.Aircrafts):
            if enemy_aircraft.get_team() != team and enemy_aircraft.is_alive():
                rel_pos = np.linalg.norm(self.relative_pos(agent_index, i, 'aircraft'))
                if rel_pos < dist:
                    dist = rel_pos
                    closest_enemy_plane = enemy_aircraft

        if closest_enemy_plane is not None:
            # Pursuit
            mid_AN = 0.5
            a_AN = -18
            track_angle, adverse_angle = self.get_track_adverse_angles_norm(aircraft, closest_enemy_plane)
            track_angle /= np.pi
            adverse_angle /= np.pi
            logistic_term = 1.0 / (1.0 + np.exp(-a_AN * (adverse_angle - mid_AN)))
            reward['Pursuit'] = ((track_angle - 2.0) * logistic_term - track_angle + 1.0) * 0.2

            # Guidance
            a_G = 18
            mid_G = 0.3
            reward['Guidance'] = ((2.0 / (1.0 + np.exp(-a_G * (mid_G - track_angle))))-1) * 0.3

            # Closure
            mid_CL = 0.5
            a_CL = 18
            closure_norm = self.get_closure_rate_norm(aircraft, closest_enemy_plane)
            reward['Closure'] = (2 * (1.0 / (1.0 + np.exp(-a_CL * (closure_norm - mid_CL)))) - 1.0) * 0.1
        else:
            reward['Pursuit'] = 0
            reward['Guidance'] = 0
            reward['Closure'] = 0

        if missile_target != 'base':
            reward['Attack'] = missile_tone_attack * 4
        else:
            reward['Attack'] = 0  # change in subsequent trainings
        reward['Defence'] = -missile_tone_defence * 5


        reward_sum = 0.1 
        for key in reward.keys():
            reward_sum += reward[key]
        
        normalized_reward = reward_sum

        ### Non-Comulative Rewards
        if kill != 'none':
            if kill != 'base':
                print("killed")
                normalized_reward = 50 
            #elif kill == 'base':
            #    reward = 1000000

        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))
            
        #check collision or over-g
        if (self.check_collision(agent_index) or acc >= (20*9.81) or vel[0]<0 or altitude>self.env_size[2]):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_reward = -100

        self.episode_rewards[self.possible_agents[agent_index]].append(reward.copy())
        return normalized_reward, terminated, truncated

def get_individual_Pursuit_V_1(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
        terminated = False
        truncated = False
        reward_Flight = {}
        reward_Pursuit = {}
        aircraft = self.Aircrafts[agent_index]
        team = aircraft.get_team()
        telemetry = aircraft.get_agent_telemetry()
        vel = telemetry['velocity'][-1]
        altitude = -telemetry['position'][-1][2]
        agent_pos = telemetry['position'][-1]
        actions = telemetry['commands']


        #### Flight Related Rewards ####

        #we want to push the model into making delta actions small enough that the PID can follow them:
        # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
        # thus we want a step of max 2/20 = 0.1
        d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
        d_surfaces = ((d_1+d_2+d_3+d_4)/4)
        a_CE = 30
        mid_CE = 0.2
        reward_Flight['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.05

        a_A = 30
        mid_A = 0.2
        abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
        reward_Flight['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.2

        center_dist = aircraft.get_distance_from_centroid(self.bases)
        reward_Flight['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.4

        a_S = 30
        mid_S = 0.2
        abs_speed = abs(200-vel[0]) / 343
        reward_Flight['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.15

        a_SF = 30
        mid_SF = 0.2
        roll, pitch, _ = telemetry['orientation'][-1]  # radians
        abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
        reward_Flight['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.2

        reward_Flight['Stall Speed'] = -((vel[0]<100) * 3)
        
        normalized_reward_Flight = sum(reward_Flight.values()) + 0.2


        #### Pursuit related Rewards ####

        #Energy  and Speed
        mid_E = 0.3
        a_E = 10
        specific_kinetic_energy = 0.5 * np.linalg.norm(telemetry["velocity"][-1])**2
        specific_potential_energy = altitude * 9.8
        weighted_normalized_energy = (specific_kinetic_energy + specific_potential_energy*0.1) / 88224 

        reward_Pursuit['Energy'] = ((2/(1 + np.exp(-a_E * (weighted_normalized_energy - mid_E))))-1) * 0.3

        # Choose Enemy Plane
        closest_enemy_plane = None
        c = 0
        dist = 1000000
        for i, enemy_aircraft in enumerate(self.Aircrafts):
            if enemy_aircraft.get_team() != team and enemy_aircraft.is_alive():
                rel_pos = np.linalg.norm(self.relative_pos(agent_index, i, 'aircraft'))
                if rel_pos < dist:
                    dist = rel_pos
                    closest_enemy_plane = enemy_aircraft


        if closest_enemy_plane is not None:
            # Pursuit_angle
            mid_AN = 0.5
            a_AN = -18
            track_angle, adverse_angle = self.get_track_adverse_angles_norm(aircraft, closest_enemy_plane)
            track_angle /= np.pi
            adverse_angle /= np.pi
            logistic_term = 1.0 / (1.0 + np.exp(-a_AN * (adverse_angle - mid_AN)))
            reward_Pursuit['Pursuit_angle'] = ((track_angle - 2.0) * logistic_term - track_angle + 1.0) * 0.2

            # Guidance
            a_G = 18
            mid_G = 0.3
            reward_Pursuit['Guidance'] = ((2.0 / (1.0 + np.exp(-a_G * (mid_G - track_angle))))-1) * 0.3

            # Closure
            mid_CL = 0.5
            a_CL = 18
            closure_norm = self.get_closure_rate_norm(aircraft, closest_enemy_plane)
            reward_Pursuit['Closure'] = (2 * (1.0 / (1.0 + np.exp(-a_CL * (closure_norm - mid_CL)))) - 1.0) * 0.1
        else:
            #TODO: insert here some guidance to go towards the base and destroy it
            reward_Pursuit['Pursuit'] = 0
            reward_Pursuit['Guidance'] = 0
            reward_Pursuit['Closure'] = 0
        
        #Sparse Pursuit Rewards:
        if missile_target != 'base':
            reward_Pursuit['Attack'] = missile_tone_attack * 4
        else:
            reward_Pursuit['Attack'] = 0  #TODO: change in subsequent trainings to destroy the base
        reward_Pursuit['Defence'] = -missile_tone_defence * 5

        normalized_reward_Pursuit = sum(reward_Pursuit.values())


        #### Reward Merge ####
        Total_Reward = {}
        Total_Reward.extend(reward_Flight)
        Total_Reward.extend(reward_Pursuit)

        normalized_total_reward = 0.3 * normalized_reward_Flight + 0.7 * normalized_reward_Pursuit

        #### Termination Condition Rewards ####
        acc=0
        if len(telemetry['acceleration']) > 5:
            acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

        #check collision or over-g
        if (self.check_collision(agent_index) 
            or acc >= (20*9.81) 
            or vel[0]<0 
            or altitude>self.env_size[2]
            or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
            self.Aircrafts[agent_index].kill()
            terminated = True
            normalized_total_reward = -100


        self.episode_rewards[self.possible_agents[agent_index]].append(Total_Reward.copy())
        return normalized_total_reward, terminated, truncated

def get_individual_reward_Pursuit_Reach(self, agent_index, action, kill, missile_tone_attack, missile_tone_defence, missile_target):
    terminated = False
    truncated = False
    reward_Flight = {}
    reward_Pursuit = {}
    Total_Reward = {}
    aircraft = self.Aircrafts[agent_index]
    team = aircraft.get_team()
    telemetry = aircraft.get_agent_telemetry()
    vel = telemetry['velocity'][-1]
    altitude = -telemetry['position'][-1][2]
    agent_pos = telemetry['position'][-1]
    actions = telemetry['commands']


    #### Flight Related Rewards ####

    #we want to push the model into making delta actions small enough that the PID can follow them:
    # PID takes at worst 250 physics steps to reach target from -1 to 1. => 250/120 = 2 seconds = 20 action_steps
    # thus we want a step of max 2/20 = 0.1
    d_1, d_2, d_3, d_4 = abs(actions[-1] - actions[-2])
    d_surfaces = ((d_1+d_2+d_3+d_4)/4)
    a_CE = 30
    mid_CE = 0.2
    reward_Flight['Control_erratic'] = -(1/(1 + np.exp(-a_CE * (d_surfaces - mid_CE)))) * 0.05

    a_A = 30
    mid_A = 0.2
    abs_alt = abs(self.env_size[2]/2 - altitude) / self.env_size[2]
    reward_Flight['Altitude'] = -(1/(1 + np.exp(-a_A * (abs_alt - mid_A)))) * 0.4

    center_dist = aircraft.get_distance_from_centroid(self.bases)
    reward_Flight['InBounds'] = -min(center_dist / (self.max_size/2), 1.0) * 0.2

    a_S = 30
    mid_S = 0.2
    abs_speed = abs(200-vel[0]) / 343
    reward_Flight['Cruise Speed'] = -(1/(1 + np.exp(-a_S * (abs_speed - mid_S)))) * 0.2

    a_SF = 30
    mid_SF = 0.2
    roll, pitch, _ = telemetry['orientation'][-1]  # radians
    abs_attitude = (max(abs(roll)-np.pi/6, 0) + max(abs(pitch)-np.pi/12, 0)) / (2*np.pi)
    reward_Flight['Stable Flight'] = -(1/(1 + np.exp(-a_SF * (abs_attitude - mid_SF)))) * 0.15

    Total_Reward['Stall Speed'] = -((vel[0]<100) * 3)
    
    normalized_reward_Flight = sum(reward_Flight.values()) + 0.3


    #### Pursuit related Rewards ####

    #Energy  and Speed
    mid_E = 0.3
    a_E = 10
    specific_kinetic_energy = 0.5 * np.linalg.norm(telemetry["velocity"][-1])**2
    specific_potential_energy = altitude * 9.8
    weighted_normalized_energy = (specific_kinetic_energy + specific_potential_energy*0.1) / 88224 

    reward_Pursuit['Energy'] = ((2/(1 + np.exp(-a_E * (weighted_normalized_energy - mid_E))))-1) * 0.1

    # Choose Enemy Plane
    closest_enemy_plane = None
    c = 0
    dist = 1000000
    for i, enemy_aircraft in enumerate(self.Aircrafts):
        if enemy_aircraft.get_team() != team and enemy_aircraft.is_alive():
            rel_pos = np.linalg.norm(self.relative_pos(agent_index, i, 'aircraft'))
            if rel_pos < dist:
                dist = rel_pos
                closest_enemy_plane = enemy_aircraft


    if closest_enemy_plane is not None:
        track_angle, adverse_angle = self.get_track_adverse_angles_norm(aircraft, closest_enemy_plane)
        track_angle /= np.pi
        adverse_angle /= np.pi

        # Guidance
        a_G = 18
        mid_G = 0.3
        reward_Pursuit['Guidance'] = ((2.0 / (1.0 + np.exp(-a_G * (mid_G - track_angle))))-1) * 0.8

        # Closure
        mid_CL = 0.5
        a_CL = 18
        closure_norm = self.get_closure_rate_norm(aircraft, closest_enemy_plane)
        reward_Pursuit['Closure'] = (2 * (1.0 / (1.0 + np.exp(-a_CL * (closure_norm - mid_CL)))) - 1.0) * 0.2

        if dist < aircraft.get_cone()[2]:
            reward_Pursuit['Final'] = 200
            terminated = True

    else:
        #TODO: insert here some guidance to go towards the base and destroy it
        reward_Pursuit['Pursuit'] = 0
        reward_Pursuit['Guidance'] = 0
        reward_Pursuit['Closure'] = 0
    
    #Sparse Pursuit Rewards:
    if missile_target != 'base':
        Total_Reward['Attack'] = missile_tone_attack * 4
    else:
        Total_Reward['Attack'] = 0  #TODO: change in subsequent trainings to destroy the base
    Total_Reward['Defence'] = -missile_tone_defence * 5

    normalized_reward_Pursuit = sum(reward_Pursuit.values())

    #TODO: Kill reward


    #### Reward Merge ####
    Total_Reward.update(reward_Flight)
    Total_Reward.update(reward_Pursuit)

    normalized_total_reward = 0.4 * normalized_reward_Flight + 0.6 * normalized_reward_Pursuit

    #### Termination Condition Rewards ####
    acc=0
    if len(telemetry['acceleration']) > 5:
        acc = np.linalg.norm(np.mean(telemetry['acceleration'][-5: -1]))

    #check collision or over-g
    if (self.check_collision(agent_index) 
        or acc >= (20*9.81) 
        or vel[0]<0 
        or altitude>self.env_size[2]
        or aircraft.get_distance_from_centroid(self.bases) > self.max_size):
        self.Aircrafts[agent_index].kill()
        terminated = True
        normalized_total_reward = -100


    self.episode_rewards[self.possible_agents[agent_index]].append(Total_Reward.copy())
    return normalized_total_reward, terminated, truncated