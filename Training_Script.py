import math
import os
from typing import Optional

import imageio
import wandb
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.tune import RunConfig, TuneConfig

from EnvironmentClass import AerialBattle
import numpy as np
import yaml
import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

#TODO: configure replay buffer for training
#TODO: configure the self_play routine for team vs team

Folder = 'Training_Runs'
RunName = 'Test_Good_Flight_2' #name of the directory inside Training Runs
storage_path = os.path.join("/home/lsp/Desktop/ThesisCode",
                                Folder)

os.environ["WANDB_API_KEY"] = "1b8b77cc6fc3631890702b9ecbfed2fdc1551347"

### Custom Callbacks ###
class SaveArtifactsOnCheckpoint(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Create folder to store artifacts
        if algorithm.iteration % alg_config['checkpoint_freq'] == 0:
            trial_name = os.path.basename(algorithm._logdir)
            trial_dir = os.path.join(storage_path, RunName, trial_name)
            checkpoint_dir = os.path.join(trial_dir, f"checkpoint_{result['training_iteration']-1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Access the env (only works if using num_env_runners = 1)
            env = algorithm.env_creator({})

            for i in range(3):
                checkpoint_dir_i = os.path.join(checkpoint_dir, f"{i}")
                os.makedirs(checkpoint_dir_i, exist_ok=True)

                # Reset environment and collect rendered frames
                obs, _ = env.reset()
                terminated = {"__all__": False}
                frames = []
                print(f"Evaluating...")
                while not(terminated["__all__"]) and len(frames)<alg_config['checkpoint_length']:
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = policy_mapping_fn(agent_id)
                        p = algorithm.get_policy(policy_id)
                        obs_tensor = torch.as_tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                        action, _, _ = p.compute_single_action(agent_obs, explore=False)
                        actions[agent_id] = action

                    obs, _, terminated, truncated, _ = env.step(actions)
                    terminated["__all__"] = terminated["__all__"] or truncated["__all__"]

                    if hasattr(env, "render"):
                        frame = env.render(mode="rgb_array")
                        frames.append(frame)

                # Save video
                video_path = os.path.join(checkpoint_dir_i, f"episode_video.mp4")
                imageio.mimsave(video_path, frames, fps=10)

                # Save trajectory plot
                if hasattr(env, "render_trajectory"):
                    env.render_trajectory(checkpoint_dir_i)

                # Save telemetry plots
                if hasattr(env, "plot_telemetry"):
                    env.plot_telemetry(checkpoint_dir_i)

                # Save telemetry plots
                if hasattr(env, "plot_rewards"):
                    env.plot_rewards(checkpoint_dir_i)

            print(f"Finished_Checkpoint at {checkpoint_dir}")

class CustomWandbCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.initialized = False  # only init once per worker

    def on_train_result(self, *, algorithm, result, **kwargs):
        trial_name = os.path.basename(algorithm._logdir)
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        if not self.initialized:
            wandb.init(project="aerial-battle", group=f"{RunName}",
                       name=f'{RunName}/{trial_name}',config=algorithm.config, mode="online")
            self.initialized = True

        metrics = {}

        # Add environment-wide metrics if available
        env_metrics = result.get("env_runners", {})
        metrics["reward_mean"] = env_metrics.get("episode_reward_mean", None)
        metrics["reward_max"] = env_metrics.get("episode_reward_max", None)
        metrics["reward_min"] = env_metrics.get("episode_reward_min", None)
        metrics["episode_len_mean"] = env_metrics.get("episode_len_mean", None)

        # Add learner stats (assuming multi-agent and you're interested in 'team_0')
        learner_stats = result.get("info", {}).get("learner", {}).get("team_0", {}).get("learner_stats", {})

        # Filter only the metrics you're interested in
        for key in ["entropy", "kl", "cur_lr", "total_loss", "policy_loss", "vf_loss"]:
            if key in learner_stats:
                metrics[key] = learner_stats[key]

        metrics = {
            k: float(v) for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        wandb.log(metrics, step=result['training_iteration'])


class CallbacksBroker(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.Artifacts = SaveArtifactsOnCheckpoint()
        self.WandbCallBack = CustomWandbCallback()

    def on_train_result(self, *, algorithm, result, **kwargs):
        self.Artifacts.on_train_result(algorithm=algorithm, result=result, **kwargs)
        self.WandbCallBack.on_train_result(algorithm=algorithm, result=result, **kwargs)



### Run Section ###
with open(f"{Folder}/{RunName}/{RunName}_config.yaml") as f:
    yaml_config = yaml.load(f, Loader=yaml.FullLoader)

alg_config = yaml_config['alg_config']
env_config = yaml_config['env_config']
uav_config = yaml_config['uav_config']


def env_creator(cfg):
    return AerialBattle(env_config, uav_config, discretize=True)
register_env("aerial_battle", env_creator)


dummy_env = AerialBattle(env_config=env_config, UAV_config=uav_config)
obs_space = dummy_env.get_observation_space('agent_0_0')
act_space = dummy_env.get_action_space("agent_1_0")
dummy_env.close()

### section for loading saved policies in self_play ###
#checkpoint_path = "/path/to/checkpoint_000200"  # <- Update this

# Load full algorithm from checkpoint
#restored_algo = Algorithm.from_checkpoint(checkpoint_path)
#team_B_policy = restored_algo.get_policy("team_B_policy")
###

policies = {
    "team_0": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": [256, 256],
                  ""
                  "fcnet_activation": 'tanh'},
    }),
    "team_1": (None, obs_space, act_space, {
        "model": {"fcnet_hiddens": [512, 512],
                  "fcnet_activation": 'tanh'},
    }),
}


### Define shared policy (CTDE) ###
def policy_mapping_fn(agent_id, episode=0, **kwargs):
    if agent_id.startswith("agent_0"):
        return "team_0"
    if agent_id.startswith("agent_1"):
        return "team_1"
    if agent_id.startswith("agent_2"):
        return "team_2"
    if agent_id.startswith("agent_3"):
        return "team_3"


algo_config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(env="aerial_battle", env_config={})
    .training(
        train_batch_size=tune.grid_search(alg_config['batch_size_per_learner']),
        lr_schedule=alg_config['lr'],
        gamma=tune.grid_search(alg_config['gamma']),
        grad_clip = 50
    )
    .env_runners(
        num_env_runners=15, #15
        num_envs_per_env_runner=2,
        num_cpus_per_env_runner=1, #1
        num_gpus_per_env_runner=0.065,
        batch_mode="truncate_episodes",
        sample_timeout_s=120
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=['team_0'],
    )
    .callbacks(CallbacksBroker)
)


algo_config = algo_config.to_dict()

# Use Tuner with correct RunConfig
tuner = tune.Tuner(
    trainable="PPO",
    param_space=algo_config,
    tune_config= TuneConfig(
        trial_name_creator=lambda trial: f"trial_{trial.trial_id[:10]}",
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id[:10]}"
    ),
    run_config=RunConfig(
        name=RunName,
        storage_path=storage_path,
        stop={"training_iteration": alg_config['train_iterations']},

        checkpoint_config=tune.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=alg_config['checkpoint_freq'],
        ),
        failure_config=tune.FailureConfig(
            max_failures=2,
        )
    )
)

# Run training
tuner.fit()

