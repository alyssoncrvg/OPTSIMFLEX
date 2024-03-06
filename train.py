import argparse
import os
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from Ambiente_SOMN.make_env import make_env

from ray.air.integrations.wandb import WandbLoggerCallback
from Ambiente_SOMN.MyCallbacks import MyCallbacks

objetivo = {
        0 : 0,
        1 : 1,
        2 : 2
    }

policies = ["Lucro", "Variabilidade", "Sustentabilidade"]

def env_creator(args):
    return ParallelPettingZooEnv(make_env(-1, 3, objetivo))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    policie = objetivo[int(agent_id)]
    policie = policies[policie]
    return policie

if __name__ == "__main__":

    env = env_creator({})
    register_env("SOMN", env_creator)

    config = (
        PPOConfig()
        .environment("SOMN")
        .callbacks(MyCallbacks)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rollouts(num_rollout_workers=1)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    stop = {"timesteps_total": 60000}

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
            callbacks=[WandbLoggerCallback(project="Treinamento multi-agent", group="Testes Comunicação")]
        ),
        param_space=config,
    ).fit()