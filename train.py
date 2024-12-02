import os
import wandb
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from Ambiente_SOMN.make_env import make_env

from ray.air.integrations.wandb import WandbLoggerCallback
from Ambiente_SOMN.MyCallbacks import MyCallbacks

# policies = ["Lucro", "Variabilidade", "Sustentabilidade"]
policies = ["All"]

def env_creator(args):
    return ParallelPettingZooEnv(make_env(-1, 3))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    policie = policies[0]
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
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    stop = {"timesteps_total": 4000 * 100}

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
            callbacks=[WandbLoggerCallback(project="testes fuzzer control", group="6")]
        ),
        param_space=config,
    ).fit()