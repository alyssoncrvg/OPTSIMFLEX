from pettingzoo.test import parallel_api_test
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.tune.registry import register_env
import ray

from Ambiente_SOMN.make_env import make_env

if __name__=="__main__":
    ray.init()

    env_name = "SOMN"

    register_env(env_name, lambda config: ParallelPettingZooEnv(make_env(0,3,0)))

    config = (
        PPOConfig()
        .environment(env=env_name)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
        )
        .framework(framework="torch")
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 100},  
        config=config.to_dict()
    )