import argparse

from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from Ambiente_SOMN.make_env import make_env

from ray.air.integrations.wandb import WandbLoggerCallback
from Ambiente_SOMN.MyCallbacks import MyCallbacks

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=0,
    help="Number of GPUs to use for training.",
)

parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)

parser.add_argument(
    "--num_agents",
    type=int,
    default=3
)

parser.add_argument(
    "--objetivo",
    type=int,
    default=0
)

def getPolicie(objetivo):
    pass

if __name__ == "__main__":
    args = parser.parse_args()

    objetivo = {
        0 : 0,
        1 : 1,
        2 : 2
    }

    def env_creator(args):
        return ParallelPettingZooEnv(make_env(-1, 3, objetivo))

    env = env_creator({})
    register_env("SOMN", env_creator)

    config = (
        PPOConfig()
        .environment("SOMN")
        .callbacks(MyCallbacks)
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=1)#QUANDO COLOCADO EM 2 DÁ ERRO NA MÁQUINA, DESCOBRIR O PQ!!!!
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    stop = {"timesteps_total": 250000}

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
            callbacks=[WandbLoggerCallback(project="Treinamento multi-agent", group="Testes Callback")]
        ),
        param_space=config,
    ).fit()