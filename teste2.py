from ray.rllib.utils import check_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from Ambiente_SOMN.make_env import make_env

check_env(ParallelPettingZooEnv(make_env(0,3,0)))