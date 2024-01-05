from pettingzoo.test import parallel_api_test

from Ambiente_SOMN.make_env import make_env

if __name__=="__main__":
    env = make_env(0,3,0)

    parallel_api_test(env, num_cycles=1000000)