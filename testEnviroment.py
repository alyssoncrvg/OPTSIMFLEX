from pettingzoo.test import parallel_api_test
from Ambiente_SOMN.make_env import make_env
from datetime import date, datetime
from Ambiente_SOMN.Statistcs import Statistcs

objetivo = {
        0 : 0,
        1 : 1,
        2 : 2
    }
env = make_env(-1,3,objetivo)

parallel_api_test(env, num_cycles=100000)
