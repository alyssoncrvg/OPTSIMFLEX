from pettingzoo.test import parallel_api_test
from Ambiente_SOMN.Demand import Demand
from Ambiente_SOMN.make_env import make_env
from datetime import date, datetime
from Ambiente_SOMN.Statistcs import Statistcs

objetivo = {
        0 : 0,
        1 : 1,
        2 : 2
    }
env = make_env(-1,3)

parallel_api_test(env, num_cycles=1000000)
# a = Demand(M=10,
#                 N=10,
#                 MAXDO=100,
#                 MAXAM=2,
#                 MAXPR=2,
#                 MAXPE=10,
#                 MAXFT=5,
#                 MAXMT=3,
#                 MAXTI=2,
#                 MAXEU = 5,
#                 atraso=-1,
#                 t = 0,
#                 )
# a(0,0,1)
# print(a)