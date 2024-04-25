from pettingzoo.test import parallel_api_test
from Ambiente_SOMN.make_env import make_env
from datetime import date, datetime
from Ambiente_SOMN.Statistcs import Statistcs

# objetivo = {
#         0 : 0,
#         1 : 1,
#         2 : 2,
#         3 : 0

#     }
# env = make_env(-1,3,objetivo)

# parallel_api_test(env, num_cycles=100000)


# import pandas as pd
# import matplotlib.pyplot as plt

# # Carregar dados do arquivo CSV usando a biblioteca pandas
# dados_csv = pd.read_csv("C:/Users/Alysson/Documents/GitHub/OPTSIMFLEX/plots/2024-02-23/17-21-12/Yard.csv")


# # Filtrar dados para cada agente
# agentes = dados_csv['Agent_ID'].unique()

# # Criar um gráfico de linhas para cada agente
# for agente in agentes:
#     dados_agente = dados_csv[dados_csv['Agent_ID'] == agente]
#     plt.plot(dados_agente['Step'], dados_agente['Yard'], label=agente)

# # Adicionar rótulos e legenda ao gráfico
# plt.xlabel('Step')
# plt.ylabel('Yard Value')
# plt.title('Variação do Yard ao longo dos Steps para Cada Agente')
# plt.legend()

# # Exibir o gráfico
# plt.savefig("C:/Users/Alysson/Documents/GitHub/OPTSIMFLEX/plots/2024-02-23/17-21-12/yard.png")

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import torch

print(torch.cuda.is_available())

# import torch
# import sys
# print('__Python VERSION:', sys.version)
# print('__pyTorch VERSION:', torch.__version__)
# print('__CUDA VERSION')
# from subprocess import call
# # call(["nvcc", "--version"]) does not work
# # ! nvcc --version
# print('__CUDNN VERSION:', torch.backends.cudnn.version())
# print('__Number CUDA Devices:', torch.cuda.device_count())
# print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
# print('Active CUDA Device: GPU', torch.cuda.current_device())
# print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device())