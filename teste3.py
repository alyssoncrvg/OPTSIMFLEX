from pettingzoo.test import parallel_api_test
from Ambiente_SOMN.make_env import make_env
from datetime import date, datetime
# objetivo = {
#         0 : 0,
#         1 : 1,
#         2 : 2
#     }
# env = make_env(-1,3,objetivo)

# print(objetivo[0])
#  # Obter a hora atual
# hora_atual = datetime.now()

# # Formatando a hora como uma string
# hora_formatada = hora_atual.strftime("%Y-%m-%d/%H:%M:%S")
# # parallel_api_test(env, num_cycles=10000000)
# print(hora_formatada)

import pandas as pd
import matplotlib.pyplot as plt

# Carregar dados do arquivo CSV usando a biblioteca pandas
dados_csv = pd.read_csv("C:/Users/Alysson/Documents/GitHub/OPTSIMFLEX/plots/2024-02-23/17-21-12/Yard.csv")


# Filtrar dados para cada agente
agentes = dados_csv['Agent_ID'].unique()

# Criar um gráfico de linhas para cada agente
for agente in agentes:
    dados_agente = dados_csv[dados_csv['Agent_ID'] == agente]
    plt.plot(dados_agente['Step'], dados_agente['Yard'], label=agente)

# Adicionar rótulos e legenda ao gráfico
plt.xlabel('Step')
plt.ylabel('Yard Value')
plt.title('Variação do Yard ao longo dos Steps para Cada Agente')
plt.legend()

# Exibir o gráfico
plt.savefig("C:/Users/Alysson/Documents/GitHub/OPTSIMFLEX/plots/2024-02-23/17-21-12/yard.png")