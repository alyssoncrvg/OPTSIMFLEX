import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Passo 2: Leia os dados do CSV
dados = pd.read_csv('C:/Users/Alysson/Desktop/SOMN2/OPTSIMFLEX/plots/2024-03-05/16-59-59/Yard.csv')

# Passo 3: Crie subgráficos para cada Agent_ID
plt.figure(figsize=(10, 6))

# Itera sobre os Agent_ID
for agent_id in dados['Agent_ID'].unique():
    agente_dados = dados[dados['Agent_ID'] == agent_id]
    plt.plot(agente_dados['Step'], agente_dados['Yard'], marker='o', label=f'Agent_ID {agent_id}')

# Personalize o gráfico
plt.title('Gráfico a partir de dados CSV')
plt.xlabel('Step')
plt.ylabel('Yard')
plt.legend()
plt.grid(True)
plt.show()
