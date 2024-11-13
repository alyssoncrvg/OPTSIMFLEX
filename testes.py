from Ambiente_SOMN.Demand import Demand


list = []

Y=10,
M=10
N=10
MAXDO=100
MAXAM=2
MAXPR=2
MAXPE=10
MAXFT=5
MAXMT=3
MAXTI=2
MAXEU = 5 
atraso=-1
numAgents=3

demand_list = [
    Demand(M, N, MAXDO, MAXAM, MAXPR, MAXPE, MAXFT, MAXMT, MAXTI, MAXEU, 0, atraso)
    for _ in range(1000000)  # ou a quantidade desejada de instâncias
]

# Atualize a lista chamando o método que inicializa os valores, caso necessário
for demand in demand_list:
    demand(0, 0, 0)  # Passa os argumentos adequados para chamar cada demanda e inicializar `PR`
# Encontra o maior valor de PR na lista de demandas
max_pr = max(demand.PR for demand in demand_list)
min_pr = min(demand.PR for demand in demand_list)

print("Maior valor de PR:", max_pr)
print("menor valor: ", min_pr)