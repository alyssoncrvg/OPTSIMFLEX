from Ambiente_SOMN.Demand import Demand
from Ambiente_SOMN.Yard import Yard
from Ambiente_SOMN.Statistcs import Statistcs

# a biblioteca gym mudou
from gymnasium import spaces  # Discrete, Box, Tuple,  Dict
from heapdict import heapdict

# outras bibliotecas
import os
import numpy as np
import random
import numpy as np
from scipy.stats import poisson
import copy

#############################################################
from pettingzoo import ParallelEnv
#############################################################

# LOAD
MAX_LOAD = 30

# STATUS
REJECTED_W_WASTE = -2
FREE = -1
RECEIVED = 0
READY = 1
REJECTED = 2
PRODUCTION = 3
STORED = 4
DELIVERED = 5

COVERED = True
NOT_COVERED = False

IN_TIME = True
OUT_TIME = False

FINAL_STATES = [REJECTED_W_WASTE, REJECTED, STORED, DELIVERED]


class Somn(ParallelEnv):


    metadata = {
        "render_mode": ["human", "rgb_array"],
        "name": "SOMN",
        "is_parallelizable": True
    }

    """Custom Environment that follows gym interface."""
    def __init__(
        self,
        M: int,
        N: int,
        Y: int,
        MAXDO: int,
        MAXAM: int,
        MAXPR: int,
        MAXPE: int,
        MAXFT: int,
        MAXMT: int,
        MAXTI: int,
        MAXEU: int,
        #seed: int,
        atraso: int,
        numAgents: int, #Numero de agentes que vão agir no espaço
        objetivo: dict[str, int]
    ):
        super(Somn).__init__()

        
        Somn.obj_list = ['pr', 'va', 'su']
        Somn.priorq = {agente: [heapdict() for _ in Somn.obj_list] for agente in range(numAgents)}
        Somn.objetivo = objetivo
        # Somn.instance = JobShop()
        # Somn.priorqsu = heapdict()
        # Somn.priorqva = heapdict()
        Somn.time =[]
        
        """
        Variáveis de verificação do que ocorre nas transferências de demandas
        """
        self.total_acepts_produce = []
        self.total_acepts_yard = []

        for i in range(numAgents):
            Somn.time.append(1)
            self.total_acepts_produce.append(0)
            self.total_acepts_yard.append(0)

        ##########################################################################
        self.agents = {f'{i}' for i in range(numAgents)}
        self.possible_agents = [f'{i}' for i in range(numAgents)] #LISTA DOS AGENTES
        ##########################################################################
        
        # variaveis para salvar os valores para avaliar cada passo
        self.totReward = 0.0
        self.totPenalty = 0.0
        self.totPenalty2 = 0.0

        ################################## ALTERANDO AS RECOMPENSAS E PENALIDADES PARA UM DICT
        self.reward = {}
        self.penalty = {}
        ##################################

        self.rw_pr = 0.0
        self.rw_va = 0.0
        self.rw_su = 0.0
        self.variabilidade = []
        self.sustentabilidade = []
        self.F = []
        self.acoes = []
        self.atrasos_reais = []
        
        self.acao_on_state_plan = []
        self.patio_on_state_plan = []
        self.carga_on_state_plan = []
        
        ###################################################################################
        self.match = []

        for i in range(numAgents):
            self.match.append(np.zeros(N))
        ###################################################################################

        self.M = M
        self.N = N
        self.Y = Y
        self.MAXDO = MAXDO
        self.MAXAM = MAXAM
        self.MAXPR = MAXPR
        self.MAXPE = MAXPE
        self.MAXFT = MAXFT
        self.MAXMT = MAXMT
        self.MAXTI = MAXTI
        self.MAXEU = MAXEU
        # self.MT = np.random.randint(0,MAXFT,M)


        self.EU = {agent : np.random.random(M) * MAXEU for agent in range(numAgents)}
        self.BA = {agent : np.random.randint(10, 10*MAXFT, M) for agent in range(numAgents)}
        self.IN = {agent : np.random.randint(0, MAXFT, M) for agent in range(numAgents)}
        self.OU = {agent : np.random.randint(0, MAXFT, M) for agent in range(numAgents)}
        #self.seed = seed
        self.atraso = atraso # (by_frederic)
        # self.DE_state = np.zeros((N,5))

        # print('Inicializado', M, N , Y)
        
        self.DE = []  
        self.YA = []


        self.statistcs = []
        self.rejecteds = []
        self.demands_rejects_all = 0
        self.aux = []
        self.acept_reject = []

        self.MT = []

        for agent in range(numAgents):
            agentDemands = [
                Demand(
                    M, N, MAXDO, MAXAM, MAXPR, MAXPE, MAXFT, MAXMT, MAXTI, MAXEU, Somn.time[agent], self.atraso
                )
                for _ in range(N)  # ou a quantidade desejada de instâncias por agente
            ]
            self.DE.append(agentDemands)
            self.YA.append(Yard(Y))
            self.statistcs.append(Statistcs())
            self.MT.append([])
            self.acept_reject.append(0)

        ######################
        #      lb e ub       #
        ######################
        """
        (lb=lowerbound ub=upperbound) para o espaco de Observacao e Acao
        """

        # time varia de 1 a 100 (era de 1 ate 10*MAXDO + M)
        self.lb_time = 1
        # self.ub_time = 10 * self.MAXDO + self.M
        self.ub_time = 100

        # ST varia de -2 a 5
        self.lb_ST = -2
        self.ub_ST = 5
        # LT varia de 2 a (M/2 + 2)

        self.lb_LT = 1
        # self.ub_LT = int(self.M / 2) + 2    #### ACMO LT AFETADO POR LT(M) + CARGA(N)
        #self.ub_LT = self.M + self.N
        self.ub_LT = self.M * (self.MAXFT - 1) * (self.MAXAM - 1)
        
        # DI varia de 1 a (ub_time + ub_LT + MAXDO)
        self.lb_DI = 1
        self.ub_DI = self.ub_time
        # DO varia de 3 a (ub_time + ub_LT + MAXDO)
        self.lb_DO = 3
        self.ub_DO = self.ub_time + self.ub_LT + self.MAXDO
        # TP varia de 2 a (ub_time + ub_LT + 2) onde 2 e um ruido (troquei 2 pela distribuicao de poisson)
        self.lb_TP = 2
        p = [poisson.rvs(mu=self.ub_LT + MAX_LOAD) for _ in range(10000)]
        self.MAX_ATRASO = max(p)
        self.ub_TP = self.ub_time + self.ub_LT + self.MAX_ATRASO
        
        # PR varia de 0 a (M * (MAXFT-1) * MAXEU) * MAXPR
        self.lb_PR = 0
        self.ub_PR = self.M * (self.MAXFT-1) * self.MAXEU * self.MAXPR
        # CO varia de 0 a (M * (MAXFT-1) * MAXEU)
        self.lb_CO = 0
        self.ub_CO = self.M * (self.MAXFT-1) * self.MAXEU
        # AM varia de 1 a MAXAM - 1
        self.lb_AM = 1
        self.ub_AM = self.MAXAM - 1
        # SP varia de 0 a 1
        self.lb_SP = 0
        self.ub_SP = 1
        # PE varia de 0 a MAXPE
        self.lb_PE = 0
        self.ub_PE = self.MAXPE

        # VA varia de 0 a 1
        self.lb_VA = 0
        self.ub_VA = 1
        # SU varia de 0 a 1
        self.lb_SU = 0
        self.ub_SU = 1
        self.lb_F = 1
        self.ub_F = self.M

        
        self.lb_real_LT = 0
        self.ub_real_LT = self.ub_LT + self.MAX_ATRASO
        self.lb_atraso_real = 0
        self.ub_atraso_real = self.MAX_ATRASO
        self.lb_action = 0
        self.ub_action = self.MAX_ATRASO
        self.lb_err = 0
        self.ub_err = self.MAX_ATRASO

        # MT varia de 0 a MAXFT
        self.lb_MT = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_MT = np.array([MAXFT-1 for _ in range(self.M)]).astype(np.int64)
        # EU varia de 0 a MAXEU
        self.lb_EU = np.array([0 for _ in range(self.M)]).astype(np.float64)
        self.ub_EU = np.array([self.MAXEU for _ in range(self.M)]).astype(np.float64)
        # BA varia de 0 a MAXFT
        self.lb_BA = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_BA = np.array([100*self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)
        # IN varia de 0 a MAXFT
        self.lb_IN = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_IN = np.array([self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)
        # OU varia de 0 a MAXFT
        self.lb_OU = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_OU = np.array([self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)

        # OU varia de 0 a MAXFT
        self.lb_FT = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_FT = np.array([self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)

        # yard varia de 1 a self.Y
        self.lb_yard = 0
        self.ub_yard = self.Y

        # yard varia de 1 a self.Y
        self.lb_load = 0
        self.ub_load = 100



        # lb e ub--- segunda versao (sem a coluna com os valores de Somn.time)
        # self.lb = np.array([[self.lb_ST, self.lb_LT, self.lb_DO, self.lb_TP] for _ in range(self.N)])
        # self.ub = np.array([[self.ub_ST, self.ub_LT, self.ub_DO, self.ub_TP] for _ in range(self.N)])

        # # lb e ub--- primeira versao (com a coluna com os valores de Somn.time)
        # self.lb = np.array([[self.lb_ST, self.lb_time, self.lb_LT, self.lb_DO, self.lb_TP] for _ in range(self.N)])
        # self.ub = np.array([[self.ub_ST, self.ub_time, self.ub_LT, self.ub_DO, self.ub_TP] for _ in range(self.N)])



        ######################
        #      Espacos       #
        ######################
        """
        Precisa mudar o espaco de acao
        de acordo com o algoritmo utilizado
        """

        # accept to produce or reject
        # self.action_space = spaces.Box(0, 4, shape=(1,)) # usar o TD3
        #self.action_space = spaces.Discrete(self.MAXDO)  # usar com o PPO, DQN, A2C
        self.action_spaces = {f"{i}" : spaces.Discrete(self.MAX_ATRASO) for i in range(numAgents)}

        self.observation_spaces = spaces.Dict(
            {
                "time": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "MT": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "EU": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "BA": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "IN": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "OU": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "DE_state": spaces.Box(
                    low=0.0, high=1.0, shape=(self.N, 17), dtype=np.float64
                ),
                "FT_state": spaces.Box(low=0.0, high=1.0, shape=(self.N,self.M), dtype=np.float64),
                "yard": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "load": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
            }
        )  # versao para MultiInputPolicy Normalizada

        self.observation_spaces = spaces.Dict({f"{i}" : self.observation_spaces for i in range(numAgents)})

        self.stepnum = 0
    ######################
    #      funcoes       #
    ######################

    # recebe um atributo por exemplo 'LT' ou 'real_LT' e devolve os valores
    # de Lead Time das Demandas ou Real Lead Time, conforme o atributo passado.
    def get_Demands_Attr(env, atributo):
        d = [getattr(demandas, atributo) for demandas in env.DE]
        return d
    def get_atraso(self):
        a = self.atraso
        return a
    def get_reward(self):
        rw = self.reward
        return rw
    def get_lucro(self):
        lu = self.lucro
        return lu
    def get_variabilidade(self):
        va = self.variabilidade
        return va
    def get_sustentabilidade(self):
        su = self.sustentabilidade
        return su

    # Normaliza o valor dentro do range passado como parametro
    def normaliza(self, x, min, max):
        # verificar se eh um escalar ou um np.array
        # se for um escalar evitar a divisao por zero.
        if type(x).__module__ != np.__name__:
            if max == min: return 1
        x_norm = (x - min) / (max - min)
        x_norm = np.clip(x_norm, 0.0, 1.0).astype(np.float64)
        return x_norm

    def readDemand(self, agent, demand: Demand = None):
        if demand == None:
            for i in range(len(self.DE[agent])):
                if (self.DE[agent][i].ST == -1):  # free(-1)
                    self.DE[agent][i](Somn.time[agent], self.statistcs[agent].cont, self.statistcs[agent].load)
                    self.match[agent][i] = 0
                    self.DE[agent][i].rejects = []
                    self.DE[agent][i].posDemand = -1
     
        else:
            if demand.ST == -1:
                demand(Somn.time[agent], self.statistcs[agent].cont, self.statistcs[agent].load)
                self.match[agent][demand.posDemand] = 0
    
    def match_demand_with_inventory(self, agent, demand: Demand = None) -> bool:
        if demand == None:
            for i in range(len(self.DE[agent])):
                if self.DE[agent][i].ST == 0: ## SÓ PODE DAR MATCH DEMANDAS CHEGADAS
                    if self.Y > 0: # ALTERAÇÃO PARA TESTE DE YARD == 0
                        # idx é a posição se mask_FT deu match no Yard
                        # -1 se não tiver dado match com nada no Yard
                        idx = self.YA[agent].inYard(self.DE[agent][i].mask_FT)
                        if idx >= 0:
                            self.YA[agent].remove_yard(idx)
                            # adiciona a recompensa
                            # tx_ambiente = self.DE[agent][i].err
                            self.rw_pr += self.DE[agent][i].AM * self.DE[agent][i].PR
                            # - self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.1
                            self.rw_va += self.DE[agent][i].AM * self.DE[agent][i].PR * self.DE[agent][i].VA
                            # - self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.1
                            self.rw_su += self.DE[agent][i].AM * self.DE[agent][i].PR * self.DE[agent][i].SU
                            # - self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.1
                            # libera o espaço i para entrar outra demanda
                            self.DE[agent][i].ST = -1
                            self.match[agent][i] = 0

        else:
            if demand.ST == 0:
                if self.Y > 0:
                    idx = self.YA[agent].inYard(demand.mask_FT)
                    if idx >= 0:
                        self.YA[agent].remove_yard(idx)
                        self.rw_pr += demand.AM * demand.PR
                        self.rw_va += demand.AM * demand.PR * demand.VA
                        self.rw_su += demand.AM * demand.PR * demand.SU

                        demand.ST = -1
                        self.match[agent][demand.posDemand] = 0

    def stock_covers_demand(self, agent, demand: Demand = None):
        covered = True
        if demand == None:

            for i in range(len(self.DE[agent])):
                if self.DE[agent][i].ST == 0: # status RECEIVED
                    DF = self.BA[agent] - self.DE[agent][i].FT
                    OR = np.array(
                        [abs(i) if i < 0 else 0 for i in DF]
                    )  # O QUE PRECISA SER COMPRADO
                    # print('\n ORDER from ', DF, ':', OR)
                    if not np.any(OR):
                        self.DE[agent][i].ST = 1
                        # fila de prioridade 0 = price
                        prio_lucro = 1/(self.DE[agent][i].AM * self.DE[agent][i].PR)
                        # fila de prioridade 1 = variabilidade
                        prio_variabilidade = 1 - self.DE[agent][i].VA
                        # fila de prioridade 2 = sustentabilidade
                        prio_sustentabilidade = 1 - self.DE[agent][i].SU

                        objetivo_agente = Somn.objetivo[agent]

                        prioridades_atualizadas = {
                            0: prio_lucro,
                            1: prio_variabilidade,
                            2: prio_sustentabilidade
                        }

                        Somn.priorq[agent][objetivo_agente][i] = prioridades_atualizadas[objetivo_agente]

                        # print ((1 - self.DE[agent][i].SU))
                        self.BA[agent] -= np.array(DF)  # ATUALIZA O SALDO
                        self.OU[agent] += np.array(DF)  # ATUALIZA A SAÍDA
                        # print('\n balance:', self.BA,  'because not buying',self.OU)
                        self.match[agent][i] = 1
                    else:
                        covered = False
                        self.IN[agent] += np.array(OR)  # ATUALIZA O TOTAL DE COMPRAVEIScl
                        self.match[agent][i] = 0
                        # print('\n balance: ', self.BA, 'because buying',OR, 'accumulating', self.IN)
            
        else:
            if demand.ST == 0:
                DF = self.BA[agent] - demand.FT
                OR = np.array(
                    [abs(i) if i<0 else 0 for i in DF]
                )

                if not np.any(OR):
                    demand.ST = 1
                    prio_lucro = 1/(demand.AM * demand.PR)
                    prio_variabilidade = 1 - demand.VA
                    prio_sustentabilidade = 1 - demand.SU 
                    objetivo_agente = Somn.objetivo[agent]

                    prioridades_atualizadas = {
                        0: prio_lucro,
                        1: prio_variabilidade,
                        2: prio_sustentabilidade
                    }

                    pos_final = demand.posDemand
                    Somn.priorq[agent][objetivo_agente][pos_final] = prioridades_atualizadas[objetivo_agente]

                    # print ((1 - self.DE[agent][i].SU))
                    self.BA[agent] -= np.array(DF)  # ATUALIZA O SALDO
                    self.OU[agent] += np.array(DF)  # ATUALIZA A SAÍDA

                    self.match[agent][demand.posDemand] = 1
                else:
                    covered = False
                    self.IN[agent] += np.array(OR)
                    self.match[agent][demand.posDemand] = 0

        return covered

    def order_receive_and_match(self, agent, demand: Demand = None):
        if demand == None:
            covered = False
            # receive RAW MATERIAL AND ORDERS (DEMANDS)
            self.MT[agent] = np.array([random.randint(0, i) if i > 0 else 0 for i in self.IN[agent]])
            self.readDemand(agent)
            # IF PREVIOUS ORDERS INVENTORY AVAILABLE, PLEASE DISPATCH
            self.match_demand_with_inventory(agent)

            # ANYWAY, UPDATE BALANCE AND INCOME RAW MATERIAL REGARDING MT RECEIVED
            self.IN[agent] -= self.MT[agent]
            self.BA[agent] += self.MT[agent]
            # IF RAW MATERIAL INVENTORY DOES NOT COVER PLEASE REQUEST RAW MATERIAL
            if not self.stock_covers_demand(agent):
                self.IN[agent] = np.array(
                    [random.randint(0, i) if i > 0 else 0 for i in self.IN[agent]]
                ).astype(np.int64)
            if self.match[agent].all():
                covered = True

        else:
            covered = False
            self.MT[agent] = np.array([random.randint(0,i) if i > 0 else 0 for i in self.IN[agent]])
            self.readDemand(agent, demand)
            self.match_demand_with_inventory(agent)

            self.IN[agent] -= self.MT[agent]
            self.BA[agent] += self.MT[agent]

            if not self.stock_covers_demand(agent): 
                self.IN[agent] = np.array(
                    [random.randint(0, i) if i > 0 else 0 for i in self.IN[agent]]
                ).astype(np.int64)
            if self.match[agent][demand.posDemand] == 1:
                covered = True
        return covered
    
    def plan(self, t: int, action, agent) -> int:
        """
        Avalia se a Demanda, de acordo com a ordem de prioridade, vai ser produzida ou rejeitada.\n
        Retorna o valor da posição no vetor de demandas a demanda que foi analizada
        """

        if len(Somn.priorq[agent][Somn.objetivo[agent]]) > 0:
            # objetivo {0: price, 1: variabilidade, 2: sustentabilidade}
            obj = Somn.priorq[agent][Somn.objetivo[agent]].popitem()
            i = obj[0]
            if i >= 0:
                if self.DE[agent][i].ST == 1:  ## DE[I].ST VAI SER SEMPRE 1 PORQUE VEM DA FILAP
                    
                    # COPY JOB TO JOBSHOP SCHEDULING
                    # for j in range(self.M):
                    #     if self.DE[agent][i].FT[j]!= 0:
                    #         Somn.instance.InsertJobs(i, j, self.DE[agent][i].FT[j])
                    #         flag = 1

                    # salva o valor do patio depois da acao
                    self.patio_on_state_plan.append((self.YA[agent].cont/self.YA[agent].Y)*100)
                    # salva o valor da carga depois da acao
                    self.carga_on_state_plan.append(sum([self.DE[agent][i].ST == 3 for i in range(len(self.DE[agent]))]))
                    # salva a acao
                    self.DE[agent][i].action = action
                    self.acao_on_state_plan.append(action)

                    # executa a acao
                    if self.DE[agent][i].DO > (t + self.DE[agent][i].LT + action):
                        self.DE[agent][i].ST = 3  ## produced status --- remember to run time for each case
                        self.OU[agent] += self.DE[agent][i].FT  ## CONSOME OS RECURSOS
                        self.statistcs[agent].load = self.statistcs[agent].load + 1
                        self.DE[agent][i].real_LT = poisson.rvs(mu=(self.DE[agent][i].LT+self.statistcs[agent].load)) # by_frederic
                        self.DE[agent][i].TP = t + self.DE[agent][i].real_LT
                        self.DE[agent][i].atraso_real = abs(self.DE[agent][i].real_LT - self.DE[agent][i].LT)
                        self.DE[agent][i].err = abs(self.DE[agent][i].action - self.DE[agent][i].atraso_real)
                        
                        self.acoes.append(self.DE[agent][i].action)
                        self.atrasos_reais.append(self.DE[agent][i].atraso_real)
                        self.variabilidade.append(self.DE[agent][i].VA)
                        self.sustentabilidade.append(self.DE[agent][i].SU)
                        self.F.append(self.DE[agent][i].F)
                    else:
                        self.DE[agent][i].ST = 2  ## rejected status
                        self.OU[agent] -= self.DE[agent][i].FT  ### libera do buffer de produção
                        self.BA[agent] += self.DE[agent][i].FT  ## devolve para o saldo para os próximos
                        self.statistcs[agent].reject = self.statistcs[agent].reject + 1
                        # se a demanda tivesse sido produzida, teria tido esse real_LT, TP, atraso_real e err abaixo
                        # valores calculados só para salvar no log e avaliar o modelo
                        self.DE[agent][i].real_LT = poisson.rvs(mu=(self.DE[agent][i].LT + self.statistcs[agent].load))
                        self.DE[agent][i].TP = t + self.DE[agent][i].real_LT
                        self.DE[agent][i].atraso_real = abs(self.DE[agent][i].real_LT - self.DE[agent][i].LT)
                        self.DE[agent][i].err = abs(self.DE[agent][i].action - self.DE[agent][i].atraso_real)
                        # self.atrasos_reais.append(self.DE[agent][i].atraso_real)

        # se formou buffer, resolve para comparar depois
        # if flag == 1:
        #   Somn.instance.BuildModel()
        #   Somn.instance.Solve()
        #   Somn.instance.Output()  ## precisa salvar a lista de resultados
    
    def destine(self, agents: list, demand: Demand) -> int:
        """
            Define a qual agente certa demanda vai ser destinada com base no calculo da fila de prioridade\n
            A politica de decisão é feita da seguinte maneira:
            - 
            - Verifica se agentes tem um objetivo que prioriza melhor essa demanda a ele (o seu maior valor de prioridade de acordo com o objetivo do agente)
            - Escolhe o agente com a menor fila de prioridade, assim não inflando apenas um agente com essas demandas rejeitadas
            - Verifica se o agente tem um espaço de demanda livre para que ela seja acolhida
            - Se as opções assima não forem satisfeitas retorna o valor de -1 indicando que não tem agente presente para ficar com essa demanda
        """
        agentes = agents
        obj = []
        for i in agents:
            obj.append(self.objetivo[i])

        prio_lucro = 1/(demand.AM * demand.PR)
        prio_variabilidade = 1 - demand.VA
        prio_sustentabilidade = 1 - demand.SU 

        valores = [prio_lucro, prio_variabilidade, prio_sustentabilidade ]

        for k in range(len(valores)):
            maximo = max(valores)
            qtd = {}
            for i in agentes:
                if self.objetivo[i] == valores.index(maximo):
                    qtd[i] = len(Somn.priorq[i][self.objetivo[i]])
            
            while len(qtd)>0:
                minimo = min(qtd, key=qtd.get)

                for j in self.DE[minimo]:
                    if j.ST == -1:
                        return minimo
                    
                del qtd[minimo]

            index = valores.index(maximo)
            valores[index] = -1
        
        return -1

    
    def rejected(self, demand: Demand):
        """
        Demandas rejeitadas anteriormente podem ser aceitas por outros agentes assim entrando em produção e evitando perca de demandas!\n
        FUNCIONALIDADE:
            - Ao final de cada interação a uma distribuição das encomendas rejeitadas por um agente para os demais
            - Ao entrar em um novo destino essa demanda tem todo o tratamento inical de pedir materia prima e verificar se já existe no pátio
            logo depois ela entra na fila de prioridade do seu novo agente para que ocorra o tratamento novamente
            - Há a possibilidade dessa nova demanda ser rejeitada novamente, ao ser rejeitada por todos os agentes será somado +1 em reject_all
            - Por questões de implementações anteriores se for rejeitada por todos ela será apenas tratada novamente e não removida do vetor de demandas
        """
        demand.ST = -1
        alocado = False
        agents = demand.rejects
        # self.DE[agent].remove(demand)
        if len(demand.rejects) < self.num_agents:
            possibles = [i for i in range(self.num_agents) if i not in agents]
            # novoAgente = random.choice(possibles)
            novoAgente = self.destine(possibles, demand)

            if novoAgente != -1:
                for num, i in enumerate(self.DE[novoAgente]):
                    if i.ST == -1:
                        self.DE[novoAgente][num] = demand
                        self.aux.remove(demand)
                        self.match[novoAgente][num] = 0
                        self.DE[novoAgente][num].posDemand = num
                        alocado = True
                        break

                if alocado:
                    covered = False
                    while not covered:
                        covered = self.order_receive_and_match(novoAgente, demand)
        else:
            self.demands_rejects_all += 1
            self.aux.remove(demand) #ALTERAÇÃO VÁLIDA APENAS PARA OS DOIS ÚLTIMOS TREINAMENTOS DE 6 AGENTES COM COMPARAÇÃO, PEGAR O GRÁFICO DA MÉDIA DESSES DOIS AGENTES
        

    def produce(self, t: int, i: int, agent):
        
        if self.DE[agent][i].ST == 3:
            if self.DE[agent][i].TP < t:  ### TP eh resultado de LT(#f) + RAND
                if self.DE[agent][i].rejects != []:
                    self.acept_reject[agent] += 1
                #Somn.producing = Somn.producing - 1
                self.statistcs[agent].load = self.statistcs[agent].load - 1
                if t < self.DE[agent][i].DO:
                    self.DE[agent][i].ST = 5  ## produced status --- remember to run time for each case
                    # print("\n Destination: Enviou", Yard.cont)
                    if self.DE[agent][i].rejects != []:
                        self.total_acepts_produce[agent] += 1
                    
                else:
                    self.DE[agent][i].ST = 4  ## stored status

                    if self.DE[agent][i].rejects != []:
                        self.total_acepts_yard[agent] += 1
                    
                    # VALIDAÇÃO DE TESTE PARA YARD = 0 #####################################################
                    if self.Y == 0:
                        self.DE[agent][i].ST = -2  ## NAO CABE ... PRODUCAO COM GERAÇÃO DE LIXO (CASO MAIS GRAVE)
                        # production with waste
                        self.statistcs[agent].production_w_waste = self.statistcs[agent].production_w_waste + 1

                    elif self.YA[agent].cont < self.YA[agent].Y:
                        self.YA[agent].yard.append(self.DE[agent][i].FT)

                        mask_YA = self.DE[agent][i].FT.copy()
                        mask_YA[mask_YA > 0] = 1

                        self.YA[agent].mask_YA.append(mask_YA)
                        self.YA[agent].cont = len(self.YA[agent].yard)
                        
                    else:
                        self.DE[agent][i].ST = -2  ## NAO CABE ... PRODUCAO COM GERAÇÃO DE LIXO (CASO MAIS GRAVE)
                        # production with waste
                        self.statistcs[agent].production_w_waste = self.statistcs[agent].production_w_waste + 1

    def dispatch(self, i: int, agent):
             
        if self.DE[agent][i].ST == 5:
            
            self.rw_pr += self.DE[agent][i].AM * self.DE[agent][i].PR
            self.rw_va += self.DE[agent][i].AM * self.DE[agent][i].PR * self.DE[agent][i].VA
            self.rw_su += self.DE[agent][i].AM * self.DE[agent][i].PR * self.DE[agent][i].SU

            self.DE[agent][i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
            self.match[agent][i] = 0

    def store(self, i: int, agent):
        if self.DE[agent][i].ST == 4:
            self.totPenalty += (self.YA[agent].cont/self.YA[agent].space) * self.DE[agent][i].AM * self.DE[agent][i].CO
            tx_ambiente = self.DE[agent][i].err
            self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.01
            self.totPenalty2 += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.01
                
            self.DE[agent][i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
            self.match[agent][i] = 0
            
    def reject(self, i: int, agent):
        if self.DE[agent][i].ST == 2:
            self.totPenalty += 0
            tx_ambiente = self.DE[agent][i].err
            self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.01
            self.totPenalty2 += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.005
            
            self.DE[agent][i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
            self.match[agent][i] = 0
            
            self.DE[agent][i].rejects.append(agent)
            demandCopy = copy.deepcopy(self.DE[agent][i])
            self.rejecteds.append(demandCopy)
                
    def reject_w_waste(self, i: int, agent):
        if self.DE[agent][i].ST == -2:
            self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].CO         # PENALIDADE PELO DESCARTE
            tx_ambiente = self.DE[agent][i].err
            self.totPenalty += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.1
            self.totPenalty2 += self.DE[agent][i].AM * self.DE[agent][i].PR * tx_ambiente * 0.1
            
            self.DE[agent][i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
            self.match[agent][i] = 0
    
    def atualiza_upper_bounds(self, agent):
        # Atualiza o upper bounds
        if np.amax(self.ub_MT) <= np.amax(self.MT[agent]):
            self.ub_MT = np.full(self.M, np.amax(self.MT[agent])) 

        if np.amax(self.ub_BA) <= np.amax(self.BA[agent]):
            self.ub_BA = np.full(self.M, np.amax(self.BA[agent]))

        if np.amax(self.ub_IN) <= np.amax(self.IN[agent]):
            self.ub_IN = np.full(self.M, np.amax(self.IN[agent]))
        
        if np.amax(self.ub_OU) <= np.amax(self.OU[agent]):
            self.ub_OU = np.full(self.M, np.amax(self.OU[agent]))
    
    def observa_demanda(self, agent):
        DE_arrayState = []
        FT_arrayState = []

        for i in range(len(self.DE[agent])):
            aux_row = [
                self.normaliza(x=self.DE[agent][i].ST, min=self.lb_ST, max=self.ub_ST),

                self.normaliza(x=self.DE[agent][i].DI, min=self.lb_DI, max=self.ub_DI),
                self.normaliza(x=self.DE[agent][i].DO, min=self.lb_DO, max=self.ub_DO),
                self.normaliza(x=self.DE[agent][i].TP, min=self.lb_TP, max=self.ub_TP),

                self.normaliza(x=self.DE[agent][i].PR, min=self.lb_PR, max=self.ub_PR),
                self.normaliza(x=self.DE[agent][i].CO, min=self.lb_CO, max=self.ub_CO),
                self.normaliza(x=self.DE[agent][i].AM, min=self.lb_AM, max=self.ub_AM),
                self.normaliza(x=self.DE[agent][i].SP, min=self.lb_SP, max=self.ub_SP),
                self.normaliza(x=self.DE[agent][i].PE, min=self.lb_PE, max=self.ub_PE),

                self.normaliza(x=self.DE[agent][i].VA, min=self.lb_VA, max=self.ub_VA),
                self.normaliza(x=self.DE[agent][i].SU, min=self.lb_SU, max=self.ub_SU),
                self.normaliza(x=self.DE[agent][i].F, min=self.lb_F, max=self.ub_F), 
                
                self.normaliza(x=self.DE[agent][i].LT, min=self.lb_LT, max=self.ub_LT),
                self.normaliza(x=self.DE[agent][i].real_LT, min=self.lb_real_LT, max=self.ub_real_LT), 
                self.normaliza(x=self.DE[agent][i].atraso_real, min=self.lb_atraso_real, max=self.ub_atraso_real), 
                self.normaliza(x=self.DE[agent][i].action, min=self.lb_action, max=self.ub_action), 
                self.normaliza(x=self.DE[agent][i].err, min=self.lb_err, max=self.ub_err), 

            ]
            DE_arrayState.append(aux_row)
        for i in range(len(self.DE[agent])):
            aux_FT = self.normaliza(x=self.DE[agent][i].FT, min=self.lb_FT, max=self.ub_FT)
            FT_arrayState.append(aux_FT)
        
        self.DE_state = np.array(DE_arrayState)
        self.FT_state = np.array(FT_arrayState)

        return self.DE_state, self.FT_state

    ######################
    #       step         #
    ######################
    def step(self, actions: dict):
        """
        Atualiza tudo aqui e devolve o próximo estado: n_state, reward, done, info

            - n_state: próximo estado;
            - reward: recompensa da ação;
            - done: flag de conclusão;
            - info: informaões extras (opcional)

        Primeira versão vai fazer uma iteração para cada episódio ...
        O Tempo t precisa ser controlado
        """
        self.stepnum += 1

        info = {f"{i}" : {} for i in range(len(self.possible_agents))}
        observation = {f"{i}" : {} for i in range(len(self.possible_agents))}
        done = {f"{i}": {} for i in range(len(self.possible_agents))}
        truncated = {f"{i}" : {} for i in range(len(self.possible_agents))}

        for agent in actions:

            action = actions[agent]
            agent = int(agent)

            self.totReward = 0.0
            self.totPenalty = 0.0
            self.totPenalty2 = 0.0

            self.rw_pr = 0.0                 
            self.rw_va = 0.0
            self.rw_su = 0.0

            self.variabilidade = []
            self.sustentabilidade = []
            self.F = []

            self.acoes = []
            self.atrasos_reais = []

            self.acao_on_state_plan = []
            self.carga_on_state_plan = []
            self.patio_on_state_plan = []


            # se a fila de prioridade estiver vazia 
            # entra em order_receive_and_match() senao pula para plan()
            if len(Somn.priorq[agent][Somn.objetivo[agent]]) == 0:
                covered = False
                while not covered:
                    covered = self.order_receive_and_match(agent) 
                    
            self.plan(Somn.time[agent], action, agent)
            for i in range(len(self.DE[agent])):
                self.produce(Somn.time[agent], i, agent)
                self.dispatch(i, agent)
                self.store(i, agent)
                self.reject(i, agent)
                self.reject_w_waste(i, agent)  

            if Somn.objetivo[agent] == 0: # lucro
                self.totReward = self.rw_pr
                self.reward[f"{agent}"] = self.totReward - self.totPenalty
                self.penalty[agent] = self.totPenalty
            if Somn.objetivo[agent] == 1: # variabilidade
                self.totReward = self.rw_va
                self.reward[f"{agent}"] = self.totReward - self.totPenalty2
                self.penalty[agent] = self.totPenalty2
            if Somn.objetivo[agent] == 2: # sustentabilidade
                self.totReward = self.rw_su
                self.reward[f"{agent}"] = self.totReward - self.totPenalty2
                self.penalty[agent] = self.totPenalty2
            
            # desconta as penalidades
            self.rw_pr -= self.totPenalty
            self.rw_va -= self.totPenalty2
            self.rw_su -= self.totPenalty2


            
            """
            - avalia os estados finais
            (
                reward,             # recompensa calculada com a penalidade aplicada
                penalty,            # penalidade que foi aplicada
                rw_pr,              # recompensa para lucro
                rw_va,              # recompensa para a variabilidade
                rw_su,              # recompensa para a sustentabilidade
                variabilidade,      # variabilidade de 0 a 1
                sustentabilidade,   # sustentabilidade de 0 a 1
                F,                  # numero de features utilizadas (numero de maquinas)
                acoes,              # acoes no estado ready que geraram os estados finais contabilizados
                atrasos_reais       # atrasos reais para compararar com as acoes
            ) = self.eval_final_states()  # aqui vai a função que calcula a recompensa

            logs pontuais Yard e Penalidade
            """

            # condição de parada
            done[f"{agent}"] = False
            truncated[f"{agent}"] = False
            if Somn.time[agent] >= self.ub_time:  # 10*Demand.MAXDO + Demand.M   (TEMPOMAX)
                self.agents.remove(f"{agent}")
                done[f"{agent}"] = True

            # atualiza o upper bounds de MT, BA, IN e OU
            self.atualiza_upper_bounds(agent)
        
            # Informações adicionais
            
            info[f"{agent}"] = {"rw": self.reward[f"{agent}"],
                    "rw_pr": self.rw_pr,
                    "rw_va": self.rw_va,
                    "rw_su": self.rw_su,
                    "VA": self.variabilidade,
                    "SU": self.sustentabilidade,
                    "F": self.F,
                    "acoes": action,
                    "reject": self.statistcs[agent].reject,
                    "reject_w_west": self.statistcs[agent].production_w_waste,
                    "atrasos_reais": self.atrasos_reais,
                    "acao_on_state_plan": self.acao_on_state_plan,
                    "carga_on_state_plan": self.carga_on_state_plan,
                    "patio_on_state_plan": self.patio_on_state_plan,
                    "yard" : (self.YA[agent].cont/self.YA[agent].Y)*100,
                    "reject_all": self.demands_rejects_all,
                    "acept_reject": self.acept_reject[agent],
                    "produced_reject": self.total_acepts_produce[agent],
                    "yard_reject": self.total_acepts_yard[agent]
                    }  
            
            # observação
            self.DE_state, self.FT_state = self.observa_demanda(agent)
            observation[f"{agent}"] = {
                "time": np.array([self.normaliza(self.time[agent], self.lb_time, self.ub_time)]),
                "MT": self.normaliza(self.MT[agent], self.lb_MT, self.ub_MT),
                "EU": self.normaliza(self.EU[agent], self.lb_EU, self.ub_EU),
                "BA": self.normaliza(self.BA[agent], self.lb_BA, self.ub_BA),
                "IN": self.normaliza(self.IN[agent], self.lb_IN, self.ub_IN),
                "OU": self.normaliza(self.OU[agent], self.lb_OU, self.ub_OU),
                "DE_state": self.DE_state,
                "FT_state": self.FT_state,
                "yard": np.array([self.normaliza(self.YA[agent].cont, self.lb_yard, self.ub_yard)]),
                "load": np.array([self.normaliza(self.statistcs[agent].load, self.lb_load, self.ub_load)]),

            }  # by_frederic: retorna quando e um tipo Dict

            # se não tiver mais demandas na fila de prioridade atualiza o tempo
            #if len(Somn.priorq[Somn.objetivo]) == 0:
            Somn.time[agent] += 1

        if observation == {}:
            print("\n\n\nVAZIOOOOOOOOOOOOOOOOOO\n\n\n")
        
        self.aux = self.rejecteds
        for i in self.rejecteds:
            self.rejected(i)

        self.rejecteds = self.aux

        return (
            observation,
            self.reward,
            done,
            truncated,
            info,
        )  # , exprofit   # by_frederic:

    ######################
    #       reset        #
    ######################

    def reset(self, *, seed=None, options=None):
        #super().reset(seed=None)

        self.agents = self.possible_agents[:]
        
        Somn.priorq = {agente: [heapdict() for _ in Somn.obj_list] for agente in range(self.num_agents)}
        # Somn.priorqsu = heapdict()
        # Somn.priorqva = heapdict()

        self.match = []  

        self.EU = {agent : np.random.random(self.M) * self.MAXEU for agent in range(self.num_agents)}
        self.BA = {agent : np.random.randint(10, 10*self.MAXFT, self.M) for agent in range(self.num_agents)}
        self.IN = {agent : np.random.randint(0, self.MAXFT, self.M) for agent in range(self.num_agents)}
        self.OU = {agent : np.random.randint(0, self.MAXFT, self.M) for agent in range(self.num_agents)}
        
        #LOGS PONTUAIS
        # wandb.log({
        #     'reject_w_waste Somn' : self.statistcs[agent].reject_w_waste
        # })

        self.reward = {}
        self.penalty = {}
        self.totReward = 0.0
        self.totPenalty = 0.0
        self.totPenalty2 = 0.0
        
        self.acao_on_state_plan = []
        self.carga_on_state_plan = []
        self.patio_on_state_plan = []

        self.YA = []

        self.DE = []

        self.acept_reject = []

        self.total_acepts_produce = []
        self.total_acepts_yard = []

        for agent in range(len(self.agents)):
            agentDemands = [
                Demand(
                    self.M, self.N, self.MAXDO, self.MAXAM, self.MAXPR, self.MAXPE, self.MAXFT, self.MAXMT, self.MAXTI, self.MAXEU, Somn.time[agent], self.atraso
                )
                for _ in range(self.N)
            ]

            self.match.append(np.zeros(self.N))
            self.MT[agent] = np.random.randint(0, self.MAXFT, self.M)
            Somn.time[agent] = 1

            self.rejecteds = []
            self.demands_rejects_all = 0

            self.DE.append(agentDemands)
            self.YA.append(Yard(self.Y))
            self.statistcs[agent].load = 0
            self.statistcs[agent].reject = 0
            self.statistcs[agent].production_w_waste=0

            self.acept_reject.append(0)
            self.total_acepts_yard.append(0)
            self.total_acepts_produce.append(0)

            for i in range(self.N):
                self.DE[agent][i](Somn.time[agent], self.statistcs[agent].cont, self.statistcs[agent].load)

            self.DE_state, self.FT_state = self.observa_demanda(agent)
        
        #############################################################################################################################    
        info = {f"{i}" : {} for i in range(self.num_agents)}
        # observation = (self.DE_state, info)  # by_frederic: retorna quando o tipo é Box

        observationParcial = {}
        for agent in range(len(self.agents)):
            observation = {
                "time": np.array([self.normaliza(self.time[agent], self.lb_time, self.ub_time)]),
                "MT": self.normaliza(self.MT[agent], self.lb_MT, self.ub_MT),
                "EU": self.normaliza(self.EU[agent], self.lb_EU, self.ub_EU),
                "BA": self.normaliza(self.BA[agent], self.lb_BA, self.ub_BA),
                "IN": self.normaliza(self.IN[agent], self.lb_IN, self.ub_IN),
                "OU": self.normaliza(self.OU[agent], self.lb_OU, self.ub_OU),
                "DE_state": self.DE_state,
                "FT_state": self.FT_state,
                "yard": np.array([self.normaliza(self.YA[agent].cont, self.lb_yard, self.ub_yard)]),
                "load": np.array([self.normaliza(self.statistcs[agent].load, self.lb_load, self.ub_load)]),
            }  # by_frederic: retorna quando e um tipo Dict

            observationParcial[f"{agent}"] = observation
        ##############################################################################################################################

        return (observationParcial, info)  # by_alysson: para se adequar ao MultAgent

    ######################
    #       render       #
    ######################

    def render(self):
        # print("Current state (RENDER): \n", self.DE_state)
        pass

    ######################
    #       close        #
    ######################

    def close(self):
        pass

    def observation_space(self, agent):
        """Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
    
        return self.action_spaces[agent]