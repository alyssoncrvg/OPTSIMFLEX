import random
import numpy as np
from scipy.stats import poisson

class Demand:

    # Somn(Y=10,M=10,N=10,MAXDO=10,MAXAM=3,MAXPR=2,MAXPE=10,MAXFT=5,MAXMT=3,MAXTI=2,MAXEU = 5, atraso=atraso)
    def __init__(self,
                 M:int,
                 N:int,
                 MAXDO:int,
                 MAXAM:int,
                 MAXPR:float,
                 MAXPE:int,
                 MAXFT:int,
                 MAXMT:int,
                 MAXTI:int,
                 MAXEU:int,
                 t: int,
                 atraso: int = None):

        Demand.M=M
        Demand.N=N
        Demand.MAXDO=MAXDO
        Demand.MAXAM=MAXAM
        Demand.MAXPR=MAXPR
        Demand.MAXPE=MAXPE
        Demand.MAXFT=MAXFT
        Demand.MAXMT=MAXMT
        Demand.MAXTI=MAXTI
        Demand.MAXEU=MAXEU
        Demand.EU = np.random.random(M)*MAXEU
        self.ST = int(-1)   # reject_w_wast(-2) free(-1) received(0), ready(1), rejected(2), producing(3), stored(4) and delivered(5)
        
        self.action = int(-1)       # acao atribuida quando ST=1 (estado ready)
                                    # -1 significa que nenhuma acao foi atribuida ainda
        self.atraso_real = int(-1)  # atraso real da demanda (real_LT - LT)
        self.err = int(-1)          # diferença entra o atraso real e a acao atribuida
       
        Demand.atraso=atraso

        self.rejects = []

        self.posDemand = -1


    def __call__(self, t:int, cont: int, load: int):

        cont +=1
        self.CU = cont
        self.AM = random.randrange(1,Demand.MAXAM)
        self.PE = random.randint(1,Demand.MAXPE)
        self.ST = int(0)                  ###received0, ready1, rejected2, produced3, stored4 and delivered5
        
        # Escolhe aleatoriamente as features (tempos para cada maquina)
        # Exemplo: self.FT = array([2, 4, 0, 1, 3]) #valores de: 0 a (MAXFT -1)
        #self.FT = np.random.randint(0,Demand.MAXFT,self.M)
        self.F, self.FT, self.mask_FT = self.gera_features()

        self.LT = self.fun_tau()
        self.real_LT = poisson.rvs(mu=(self.LT + load))
        self.TP = t + self.real_LT
        
        self.atraso_real = abs(self.real_LT - self.LT)
        self.action = self.atraso_real                  # action = atraso_real
        self.err = abs(self.action - self.atraso_real)  # err = 0

        self.DI = t
        self.DO = t + self.LT + random.randint(0,Demand.MAXDO)

        self.SP = self.fun_gamma() ####* 'cpu'.Y   #SPACE CONSUMPTION FACTOR
        self.VA = self.fun_upsilon() ### [0low 1up]
        self.SU = 1- self.fun_sigma() ### [0low 1up]
        
        self.CO = 0.0
        for j in range(Demand.M):
            if self.FT[j] != 0:
                # self.CO += ((self.MAXFT - 1) / self.FT[j]) * Demand.EU[j] #/ self.MAXFT * self.MAXEU
                self.CO += self.FT[j] * Demand.EU[j]
            else:
                self.CO += 0

            # self.CO /= self.F
        # sustentabilidade tem um custo maior    
        # self.CO = self.CO * float(self.M/self.F)
        self.CO = self.AM * self.CO #/ self.MAXAM - 1  # custo com o amount
        
        self.PR = self.fun_theta(self.SU + self.VA)  ### LUCRO EH 2X CUSTO  self.PR = Demand.MAXPE  (by fred)
        self.LU = (self.PR * self.AM) / (self.MAXAM * 1.5 * self.M * self.MAXFT)  # LUCRO NORMALIZADO
    
    # def __call__(self, t: int, cont: int, load: int):
    #     cont += 1
    #     self.CU = cont
    #     self.AM = random.uniform(0, Demand.MAXAM)  # Amount normalizado
    #     self.PE = random.uniform(0, Demand.MAXPE)  # Peso normalizado
    #     self.ST = int(0)  # Status inicial

    #     # Escolhe aleatoriamente as features (tempos para cada máquina)
    #     self.F, self.FT, self.mask_FT = self.gera_features()

    #     self.LT = self.fun_tau()
    #     self.real_LT = poisson.rvs(mu=(self.LT + load))
    #     self.TP = t + self.real_LT

    #     self.atraso_real = abs(self.real_LT - self.LT)
    #     self.action = self.atraso_real
    #     self.err = abs(self.action - self.atraso_real)

    #     self.DI = t
    #     self.DO = t + self.LT + random.randint(0, Demand.MAXDO)
        
    #     self.SP = self.fun_gamma()
        
    #     self.CO = 0.0
    #     for j in range(Demand.M):
    #         if self.FT[j] != 0:
    #             # self.CO += ((self.MAXFT - 1) / self.FT[j]) * Demand.EU[j] #/ self.MAXFT * self.MAXEU
    #             self.CO += self.FT[j] * Demand.EU[j]
    #         else:
    #             self.CO += 0

    #         # self.CO /= self.F
    #     # sustentabilidade tem um custo maior    
    #     # self.CO = self.CO * float(self.M/self.F)
    #     self.CO = self.AM * self.CO #/ self.MAXAM - 1  # custo com o amount

    #     # Sorteio da variável dominante
    #     dominant = random.choice(["VA", "SU", "PR"])
        
    #     if dominant == "VA":
    #         self.VA = random.uniform(0.8, 1.0)  # Valor alto para VA
    #         self.SU = random.uniform(0.0, 0.4)  # Valor baixo para SU
    #         self.PR = random.uniform(0.4, 0.6)  # Valor intermediário para PR
    #     elif dominant == "SU":
    #         self.SU = random.uniform(0.8, 1.0)  # Valor alto para SU
    #         self.VA = random.uniform(0.0, 0.4)  # Valor baixo para VA
    #         self.PR = random.uniform(0.4, 0.6)  # Valor intermediário para PR
    #     else:  # PR dominante
    #         self.PR = random.uniform(0.8, 1.0)  # Valor alto para PR
    #         self.VA = random.uniform(0.4, 0.6)  # Valor intermediário para VA
    #         self.SU = random.uniform(0.0, 0.4)  # Valor baixo para SU


        


    def fun_gamma(self) -> float:
        x = (self.AM*self.F)/((Demand.MAXAM -1) * self.M)
        return x

    def fun_tau(self) -> float:
        x = self.AM * self.FT
        return x.sum()

    def fun_upsilon(self) -> float:
        x = self.F/self.M
        return x

    def fun_sigma(self) -> float:
        x = (self.F * (self.MAXFT - 1) - sum(self.FT)) / (self.F * (self.MAXFT - 1) - self.F)

        return x
    
    def fun_theta(self, fator_lucro):
        min_original = 1/self.M
        max_original = 2
        novo_min = 1
        novo_max = 2

        x = self.CO * self.converter_intervalo(fator_lucro, min_original, max_original, novo_min, novo_max)# * self.MAXPR
        return x
    
    def converter_intervalo(self, valor_original, min_original, max_original, novo_min, novo_max):
        """
        Converte um valor de um intervalo original para um novo intervalo.

        Args:
        valor_original (float): O valor no intervalo original.
        min_original (float): O mínimo do intervalo original.
        max_original (float): O máximo do intervalo original.
        novo_min (float): O mínimo do novo intervalo.
        novo_max (float): O máximo do novo intervalo.

        Returns:
        float: O valor convertido no novo intervalo.
        """
        valor_convertido = novo_min + ((valor_original - min_original) / (max_original - min_original)) * (novo_max - novo_min)
        return valor_convertido
    
    def gera_features(self) -> np.ndarray:

        # joga a moeda para decidir entre variabilidade alta ou baixa
        joga_moeda = bool(random.randint(0,1))
        if joga_moeda:
            F = random.randint(1,int(Demand.M*0.3))
        else:
            F = random.randint(int(Demand.M*0.8),Demand.M)

        posicoes = sorted(random.sample(range(0, Demand.M), F))
        mask = np.zeros(Demand.M).astype(np.int32)
        mask[posicoes] = 1
        
        return F, np.random.randint(1,Demand.MAXFT,Demand.M).astype(np.int32) * mask, mask

    def __repr__(self):
        try:
            # Tenta acessar os atributos que são gerados no método __call__
            return (f"Demand("
                    f"AM={self.AM}, PR={self.PR:.2f}, SU={self.SU:.2f}, VA={self.VA:.2f} "
                    f"CO={self.CO:.2f}, DI={self.DI}, DO={self.DO}, "
                    f"atraso_real={self.atraso_real}, action={self.action}, err={self.err})")
        except AttributeError:
            # Caso o método __call__ ainda não tenha sido executado
            return "Demand instance: attributes not yet initialized. Call the instance with (t, cont, load)."