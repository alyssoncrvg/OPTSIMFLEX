from Ambiente_SOMN.Somn import Somn

def make_env(atraso: int, numAgents : int, objetivo: dict[int,int]):
    # env = Somn(Y=3,M=10,N=10,MAXDO=10,MAXAM=3,MAXPR=2,MAXPE=10,MAXFT=5,MAXMT=3,MAXTI=2,
    #            MAXEU = 5, atraso=atraso)
    env = Somn(
                Y=0,
                M=10,
                N=10,
                MAXDO=100,
                MAXAM=2,
                MAXPR=2,
                MAXPE=10,
                MAXFT=5,
                MAXMT=3,
                MAXTI=2,
                MAXEU = 5, 
                atraso=atraso,
                numAgents=numAgents,
                objetivo=objetivo
            )
    
    return env