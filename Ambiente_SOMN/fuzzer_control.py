# Definição dos conjuntos fuzzy e limites
MSS = 0.010 # TAXAS DE MUTAÇÃO --> Intervalos para valores possíveis de mutação
MMS = 0.025
MMM = 0.040
MML = 0.055
MLL = 0.070

HSS = 0.10 # LIMITES DE VALORES DE HERANÇA --> 
HMS = 0.25
HMM = 0.40
HML = 0.55
HLL = 0.70

ESS = 0.0 # LIMITES DE EXIGÊNCIA
EMS = 0.15
EMM = 0.30
EML = 0.45
ELL = 0.60

TSS = 0.005# LIMITES DE TOLERÂNCIA
TMS = 0.010
TMM = 0.050
TML = 0.100
TLL = 0.500

# Rótulos dos conjuntos fuzzy
ENG = -0.10
ENP = -0.05
EZE = 0.00
EPP = 0.05
EPG = 0.10

DNG = -0.10
DNP = -0.05
DZE = 0.0
DPP = 0.05
DPG = 0.10

UNG = -0.50
UNP = -0.10
UZE = 0.0
UPP = 0.10
UPG = 0.50

# Conjuntos fuzzy
SetErro = [ENG, ENP, EZE, EPP, EPG]
SetDeri = [DNG, DNP, DZE, DPP, DPG]
SetAcao = [UNG, UNP, UZE, UPP, UPG]

# Constantes
Ke = 1.0
Kd = 1.0
KuT = 0.15
KuM = 0.1
KuH = 0.1
KuE = 0.1
RefDiv = 1.0 #DIVERS
RefPop = 100 #TOTREC

def triangulo(x, centro, limite):
    return 1 - abs(x - centro) / abs(limite - centro)

def corte(var, ini, fim):
    return max(ini, min(var, fim))

# Controlador fuzzy
def controle(etotal, deriv, e, ld, lde):
    # Base de regras fuzzy 4 é um extremo e 0 o outro
    tabela = [
        [4, 4, 3, 2, 2],
        [4, 3, 3, 2, 2],
        [3, 3, 2, 1, 1],
        [2, 2, 1, 1, 0],
        [2, 2, 1, 0, 0]
    ]
    
    mie = [0, 0]
    mid = [0, 0]
    miu = [0] * 4
    centro = [0] * 4
    num = 0
    den = 0
    er = [0, 0]
    d = [0, 0]
    u = [0] * 4
    a = 0

    # Fuzzificação do erro
    if etotal < e[0]:
        mie[0] = mie[1] = 1
        er[0] = er[1] = 0
    elif e[0] <= etotal < e[1]:
        mie[0] = triangulo(etotal, e[0], e[1])
        mie[1] = triangulo(etotal, e[1], e[0])
        er[0] = 0
        er[1] = 1
    elif e[1] <= etotal < e[2]:
        mie[0] = triangulo(etotal, e[1], e[2])
        mie[1] = triangulo(etotal, e[2], e[1])
        er[0] = 1
        er[1] = 2
    elif e[2] <= etotal < e[3]:
        mie[0] = triangulo(etotal, e[2], e[3])
        mie[1] = triangulo(etotal, e[3], e[2])
        er[0] = 2
        er[1] = 3
    elif e[3] <= etotal < e[4]:
        mie[0] = triangulo(etotal, e[3], e[4])
        mie[1] = triangulo(etotal, e[4], e[3])
        er[0] = 3
        er[1] = 4
    else:
        mie[0] = mie[1] = 1
        er[0] = er[1] = 4

    # Fuzzificação da derivada
    if deriv < ld[0]:
        mid[0] = mid[1] = 1
        d[0] = d[1] = 0
    elif ld[0] <= deriv < ld[1]:
        mid[0] = triangulo(deriv, ld[0], ld[1])
        mid[1] = triangulo(deriv, ld[1], ld[0])
        d[0] = 0
        d[1] = 1
    elif ld[1] <= deriv < ld[2]:
        mid[0] = triangulo(deriv, ld[1], ld[2])
        mid[1] = triangulo(deriv, ld[2], ld[1])
        d[0] = 1
        d[1] = 2
    elif ld[2] <= deriv < ld[3]:
        mid[0] = triangulo(deriv, ld[2], ld[3])
        mid[1] = triangulo(deriv, ld[3], ld[2])
        d[0] = 2
        d[1] = 3
    elif ld[3] <= deriv < ld[4]:
        mid[0] = triangulo(deriv, ld[3], ld[4])
        mid[1] = triangulo(deriv, ld[4], ld[3])
        d[0] = 3
        d[1] = 4
    else:
        mid[0] = mid[1] = 1
        d[0] = d[1] = 4

    # Inferência
    for b in range(2):
        for c in range(2):
            u[a] = tabela[er[b]][d[c]]
            centro[a] = lde[u[a]]
            miu[a] = min(mie[b], mid[c])
            a += 1

    # Defuzzificação
    for t in range(4):
        num += miu[t] * centro[t]
        den += miu[t]

    return num / den


# # Conjuntos de valores normalizados
# cenarios = [
#     {"L": 0.3, "S": 0.5, "V": 1.0},
#     {"L": 0.7, "S": 0.9, "V": 0.3},
#     {"L": 0.6, "S": 0.2, "V": 0.4}
# ]

# # Erros anteriores diferentes para cada discrepância
# erro_anteriorEvl = -0.30  # Erro anterior para Evl
# erro_anteriorEls = 0.85   # Erro anterior para Els
# erro_anteriorEsv = -0.40  # Erro anterior para Esv

# # Diferença de tempo (1 segundo)
# delta_t = 1

# # Executar o teste para cada conjunto de valores
# for i, cenario in enumerate(cenarios, 1):
#     L = cenario["L"]
#     S = cenario["S"]
#     V = cenario["V"]

#     # Cálculo dos erros usando as três equações
#     erro_atualEvl = (V - L) / L  # Evl = (V - L) / L
#     erro_atualEls = (L - S) / S  # Els = (L - S) / S
#     erro_atualEsv = (S - V) / V  # Esv = (S - V) / V

#     # Cálculo das derivadas com erros anteriores distintos
#     derivEvl = (erro_atualEvl - erro_anteriorEvl) / delta_t
#     derivEls = (erro_atualEls - erro_anteriorEls) / delta_t
#     derivEsv = (erro_atualEsv - erro_anteriorEsv) / delta_t

#     # Executar o controlador com o erro calculado e a derivada
#     saidaEvl = controle(erro_atualEvl, derivEvl, SetErro, SetDeri, SetAcao)
#     saidaEls = controle(erro_atualEls, derivEls, SetErro, SetDeri, SetAcao)
#     saidaEsv = controle(erro_atualEsv, derivEsv, SetErro, SetDeri, SetAcao)

#     # Exibir o resultado para cada cenário
#     print(f"\nCenário {i} - L = {L}, S = {S}, V = {V}")
#     print(f"Erros: Evl = {erro_atualEvl:.4f}, Els = {erro_atualEls:.4f}, Esv = {erro_atualEsv:.4f}")
#     print(f"Erros Anteriores: Evl = {erro_anteriorEvl}, Els = {erro_anteriorEls}, Esv = {erro_anteriorEsv}")
#     print(f"Derivadas: Evl = {derivEvl:.4f}, Els = {derivEls:.4f}, Esv = {derivEsv:.4f}")
#     print(f"Saídas: Evl = {saidaEvl:.4f}, Els = {saidaEls:.4f}, Esv = {saidaEsv:.4f}")


# inputs = {
#     "L": [0.1,0.1,0.1,0.1,0.1,0.1,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8],
#     "S": [0.5,0.9,0.9,0.9,0.9,0.9,0.4,0.4,0.4,0.4,0.4,0.8,0.8,0.8,0.8,0.8,0.1,0.1,0.1,0.1,0.1,0.5,0.5,0.5,0.5,0.5,0.9,0.9,0.9,0.9],
#     "V": [0.9,0.2,0.4,0.6,0.8,1,0.2,0.4,0.6,0.8,1,0.3,0.5,0.7,0.9,1,0.1,0.3,0.5,0.7,0.9,0.1,0.3,0.5,0.7,0.9,0.2,0.4,0.6,0.8]
# }

# V = 0
# L = 0
# S = 0

# erro_anteriorEvl = 0  
# erro_anteriorEls = 0   
# erro_anteriorEsv = 0  

# for i in range(len(inputs["L"])):
    
#     erro_atualEls = (inputs["L"][i] - inputs["S"][i])/ inputs["S"][i]
#     erro_atualEsv = (inputs["S"][i] - inputs["V"][i])/ inputs["V"][i]
#     erro_atualEvl = (inputs["V"][i] - inputs["L"][i])/ inputs["L"][i]
    
#     derivEvl = (erro_atualEvl - erro_anteriorEvl)
#     derivEls = (erro_atualEls - erro_anteriorEls)
#     derivEsv = (erro_atualEsv - erro_anteriorEsv)
    
#     saidaEvl = controle(erro_atualEvl, derivEvl, SetErro, SetDeri, SetAcao)
#     saidaEls = controle(erro_atualEls, derivEls, SetErro, SetDeri, SetAcao)
#     saidaEsv = controle(erro_atualEsv, derivEsv, SetErro, SetDeri, SetAcao)
    
#     dl = inputs["L"][i]
#     dv = inputs["V"][i]
#     ds = inputs["S"][i]
    
#     V += dv
#     L += dl
#     S += ds
    
#     saidas = [L, V, S]

#     indice_maximo = max(enumerate(saidas), key=lambda x: x[1])[0]

    
#     # Exibir o resultado para cada cenário
#     print(f"\nCenário {i} - L = {dl}, S = {ds}, V = {dv}")
#     print(f"Valores brutos: L = {L}, V = {V}, S = {S}")
#     print(f"Erros: Evl = {erro_atualEvl:.4f}, Els = {erro_atualEls:.4f}, Esv = {erro_atualEsv:.4f}")
#     print(f"Erros Anteriores: Evl = {erro_anteriorEvl}, Els = {erro_anteriorEls}, Esv = {erro_anteriorEsv}")
#     print(f"Derivadas: Evl = {derivEvl:.4f}, Els = {derivEls:.4f}, Esv = {derivEsv:.4f}")
#     print(f"Saídas: Els = {saidaEls:.4f}, Evl = {saidaEvl:.4f}, Esv = {saidaEsv:.4f}")
#     print(f"Fila escolhida: {indice_maximo}")
#     erro_anteriorEvl = erro_atualEvl
#     erro_anteriorEls = erro_atualEls
#     erro_anteriorEsv = erro_atualEsv