import csv
from datetime import datetime

# Nome único para o arquivo CSV gerado uma vez por execução
csv_filename_filas_escolhidas = f"plots/filas_escolhidas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_filename_filas_prioridades = f"plots/filas_prioridade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def adicionar_dados_filas_escolhidas(step, agente, lucro, variabilidade, sustentabilidade, erroEls, erro_anteriorEls,
                                     erroEsv, erro_anteriorEsv, erroEvl, erro_anteriorEvl, saidaEvl, saidaEls, saidaEsv, fila):
    # Verifica se o arquivo existe e escreve o cabeçalho se necessário
    try:
        with open(csv_filename_filas_escolhidas, "x", newline="") as file:  # "x" para criar o arquivo se não existir
            writer = csv.writer(file)
            writer.writerow(["step", "agente", "lucro", "variabilidade", "sustentabilidade", 
                             "erroEls", "erro_anteriorEls", "erroEsv", "erro_anteriorEsv", 
                             "erroEvl", "erro_anteriorEvl", "saidaEvl", "saidaEls", 
                             "saidaEsv", "fila"])
    except FileExistsError:
        pass  # Arquivo já existe; pula a criação do cabeçalho

    # Adiciona a linha de dados ao arquivo
    with open(csv_filename_filas_escolhidas, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([step, agente, lucro, variabilidade, sustentabilidade, erroEls, erro_anteriorEls,
                         erroEsv, erro_anteriorEsv, erroEvl, erro_anteriorEvl, saidaEvl, saidaEls, saidaEsv, fila])

def adicionar_dados_filas_de_prioridades(step, agent, filaLucro, LU, filaSustentabilidade, SU, filaVariabilidade, VA, fila):
    # Verifica se o arquivo existe e escreve o cabeçalho se necessário
    try:
        with open(csv_filename_filas_prioridades, "x", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "agente", "fila_Lucro", "LU", "fila_Variabilidade", "VA", "fila_Sustentabilidade", "SU", "fila"])
    except FileExistsError:
        pass

    # Adiciona a linha de dados ao arquivo
    with open(csv_filename_filas_prioridades, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([step, agent, filaLucro,LU, filaSustentabilidade,SU, filaVariabilidade,VA, fila])
