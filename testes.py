inicio_cima_novo = 600
fim_cima_novo = 2300
inicio_baixo_novo = 300
fim_baixo_novo = 1800

# Calculando os valores médios dos novos gráficos
media_cima_novo = (inicio_cima_novo + fim_cima_novo) / 2
media_baixo_novo = (inicio_baixo_novo + fim_baixo_novo) / 2

# Calculando a porcentagem de quanto o gráfico de cima é maior que o de baixo
diferenca_percentual_novo = ((media_cima_novo - media_baixo_novo) / media_baixo_novo) * 100
print(media_cima_novo, media_baixo_novo, diferenca_percentual_novo)

