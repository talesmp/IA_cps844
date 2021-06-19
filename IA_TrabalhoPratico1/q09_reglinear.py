# 09) Neste exercıcio iremos avaliar o desempenho da versao pocket do PLA em um conjunto de dados que nao eh linearmente separavel. 
# Para criar este conjunto, gere um ruido simulado selecionando aleatoriamente 10% do conjunto de treinamento (de tamanho N1) 
# e inverta os rotulos dos pontos selecionados. 
# Em seguida, implemente a versao pocket do PLA e avalie o E_out medio (1000 execucoes) apos i iteracoes 
# em um conjunto gerado aleatoriamente em X de tamanho N2 de acordo com as seguintes configuracoes: 
# (Nao esqueca de gerar graficos comparando a hipotese g encontrada com a funcao target utilizada)
# a) Inicializando os pesos com 0, i = 10, N1 = 100 e N2 = 1000
# b) Inicializando os pesos com 0, i = 50, N1 = 100 e N2 = 1000
# c) Inicializando os pesos utilizando Regressao Linear, i = 10, N1 = 100 e N2 = 1000
# d) Inicializando os pesos utilizando Regressao Linear, i = 50, N1 = 100 e N2 = 1000

#Usar os conceitos de construção do PLA de Q1~4, com o ruído de Q10~12, iniciando os pesos zerados;
#Depois usar o conceito da Q8, de inicializar com os pesos de RegLin
