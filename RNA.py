# importação das bibliotecas necessárias

# pybrain
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# função para carregar os dados de treinamento
def getData( path ):
    #Open file
    file = open( path, "r" )
    
    data = []    
    
    for linha in file:        # obtem cada linha do arquivo
      linha = linha.rstrip()  # remove caracteres de controle, \n
      digitos = linha.split(" ")  # pega os dígitos
      for numero in digitos:   # para cada número da linha
        data.append( numero )  # add ao vetor de dados  
    
    file.close()
    return data
    

# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( 45, 490, 412, 2 )    # define network 
dataSet = SupervisedDataSet( 45, 2 )  # define dataSet


arquivos = ['0.txt','1.txt', '2.txt', '3.txt', '4.txt',
            '5.txt', '6.txt', '7.txt', '8.txt', '9.txt']
  
#arquivos = ['1.txt']          
# a resposta do número
#resposta = [ [1] ] 
resposta = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]] 

i = 0
for arquivo in arquivos:           # para cada arquivo de treinamento
    data =  getData( arquivo )            # pegue os dados do arquivo
    dataSet.addSample( data, resposta[i] )  # add dados no dataSet
    i = i + 1


# trainer
trainer = BackpropTrainer( network, dataSet )
error = 1
iteration = 0
outputs = []
file = open("outputs.txt", "w") # arquivo para guardar os resultados

while error > 0.001: # 10 ^ -3
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# Fase de teste
arquivos = ['0- test.txt','1- test.txt','2- test.txt','3- test.txt','4- test.txt',
'5- test.txt','6- test.txt','7- test.txt','8- test.txt','9- test.txt']
for arquivo in arquivos:
    data =  getData( arquivo )
    print ( network.activate( data ) )


# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()

