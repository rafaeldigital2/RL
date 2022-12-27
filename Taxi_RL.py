# -*- coding: utf-8 -*-
"""Taxi_RL.py

Complemento do livro:
Fundamentos de Aprendizagem por Reforço por Rafael Ris-Ala

# Aprendizagem por Reforço com Q-Learning

# 1.0 Conhecendo o ambiente

(Open Gym, Taxi v3)
"""

import gym
import random

env = gym.make('Taxi-v3').env

env.render()

env.reset()

env.render()

# 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = dropoff
print("Total de Ações: {}".format(env.action_space))

print("Total de Estados: {}".format(env.observation_space))

len(env.P)

env.P

# Função env.encode(taxi_linha,taxi_coluna,passageiro_saida,passageiro_chegada)
env.encode(4, 2, 2, 3)
# Essa conformação é o estado: 451

env.s = 451
env.render()

# Tabela inicial de recompensas "P", com estados e ações
# Este dicionário tem a estrutura {action: [(probability, nextstate, reward, done)]}. Ou seja:
# ação 0 (sul) possui probabilidade 1 de executar essa ação, próximo estado 451, recompensa -1 e se alcançou o final.
# ação 1 (norte) possui probabilidade 1 de executar essa ação, próximo estado 351, recompensa -1 e se alcançou o final.
# ação 2 (leste) possui probabilidade 1 de executar essa ação, próximo estado 451, recompensa -1 e se alcançou o final.
# ação 3 (oeste) possui probabilidade 1 de executar essa ação, próximo estado 431, recompensa -1 e se alcançou o final.
# ação 4 (pegar) possui probabilidade 1 de executar essa ação, próximo estado 451, recompensa -10 e se alcançou o final.
# ação 5 (largar) possui probabilidade 1 de executar essa ação, próximo estado 451, recompensa -10 e se alcançou o final.
env.P[451]

env.s = 479
env.render()

"""Obs.: Para sair do estado 451 até 479 tem que realizar todo o somatório (13) de passos (-1) até o final 20: resultando em +7

"""

env.P[479]

"""# 2.0 Usando ações aleatórias
(Sem utilizar a Aprendizagem por Reforço)

"""

env.s = 484  

epochs = 0   
penalties = 0   

frames = [] 

done = False

while not done:
    action = env.action_space.sample()  
    state, reward, done, info = env.step(action)  

    if reward == -10:  
        penalties += 1
    
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Total de ações executadas: {}".format(epochs))
print("Total de penalizações recebidas: {}".format(penalties))

"""# 2.1 Mostrando a animação dos movimentos realizados

"""

from IPython.display import clear_output
from time import sleep

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

env.s = 484
env.render()

"""# 3.0 Treinamento com o algoritmo

Aplicando o Q-learning: Qt(s,a) = Qt-1(s,a) +αTDt(a,s)
"""

import numpy as np
q_table = np.zeros([env.observation_space.n, env.action_space.n])
q_table.shape

q_table

q_table[451]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from IPython.display import clear_output
# 
# # Hiperparâmetros:
# alpha = 0.1     
# gamma = 0.6     
# epsilon = 0.1   
# 
# for i in range(100000):
#   estado = env.reset()
# 
#   penalidades, recompensa = 0, 0
#   done = False
#   while not done:
#     # Qnd o valor de "random.uniform(0, 1)" for menor que 0.1 (epsilon),    
#     if random.uniform(0, 1) < epsilon:
#       # executará uma ação aleatória (Exploração).
#       acao = env.action_space.sample()
#     # caso contrário, 
#     else:
#       # executará a ação de maior valor, conforme tabela q_table e estado informado (Aproveitamento).
#       acao = np.argmax(q_table[estado])
# 
#     proximo_estado, recompensa, done, info = env.step(acao)
# 
#     q_antigo = q_table[estado, acao]
#     proximo_maximo = np.max(q_table[proximo_estado])
# 
#     q_novo = (1 - alpha) * q_antigo + alpha * (recompensa + gamma * proximo_maximo)
#     q_table[estado, acao] = q_novo
# 
#     if recompensa == -10:
#       penalidades += 1
# 
#     estado = proximo_estado
# 
#   if i % 100 == 0:
#     clear_output(wait=True)
#     print('Episódio: ', i)
# 
# print('Treinamento concluído')

q_table

q_table[451]

"""# 4.0 Testando a Tabela-Q
Agora a Tabela-Q já está aprendida! Segue cada passo:
"""

env.s = 451
env.render()

# Sendo: 0 = south, 1 = north, 2 = east, 3 = west, 4 = pickup, 5 = dropoff
q_table[451]

env.step(1)
env.render()

print(env.s)

q_table[env.s]

env.step(1)
env.render()

q_table[env.s]

env.step(3)
env.render()

q_table[env.s]

env.step(3)
env.render()

q_table[env.s]

env.step(0)
env.render()

"""E assim o taxi segue as maiores recompensas...
Ou seja, até pegar o passageiro e o deixar em seu destino.
"""

env.s = 479
env.render()

q_table[479]

env.step(5)
env.render()

print(env.s)

q_table[475]

"""# 5.0 Testando o agente treinado
(Resolvendo o problema com a aprendizagem obtida (q_table))
"""

total_penalidades = 0
episodios = 50
frames = []

for ep in range(episodios):
  estado = env.reset()
  penalidades, recompensa = 0, 0
  done = False
  while not done:
    acao = np.argmax(q_table[estado])
    estado, recompensa, done, info = env.step(acao)

    if recompensa == -10:
      penalidades += 1
    
    frames.append({
        'frame': env.render(mode='ansi'),
        'episode': ep,
        'state': estado,
        'action': acao,
        'reward': recompensa
    })

  total_penalidades += penalidades

print('Episódios', episodios)
print('Penalidades', total_penalidades)

frames

frames[0]

from time import sleep

for frame in frames:
  clear_output(wait=True)
  print(frame['frame'])
  print(f"Episódio: {frame['episode']}")
  print('Estado:', frame['state'])
  print('Ação:', frame['action'])
  print('Recompensa:', frame['reward'])
  sleep(1)