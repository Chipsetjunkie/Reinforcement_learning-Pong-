from game import ping_pong
import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import os
import sys
#################################################################

#Game Parameters
pygame.init()
game_count = 100 #tries or len of game
a = ping_pong(game_count,1)
done = 0
render = 0 # render value '1' not recommended, will slow down performance.
a.set_render(render)#shows the game, performance takes a beating


# Hyperparameters
Episodes = 500
LEARNING_RATE = 0.1
DISCOUNT = 0.8
EPSILON = 1.0
DECAY_RATE = EPSILON - (1/Episodes)
DECAY_END = Episodes*0.25


# Q_table dependencies
batch = [10,10,4]
table_size = (a.get_max()/np.array(batch)).astype(np.int)
actions = [2]

# Loading Q_tables if available else start afresh
path = os.getcwd()+'/q_table'
try:
    os.listdir(path)
except OSError:
    os.mkdir(path)
    print ("Directory Created!")

if "checkpoint.npy" not in os.listdir(os.getcwd()):
    q_table = np.zeros((list(table_size)+actions))
    print ("New Start\n")

else:
    print ("loading data... ", end=" ")
    q_path = path + np.load('checkpoint.npy')[0]+'.npy'
    try:
        q_table = np.load(q_path)

    except FileNotFoundError:
        p,f = os.path.split(q_path)
        print ("Fail!!\n")
        print ("file ' {0} ' not found in the path ' {1} '".format(f,p))
        print ('Exiting..')
        sys.exit()
    print("OK!\n")

#Performance Variable
data = []
score = []
min_val = -10000
start = time.time()


#-----------------------------------
'''
############
ACTUAL TRAINING
############
'''
for episode in range(Episodes):
    observation,reward,done,progress = a.run(None)
    observation = np.array(observation)/np.array(batch)
    observation = tuple(observation.astype(np.int))
    #print (observation)

    while not done :
            ## only Window control, not related to training
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit()
            ######## ****************************


            if random.uniform(0,1) > EPSILON:
                action = np.argmax(q_table[observation])
            #print (action)
            else:
                action = random.choice((0,1))


            new_env,reward,done,progress = a.run(action)
            new_obs = np.array(new_env)/np.array(batch)
            new_obs = tuple(new_obs.astype(np.int))
            #print (new_obs)

            if not done:
                max_Q = np.max(q_table[new_obs])
                #print (max_Q)
                #print ((observation+(action,)))
                current_q = q_table[(observation+(action,))]
                #print (current_q)

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_Q)
                #print (new_q)
                q_table[observation+(action,)] = new_q  #new_q to previous observation

            observation = new_obs
            score.append(reward)


            #Epsilon decay
            if episode < DECAY_END:
                EPSILON *= DECAY_RATE

            else:
                EPSILON = 0

    data.append(sum(score))
    score = []
    if episode%10 ==0:
        print (episode)

    if data[-1] > min_val:
        min_val = data[-1]
        np.save(os.path.join(path, str(min_val)), q_table)


## Performance Statistics and Logging
print (time.time()-start)
np.save('checkpoint.npy',['/'+str(min_val)])

plt.plot(data)
plt.show()
