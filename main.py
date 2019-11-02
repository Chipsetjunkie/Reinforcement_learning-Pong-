from game import ping_pong
import numpy as np
import pygame
import sys,os

game_count = 100
a = ping_pong(game_count,0) # second parameter is the test condition, 0 for play, 1 for test



batch = [10,10,4]
path = os.getcwd()+'/best_q'
path += '/'+os.listdir(path)[0]
q_table = np.load(path)
done = 0

a.set_render(1)


observation,reward,done,progress = a.run(None)
observation = np.array(observation)/np.array(batch)
observation = tuple(observation.astype(np.int))

while not done :
        action = np.argmax(q_table[observation])
        new_env,reward,done,progress = a.run(action)
        new_obs = np.array(new_env)/np.array(batch)
        new_obs = tuple(new_obs.astype(np.int))
        observation = new_obs

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_DOWN]:
            a.run(3)
        if pressed[pygame.K_UP]:
            a.run(4)
