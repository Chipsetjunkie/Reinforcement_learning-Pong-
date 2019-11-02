import pygame
import random
import numpy as np
clock = pygame.time.Clock()

class ping_pong():

    def __init__(self,n,t):
        self.width = 600
        self.height = 500
        self.color_half = (255,0,0)
        self.color_bar = (255,255,255)
        self.bar_t = 2
        self.left = [0,int(self.height/2)]
        self.right = [int(self.width),int(self.height/2)]
        self.score = 0
        self.speed = 10
        self.ball_pos = [int(self.width/2),int(self.height/2)]
        self.x_dir,self.y_dir = (2,2)
        self.rad = 20
        self.reward = 0
        self.done = 0
        self.game_n = n
        self.render = 0
        self.train = t
        self.screen = None

    def render_init(self):

        if self.render:
            if self.screen == None:
                self.screen = pygame.display.set_mode((self.width,self.height))

    def get_max(self):
        return np.array((self.width,self.height,self.height))

    def playground(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.line(self.screen,self.color_half,(self.width/2,0),(self.width/2,self.height),self.bar_t)

    def bar(self):
        pygame.draw.line(self.screen,self.color_bar,(self.left[0],self.left[1]-int(self.height/10)),(self.left[0],self.left[1]+int(self.height/10)),20)
        pygame.draw.line(self.screen,self.color_bar,(self.right[0],self.right[1]-int(self.height/10)),(self.right[0],self.right[1]+int(self.height/10)),20)

    def ball(self):
        pygame.draw.circle(self.screen, self.color_bar, self.ball_pos, self.rad)


    def update_ball(self):
        # right paddle
        if self.ball_pos[0]+self.x_dir == self.right[0]-(self.rad+self.bar_t) and self.right[1]-(self.height/10) <= self.ball_pos[1]+self.y_dir <= self.right[1]+ (self.height/10):
            self.x_dir = -self.x_dir
            self.reward = 50

        #left paddle
        if self.ball_pos[0]+self.x_dir == self.left[0]+(self.rad+self.bar_t) and self.left[1]-int(self.height/10) <self.ball_pos[1]+self.y_dir <self.left[1]+int(self.height/10):
            self.x_dir = -self.x_dir


        #reset
        if  self.ball_pos[0]+self.x_dir < self.rad or self.ball_pos[0]+self.x_dir  > (self.width-self.rad):
            if self.ball_pos[0]+self.x_dir  > (self.width-self.rad):
                self.reward -= 10
            if self.ball_pos[0]+self.x_dir < self.rad:
                #self.reward += 20
                self.score += 1

            self.x_dir = -self.x_dir

            if self.train == 0:
                self.ball_pos = [int(self.width/2),int(self.height/2)]
                self.y_dir = random.choice((2,-2))

        # y-axis boundary
        if self.ball_pos[1]+self.y_dir < self.rad or self.ball_pos[1]+self.y_dir > (self.height-self.rad):
            self.y_dir = -self.y_dir

        self.ball_pos = [self.ball_pos[0]+self.x_dir,self.ball_pos[1]+self.y_dir]
        if self.render:
            self.ball()



    def spawn(self):
        if self.render:
            self.playground()
            self.bar()
            self.ball()
            pygame.display.flip()
        self.reward = 0

    def set_render(self,state):
        self.render = state

    def observation(self):
        #return positions, reward, done
        return (self.ball_pos + [self.right[1]]),self.reward, self.done,self.score


    def run(self,action):
        self.done = 0
        if self.render:
            self.render_init()
            self.playground()
            self.bar()
        if self.score > self.game_n:
            self.done = 1
            self.score = 0
            action = None

        if action == None:
                self.spawn()
                return self.observation()
        self.reward = -0.1
        self.update_ball()


        if action == 0:
            if self.right[1]-self.speed >= 50:
                self.right = [self.right[0],self.right[1]-self.speed]

        if action == 1:
            if self.right[1]+self.speed <= 450:
                self.right = [self.right[0],self.right[1]+self.speed]

        if self.train == 0:
            if action == 3:
                if self.left[1]+self.speed <= 450:
                    self.left = [self.left[0],self.left[1]+self.speed]


            if action == 4:
                if self.left[1]-self.speed >= 50:
                    self.left = [self.left[0],self.left[1]-self.speed]

        if self.train == 0:
            clock.tick(120)
        if self.render == 1:
            pygame.display.flip()
        return self.observation()
