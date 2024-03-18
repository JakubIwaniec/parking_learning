import ParkingCar0
import gymnasium as gym
import pygame
import numpy as np
import sys
import os
from datetime import datetime

pygame.init()
screen = pygame.display.set_mode((800, 800))

env = gym.make("ParkingCar0/ParkingCar-v0", render_mode='human')

MS_FOR_FRAME = 5

initial_state = env.reset()

for game in range(1):
    done = False
    action = 0
    env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 1
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_a:
                    action = 3
                elif event.key == pygame.K_d:
                    action = 4

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    action = 0
                elif event.key == pygame.K_s:
                    action = 0
                elif event.key == pygame.K_a:
                    action = 0
                elif event.key == pygame.K_d:
                    action = 0

        env.step(action)
        env.render()
        pygame.time.delay(MS_FOR_FRAME)
env.close()
