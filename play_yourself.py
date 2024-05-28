import ParkingCar
import gymnasium as gym
import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((400, 400))

env = gym.make("ParkingCar/ParkingCar-v1", render_mode='human')

MS_FOR_FRAME = 30

for game in range(1):
    done = False
    game_reward = 0
    action = 0
    initial_state = env.reset()
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

        state, reward, terminated, done, _ = env.step(action)
        print(reward)
        game_reward += reward
        env.render()
        pygame.time.delay(MS_FOR_FRAME)
        if terminated:
            initial_state = env.reset()
    print(f"Game reward: {game_reward}")
env.close()
