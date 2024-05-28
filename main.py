import ParkingCar
import gymnasium as gym


env = gym.make("ParkingCar/ParkingCar-v1", render_mode='human')
print(env.reset())

while True:
    env.render()
    env.step(0)
