import ParkingCar0
import gymnasium as gym


env = gym.make("ParkingCar0/ParkingCar-v0", render_mode='human')
print(env.reset())

while True:
    env.render()
    env.step(1)
