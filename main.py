import ParkingCar0
import gymnasium as gym



env = gym.make("ParkingCar0/ParkingCar-v0", render_mode='human')
print(env.reset())

for _ in range(20):
    env.render()
    print(env.step(1))