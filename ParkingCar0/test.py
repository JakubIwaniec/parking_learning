import ParkingCar0
from stable_baselines3 import PPO
import gymnasium as gym

# Stwórz środowisko
env = gym.make("ParkingCar0/ParkingCar-v0", render_mode='human')

# Załaduj wytrenowany model
model = PPO.load("ppo_parking")

# Zresetuj środowisko, pobierz tylko obserwację (ignorujemy dodatkowe informacje)
obs, info = env.reset()

# Testuj model
while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, done, _ = env.step(action)
    env.render()
    if terminated or done:
        obs, info = env.reset()

