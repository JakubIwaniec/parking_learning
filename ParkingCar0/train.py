from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import ParkingCar0
import gymnasium as gym

# Stwórz środowisko
env = gym.make("ParkingCar0/ParkingCar-v0", render_mode='rgb_array')

# Sprawdzenie środowiska pod kątem błędów
check_env(env)

# Stwórz model PPO
model = PPO("MlpPolicy", env, verbose=1)

# Trenuj model przez 3000 000 kroków
model.learn(total_timesteps=3000000)

# Zapisz wytrenowany model
model.save("ppo_parking")
