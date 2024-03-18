from gymnasium.envs.registration import register

register(
     id="ParkingCar0/ParkingCar-v0",
     entry_point="ParkingCar0.envs:ParkingCarEnv",
     max_episode_steps=300,
)
