from gymnasium.envs.registration import register

register(
     id="ParkingCar/ParkingCar-v1",
     entry_point="ParkingCar.envs:ParkingCarEnv",
     max_episode_steps=300,
)
