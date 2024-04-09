import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from typing import Optional


class ParkingCarEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.screen_width = 400
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.map_width = 400
        self.map_height = 400

        self.gas_force = 1
        self.brake_force = 2
        self.rotate_angle = 10

        self.rotation_max = 360
        self.velocity_max = 100  # ???????

        self.car_width = 15
        self.car_height = 31

        self.low = np.array([
            int(self.car_width/2 + 1),
            int(self.car_height/2 + 1),
            0,
            0,
            0,
            0
        ], dtype=np.float32)
        self.high = np.array([
            self.map_width - self.car_width,
            self.map_height - self.car_height,
            self.velocity_max,
            self.rotation_max,
            self.map_width,
            self.map_height
        ], dtype=np.float32)

        # [Do nothing, gas, brake, left, right]
        self.action_space = spaces.Discrete(5)
        # [car_x, car_y, car_velocity, car_rot, destination_x, destination_y]
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.render_mode = render_mode

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        car_x, car_y, car_v, car_r, dest_x, dest_y = self.state

        if action == 1:
            car_v += self.gas_force
        elif action == 2:
            car_v -= self.brake_force
        elif action == 3:
            car_r += self.rotate_angle
        elif action == 4:
            car_r -= self.rotate_angle

        # <- action == 0, here we can add movement resistance
        car_x += car_v * np.cos(car_r/180 * np.pi)
        car_y += car_v * np.sin(car_r/180 * np.pi)


        # <- condition of hitting the edge of the screen
        # actually without rotation included


        terminated = bool(
            # car_x == dest_x and car_y == dest_y
            car_x < self.low[0] or car_x > self.high[0]
            or car_y < self.low[1] or car_y > self.high[1]
        )
        reward = 0
        self.state = car_x, car_y, car_v, car_r, dest_x, dest_y

        print(f'State: {self.state}, reward: {reward}, terminated: {terminated}')

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # na razie na sztywno
        # car_x_min, car_x_max = utils.maybe_parse_reset_bounds(options, 0, self.map_width)
        # car_y_min, car_y_max = utils.maybe_parse_reset_bounds(options, 0, self.map_height)
        # dest_x_min, dest_x_max = utils.maybe_parse_reset_bounds(options, 0, self.map_width)
        # dest_y_min, dest_y_max = utils.maybe_parse_reset_bounds(options, 0, self.map_height)

        self.state = np.array([
            self.np_random.uniform(low=self.low[0], high=self.high[0]),
            self.np_random.uniform(low=self.low[1], high=self.high[1]),
            0,
            90,
            self.np_random.uniform(low=self.low[4], high=self.high[4]),
            self.np_random.uniform(low=self.low[5], high=self.high[5]),
        ])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((128, 128, 128))

        scale_x = self.screen_width / self.map_width
        scale_y = self.screen_height / self.map_height

        pos = self.state[0], self.state[1]
        car_pos_x = pos[0] * scale_x
        car_pos_y = pos[1] * scale_y

        # Utwórz powierzchnię reprezentującą samochód
        car_render = pygame.Surface((self.car_height, self.car_width), pygame.SRCALPHA)

        # Ustawienie gradientu kolorów
        gradient_rect = pygame.Rect(0, 0, self.car_height, self.car_width)
        color_start = (255, 255, 255)  # Kolor początkowy gradientu (biały)
        color_end = (255, 0, 0)  # Kolor końcowy gradientu (czerwony)
        pygame.draw.rect(car_render, color_start, gradient_rect)
        for x in range(self.car_height):
            progress = x / self.car_height
            current_color = (
                int(color_end[0] * (1 - progress) + color_start[0] * progress),
                int(color_end[1] * (1 - progress) + color_start[1] * progress),
                int(color_end[2] * (1 - progress) + color_start[2] * progress),
                200  # Ustawienie przezroczystości
            )
            pygame.draw.line(car_render, current_color, (x, 0), (x, self.car_width))

        # Obrót samochodu wokół jego środka
        rotated_car = pygame.transform.rotate(car_render, -self.state[3])
        rotated_rect = rotated_car.get_rect(center=(self.car_width / 2, self.car_height / 2))

        # Oblicz pozycję samochodu po obróceniu
        rotated_pos = rotated_rect.move(car_pos_x - self.car_width / 2, car_pos_y - self.car_height / 2)

        self.surf.blit(rotated_car, rotated_pos)

        self.surf = pygame.transform.flip(self.surf, False, True)

        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False