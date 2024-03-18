import gymnasium as gym
import numpy as np

from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from typing import Optional


class ParkingCarEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.screen_width = 800
        self.screen_height = 800
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

        self.low = np.array([int(self.car_width/2 + 1), int(self.car_height/2 + 1), 0, 0, 0, 0], dtype=np.float32)
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

        print(action)

        # <- condition of hitting the edge of the screen

        terminated = bool(
            car_x == dest_x and car_y == dest_y
        )
        reward = 0
        self.state = car_x, car_y, car_v, car_r, dest_x, dest_y

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # na razie na sztywno
        car_x_min, car_x_max = utils.maybe_parse_reset_bounds(options, 0, self.map_width)
        car_y_min, car_y_max = utils.maybe_parse_reset_bounds(options, 0, self.map_height)
        dest_x_min, dest_x_max = utils.maybe_parse_reset_bounds(options, 0, self.map_width)
        dest_y_min, dest_y_max = utils.maybe_parse_reset_bounds(options, 0, self.map_height)

        self.state = np.array([
            self.np_random.uniform(low=car_x_min, high=car_x_max),
            self.np_random.uniform(low=car_y_min, high=car_y_max),
            0,
            90,
            self.np_random.uniform(low=dest_x_min, high=dest_x_max),
            self.np_random.uniform(low=dest_y_min, high=dest_y_max),
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
        car_pos_x = pos[0] * scale_x - int(self.car_width/2) + 1
        car_pos_y = pos[1] * scale_y - int(self.car_height/2) + 1
        car_render = pygame.Rect(car_pos_x, car_pos_y,
                                 self.car_width, self.car_height)
        car_render = car_render.scale_by(scale_x, scale_y)

        pygame.draw.rect(
            self.surf,
            'green',
            car_render,
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # nie wiem do czego to
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
