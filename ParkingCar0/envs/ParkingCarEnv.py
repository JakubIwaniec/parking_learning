import pygame
import gymnasium as gym
from gymnasium import spaces
# from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from typing import Optional
import random
import numpy as np
import os


current_file_path = os.path.abspath(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
SKINS_PATH = os.path.join(ROOT_PATH, 'skins')
PARKING_IMAGE = pygame.image.load(os.path.join(SKINS_PATH, 'parking_lanes.png'))
# kordy srodka dla pierwszego miejsca parkowania
#   - render celu
FIRST_LANE_CENTER = (8, 15)
# kordy wielkosci parkingu
PARKING_XY = PARKING_IMAGE.get_size()


class TargetArea(pygame.sprite.Sprite):
    """
    W razie jakbyśmy chcieli jednak
    obiektowo zrobić cele/pasy to
    zrobiłem pierwszy zarys obiektu
    """
    def __init__(self, point: tuple = (0, 0)):
        super().__init__()
        self._path_to_dest_image = os.path.join(SKINS_PATH, 'dest.png')
        self.point = point
        self.is_pressed: bool = False

    def set_point(self, new_point):
        assert type(new_point) is tuple
        assert len(new_point) == 2
        self.point = new_point

    def set_image(self):
        if self.is_pressed:
            self._path_to_dest_image = os.path.join(SKINS_PATH, 'dest_active.png')
        else:
            self._path_to_dest_image = os.path.join(SKINS_PATH, 'dest.png')

    def get_path_to_image(self):
        return self._path_to_dest_image


class ParkingCarEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.screen_width = 400
        self.screen_height = 400
        self.screen = None
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((128, 128, 128))
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

        self.dest_x = FIRST_LANE_CENTER[0] + 30*random.randint(0, 5)
        self.dest_y = FIRST_LANE_CENTER[1]
        # czy narysowano cel
        self.rendered_dest = False

        self.low = np.array([
            0,
            0,
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
            # x bez korda aby auto spawnowalo sie tez 'pod' parkingiem
            self.np_random.uniform(low=int(self.car_width/2 + 1), high=self.high[0]),
            # wczesniej byla szansa ze auto bedzie wewnatrz parkingu
            self.np_random.uniform(low=(PARKING_XY[1] + int(self.car_height/2 + 1)) * 2, high=self.high[1]),
            0,
            90,
            self.dest_x,
            self.dest_y,
        ])

        if self.render_mode == "human":
            self.rendered_dest = False
            self.surf.fill((128, 128, 128))
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

        scale_x = self.screen_width / self.map_width
        scale_y = self.screen_height / self.map_height

        pos = self.state[0], self.state[1]
        car_pos_x = pos[0] * scale_x - int(self.car_width/2) + 1
        car_pos_y = pos[1] * scale_y - int(self.car_height/2) + 1
        car_render = pygame.Rect(car_pos_x, car_pos_y,
                                 self.car_width, self.car_height)
        car_render = car_render.scale_by(scale_x, scale_y)

        # nie wiem po co to transform, kiedy zmienialem renderowanie
        #   tylko mi obraz odwracalo
        self.surf = pygame.transform.flip(self.surf, False, True)

        target = TargetArea()
        target_surf = pygame.image.load(target.get_path_to_image())   # czy jest koniecznosc przy kazdym kroku ladowac na nowo zdjecie

        self.surf.blit(source=PARKING_IMAGE, dest=(0, 0))
        self.surf.blit(source=target_surf, dest=(self.dest_x, self.dest_y))
        pygame.draw.rect(
            self.surf,
            'green',
            car_render,
        )
        self.rendered_dest = True
        self.screen.blit(self.surf, (0, 0))

        self.surf.fill((128, 128, 128))

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
