import gymnasium as gym

from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from typing import Optional
import numpy as np
import pygame
import random


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
        self.is_open = True
        self.state = None

        self.map_width = 400
        self.map_height = 400

        self.gas_force = 1
        self.brake_force = 2
        self.rotate_angle = 10
        self.rotation_max = 360
        self.velocity_max = 10  # ???????

        self.car_width = 15
        self.car_height = 31

        # start params low-high
        self.low = np.array([
            int(self.car_width / 2 + 1),
            int(self.car_height / 2 + 1),
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

        # Parking params
        self.parking = None
        self.parking_slots = 6
        self.parking_slot_width = 20
        self.parking_slot_height = 40
        self.parking_slot_border_thickness = 3

        # Destination params
        self.destination = None
        self.destination_width = self.parking_slot_width / 2 + 1
        self.destination_height = self.parking_slot_height/2
        self.destination_outline_thickness = 1

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        car_x, car_y, car_v, car_r, dest_x, dest_y = self.state

        if action == 1:
            car_v += self.gas_force
        elif action == 2:
            if car_v > self.brake_force:
                car_v -= self.brake_force
            elif 0 < car_v < self.brake_force:
                car_v = 0
            else:
                car_v -= self.gas_force / 2
        elif action == 3:
            if car_v >= 0:
                car_r += self.rotate_angle
            else:
                car_r -= self.rotate_angle
        elif action == 4:
            if car_v >= 0:
                car_r -= self.rotate_angle
            else:
                car_r += self.rotate_angle

        if car_v >= self.velocity_max:
            car_v = self.velocity_max
        elif car_v <= -self.velocity_max:
            car_v = -self.velocity_max

        # <- action == 0, here we can add movement resistance
        car_x += car_v * np.cos(car_r / 180 * np.pi)
        car_y += car_v * np.sin(car_r / 180 * np.pi)

        # <- condition of hitting the edge of the screen
        # actually without rotation included

        done = bool(
            self.destination.is_inside(car_x, car_y) and
            car_v == 0
        )

        terminated = bool(
            done or
            car_x < self.low[0] or car_x > self.high[0] or
            car_y < self.low[1] or car_y > self.high[1]
        )

        reward = 0

        self.state = car_x, car_y, car_v, car_r, dest_x, dest_y

        # print(f'State: {self.state}, reward: {reward}, terminated: {terminated}')

        return np.array(self.state, dtype=np.float32), reward, terminated, done, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        super().reset(seed=seed)

        # centre of map
        parking_x = 200 - (self.parking_slot_border_thickness * (self.parking_slots/2 + 0.5) +
                           self.parking_slot_width * self.parking_slots/2)
        parking_y = 200 - (self.parking_slot_height + self.parking_slot_border_thickness * 2) / 2
        self.parking = Parking(parking_x, parking_y, self.parking_slot_width, self.parking_slot_height,
                               self.parking_slot_border_thickness, self.parking_slots)

        destination_x_center, destination_y_center = self.parking.get_random_slot_coords()
        destination_x = destination_x_center - (self.destination_width + 2 * self.destination_outline_thickness) / 2
        destination_y = destination_y_center - (self.destination_height + 2 * self.destination_outline_thickness) / 2
        self.destination = Destination(destination_x, destination_y, self.destination_width, self.destination_height,
                                       self.destination_outline_thickness)

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
            destination_x_center,
            destination_y_center,
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

        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((128, 128, 128))

        self.parking.draw(surf)  # without scaling right now
        self.destination.draw(surf)

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

        surf.blit(rotated_car, rotated_pos)

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

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
            self.is_open = False


class Destination:
    def __init__(self, x, y, dest_width, dest_height, outline_thickness,
                 color=pygame.Color("yellow"), outline_color=pygame.Color("black")):
        """
        :param x: width relative to Parking 'x'
        :param y: height relative to Parking 'y'
        """
        self.x = x
        self.y = y
        self.main_color = color
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness

        self.dest_width = dest_width
        self.dest_height = dest_height

    def is_inside(self, x, y):
        return bool(self.x <= x <= (self.x + self.dest_width) and
                    self.y <= y <= (self.y + self.dest_height))

    def get_centre(self):
        return self.x + self.dest_width/2, self.y + self.dest_height/2

    def draw(self, surface):
        pygame.draw.rect(surface, self.main_color,
                         (self.x, self.y, self.dest_width, self.dest_height))
        pygame.draw.rect(surface, self.outline_color,
                         (self.x, self.y, self.dest_width, self.dest_height),
                         width=self.outline_thickness)


class Parking:
    def __init__(self, x, y, slot_width=40, slot_height=20, border_thickness=3, slots=6,
                 fill_color=(47, 79, 79), border_color=(255, 255, 255)):
        """
        x, y - left down corner of parking space
        ~actually without scaling~
        """
        self.fill_color = fill_color
        self.border_color = border_color

        self.border_thickness = border_thickness  # Better if odd
        self.slots = slots
        self.slot_width = slot_width
        self.slot_height = slot_height

        self.x = x
        self.y = y

    def get_width(self):
        return (self.border_thickness + self.slot_width) * self.slots + self.border_thickness - 1

    def get_height(self):
        return self.border_thickness + self.slot_height

    def get_random_slot_coords(self):
        """Return coords x, y of centre of random parking slot."""
        random_slot = random.randint(0, self.slots - 1)
        x = self.border_thickness + self.slot_width / 2 + random_slot * (self.slot_width + self.border_thickness)
        y = self.get_height() / 2
        return self.x + x, self.y + y

    def draw(self, surface):
        border_side = (self.border_thickness - 1) / 2
        up_line_width = self.get_width()

        # Down line
        pygame.draw.line(surface, self.border_color, (self.x, self.y), (self.x + up_line_width, self.y),
                         self.border_thickness)
        x = self.x + border_side
        y = self.y + border_side
        # First left line
        pygame.draw.line(surface, self.border_color, (x, y), (x, y + self.slot_height), self.border_thickness)

        x += border_side + 1
        for slot in range(self.slots):
            # dest
            pygame.draw.rect(surface, self.fill_color,
                             (x, y, self.slot_width, self.slot_height))  # y+1? tak było
            x += self.slot_width + border_side
            pygame.draw.line(surface, self.border_color, (x, y),
                             (x, y + self.slot_height), self.border_thickness)
            x += border_side + 1
