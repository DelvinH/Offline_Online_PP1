import pygame as pg
from pygame.color import THECOLORS as color
import math
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import inf

random.seed()


# ### Observation Space
# The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
# | Num | Observation                                                 | Min     | Max    | Unit         |
# |-----|-------------------------------------------------------------|---------|--------|--------------|
# | 0   | distance of ship from target waypoint                       | -Inf    | Inf    | position (m) |
# | 1   | heading of ship to target waypoint                          | -pi     | pi     | angle (rad)  |
# ### Action Space
# There are 7 discrete deterministic actions:
# | Num | Action                                                      |
# |-----|-------------------------------------------------------------|
# | 0   | Change course 15 degrees left                               |
# | 1   | Change course 10 degrees left                               |
# | 2   | Change course 5 degrees left                                |
# | 3   | Don't change course                                         |
# | 4   | Change course 5 degrees right                               |
# | 5   | Change course 10 degrees right                              |
# | 6   | Change course 15 degrees right                              |


# Constants
ship_length = 20  # pixels
ship_width = 10  # pixels
waypoint_threshold = 30  # pixels


class ShipWorld:
    def __init__(self, env_size, start):
        # Pygame initialization
        pg.init()
        self.render = True
        self.env_size = env_size
        if self.render:
            self.screen = pg.display.set_mode(self.env_size)
            self.screen.set_alpha(None)

        # Colors and Rendering
        self.screen_color, self.ship_color, self.obs_color, self.wp_color, self.dist_to_wp_color, \
            self.course_reached_color, self.not_course_reached_color = color['black'], color['red'], color['gray'], \
            color['green'], color['white'], color['blue'], color['orange']
        self.ship = None  # Separate surfaces for mask collisions
        self.obstacles_screen = None

        # Variable initialization
        self.max_turn_rate = 40  # deg/sec
        self.step_per_second = 10  # step/sec
        self.v = 80  # pixel/s
        self.meter_per_pixel = 1  # m/pixel
        self.multiplier = 1  # simulation speed multiplier

        # Observation
        self.min_distance = -inf
        self.max_distance = +inf
        self.min_heading = -math.pi  # heading = number of degrees right of directin towards target waypoint
        self.max_heading = math.pi
        self.low = np.array([self.min_distance, self.min_heading], dtype=np.float32)
        self.high = np.array([self.max_distance, self.max_heading], dtype=np.float32)
        self.num_actions = 3
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # Path
        self.obstacles = []
        self.path = None

        # Current target waypoint
        self.target_waypoint = [500, 500]
        self.target_waypoint_index = 1

        # Create own ship
        self.start = start
        self.x = start[0]
        self.y = start[1]
        self.r = math.radians(start[2])  # right = 0; cw is positive; rad
        self.state = (self.distance_to_waypoint(), self.heading_to_waypoint())
        self.poly = self.get_vertices(self.x, self.y, self.r)

        # Own ship status
        self.course_reached = True
        self.done = False
        self.can_change_course = False  # controls whether course can be overridden before being reached or not
        self.course = self.r  # rad

    def step(self, action):
        # Translate action
        action = (action - (self.num_actions-1)/2) * 15  # deg

        # Clear screen
        if self.render:
            self.screen.fill(self.screen_color)

        # Update course based on action
        if self.course_reached or self.can_change_course:  # only update course if previous course has been reached
            self.course = self.r + math.radians(action)  # action = [-15, -10, -5, 0, 5, 10, 15] deg
            self.course = round(self.course * 180 / math.pi) * math.pi / 180  # round to reduce hysteresis

        # Turn ship towards course
        if action == 0 and self.course_reached:
            self.course_reached = True
        else:
            self.course_reached = False

        if not self.course_reached:
            if self.r > self.course:
                self.r -= math.radians(self.max_turn_rate) / self.step_per_second
            elif self.r < self.course:
                self.r += math.radians(self.max_turn_rate) / self.step_per_second

            if abs(self.r - self.course) < 1.5 * math.radians(self.max_turn_rate) / self.step_per_second:
                self.r = self.course
                self.course_reached = True

        # Move forward
        self.x = self.x + math.cos(self.r) * self.v / self.step_per_second
        self.y = self.y + math.sin(self.r) * self.v / self.step_per_second

        # Rendering
        self.draw_obstacles()
        self.draw_waypoint()
        self.draw_own_ship()
        self.draw_course_indicators()

        # Observation
        distance = self.distance_to_waypoint()
        heading = self.heading_to_waypoint()
        self.state = (distance, heading)

        # Finish step
        reward = self.get_reward()
        if self.render:
            pg.display.update()

        return np.array(self.state, dtype=np.float32), reward, self.done, {}

    def add_obstacle(self, *obstacles):
        # obstacle = ["polytype", dims]
        for obstacle in obstacles:
            self.obstacles.append(obstacle)

    def check_collision(self):
        # True if no obstacle intersection
        if len(self.obstacles) == 0:
            return True
        obs_mask = pg.mask.from_threshold(self.obstacles_screen, self.obs_color, (1, 1, 1, 255))
        ship_mask = pg.mask.from_threshold(self.ship, self.ship_color, (1, 1, 1, 255))
        return obs_mask.overlap(ship_mask, (0, 0)) is None

    def check_boundary(self):
        # True if within bounds
        return bool(0 < self.x < self.env_size[0] and 0 < self.y < self.env_size[1])

    def generate_path(self):
        pass

    def draw_own_ship(self):
        self.poly = self.get_vertices(self.x, self.y, self.r)
        self.ship = pg.Surface(self.env_size, pg.SRCALPHA)
        pg.draw.polygon(self.ship, self.ship_color, self.poly)
        if self.render:
            self.screen.blit(self.ship, (0, 0))

    def draw_waypoint(self):
        if self.render:
            pg.draw.circle(self.screen, self.wp_color, self.target_waypoint, waypoint_threshold)

    def draw_course_indicators(self):
        if not self.render:
            return
        pg.draw.line(self.screen, self.dist_to_wp_color, (self.x, self.y), self.target_waypoint)  # shortest path to target wp
        course_line_length = 30
        if self.course_reached:
            course_line_color = self.course_reached_color
        else:
            course_line_color = self.not_course_reached_color
        pg.draw.line(self.screen, course_line_color, (self.x, self.y), (self.x + course_line_length * math.cos(self.course),
                     self.y + course_line_length * math.sin(self.course)))  # current course

    def draw_obstacles(self):
        self.obstacles_screen = pg.Surface(self.env_size, pg.SRCALPHA)
        for obstacle in self.obstacles:
            obs_type = obstacle[0]
            if obs_type == "rect" or obs_type == "rectangle":
                pg.draw.rect(self.obstacles_screen, self.obs_color, obstacle[1])
            elif obs_type == "poly" or obs_type == "polygon":
                pg.draw.polygon(self.obstacles_screen, self.obs_color, obstacle[1])
            elif obs_type == "circ" or obs_type == "circle":
                pg.draw.circle(self.obstacles_screen, self.obs_color, obstacle[1][0], obstacle[1][1])
            elif obs_type == "elli" or obs_type == "ellipse":
                pg.draw.ellipse(self.obstacles_screen, self.obs_color, obstacle[1])
            else:
                pass
        if self.render:
            self.screen.blit(self.obstacles_screen, (0, 0))

    def distance_to_waypoint(self):
        return math.sqrt(math.pow(self.x - self.target_waypoint[0], 2) + math.pow(self.y - self.target_waypoint[1], 2))

    def heading_to_waypoint(self):
        direction_to_waypoint = math.atan2(self.target_waypoint[1] - self.y, self.target_waypoint[0] - self.x)
        heading_to_waypoint = self.r - direction_to_waypoint
        while heading_to_waypoint > math.pi:
            heading_to_waypoint = heading_to_waypoint - 2 * math.pi
        while heading_to_waypoint < -math.pi:
            heading_to_waypoint = heading_to_waypoint + 2 * math.pi
        # print(heading_to_waypoint)
        return heading_to_waypoint

    def get_reward(self):
        reward = 0
        self.done = False
        # Collision and Boundary check
        if not self.check_boundary() or not self.check_collision():
            reward += -80
            self.done = True
        # Waypoint check
        if self.distance_to_waypoint() > waypoint_threshold:
            time_reward = -1 / self.step_per_second
            distance_reward = -5 * self.distance_to_waypoint() / 1000 / self.step_per_second
            heading_reward = -5 * abs(self.heading_to_waypoint()) / math.pi / self.step_per_second
            reward += distance_reward + time_reward + heading_reward
        else:
            reward += 100
            self.done = True
        return reward

    def get_action_space(self):
        return self.action_space

    def get_vertices(self, x, y, r):
        x1 = x - ship_length * math.cos(r) / 2 - ship_width * math.sin(r) / 2
        y1 = y - ship_length * math.sin(r) / 2 + ship_width * math.cos(r) / 2
        x2 = x + ship_length * math.cos(r) / 2 - ship_width * math.sin(r) / 2
        y2 = y + ship_length * math.sin(r) / 2 + ship_width * math.cos(r) / 2
        x3 = x + ship_length * math.cos(r) / 2 + ship_width * math.sin(r) / 2
        y3 = y + ship_length * math.sin(r) / 2 - ship_width * math.cos(r) / 2
        x4 = x - ship_length * math.cos(r) / 2 + ship_width * math.sin(r) / 2
        y4 = y - ship_length * math.sin(r) / 2 - ship_width * math.cos(r) / 2
        x5 = x + ship_length * math.cos(r)
        y5 = y + ship_length * math.sin(r)

        vertices = [(x1, y1), (x2, y2), (x5, y5), (x3, y3), (x4, y4)]
        return vertices

    def reset(self):
        self.x = self.start[0]
        self.y = self.start[1]
        self.r = self.start[2]
        self.course = self.r
        self.course_reached = True
        self.done = False
        distance = self.distance_to_waypoint()
        heading = self.heading_to_waypoint()
        self.state = (distance, heading)
        return np.array(self.state, dtype=np.float32)

    def set_simulation_speed(self, multiplier):
        if multiplier <= 0:
            multiplier = 1
        self.multiplier = multiplier

    def set_waypoint(self, waypoint):
        self.target_waypoint = waypoint

    def set_start(self, start):
        self.start = start

    def set_render(self, render):
        self.render = render


