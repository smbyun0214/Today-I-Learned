# 참고 페이지
# https://github.com/ioarun/pygame-robotics/blob/master/planning/gridworld.py

import pygame
import numpy as np

class GridWorld():
    def __init__(self, rect_size, rect_margin, map_path="Pygame/racetrack/maps/sample"):
        self.RECT_WIDTH = rect_size[0]
        self.RECT_HEIGHT = rect_size[1]
        self.RECT_MARGIN = rect_margin

        self._define_color()
        self.background = self.BLACK

        self._define_actions()

        self.map, self.START, self.GOAL = self._load_map(map_path)
        self.map_size = [ len(self.map), len(self.map[0]) ]

        self.world_size = [ len(self.map[0]), len(self.map) ]
        self.world_size = [
            self.world_size[0] * (rect_size[0] + rect_margin),
            self.world_size[1] * (rect_size[1] + rect_margin)]

        self.screen = self._initialize_pygame(self.world_size)
        self.screen.fill(self.background)


    def _initialize_pygame(self, world_size):
        # PyGame 초기화
        # num_pass: 초기화가 성공한 모듈 갯수
        # num_fail: 초기화가 실패한 모듈 갯수
        num_pass, num_fail = pygame.init()
        print("[PyGame Initialize]: Pass {:d}, Fail {:d}".format(num_pass, num_fail))
        
        screen = pygame.display.set_mode(world_size)
        return screen

    def _load_map(self, map_path):
        with open(map_path) as f:
            map_data = [ list(row.strip()) for row in f.readlines() ]
            
            START, GOAL = [], []
            for (y, x), val in np.ndenumerate(map_data):
                if val == "S":
                    START.append([x, y])
                elif val == "G":
                    GOAL.append([x, y])

            return map_data, START, GOAL

    def _define_color(self):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

    def _define_actions(self):
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.ACTIONS = [ self.UP, self.DOWN, self.LEFT, self.RIGHT ]
        
    def _in_range(self, x, y):
        map_height, map_width = self.map_size[0], self.map_size[1]
        if 0 <= x < map_width and 0 <= y < map_height:
            return self.map[y][x] != "#"
        return False


    def step(self, agent, action):
        x, y = agent.here
        if action == self.UP:
            agent.there = [x, y - 1]
        elif action == self.DOWN:
            agent.there = [x, y + 1]
        elif action == self.LEFT:
            agent.there = [x - 1, y]
        elif action == self.RIGHT:
            agent.there = [x + 1, y]

        if self._in_range(agent.there[0], agent.there[1]):
            agent.here = agent.there

        return agent

    def render(self, agent):
        for (y, x), cell in np.ndenumerate(self.map):
            color = self.WHITE

            if cell == "#":
                continue
            elif cell == "G":
                color = self.GREEN
            elif cell == "S":
                color = self.RED

            pygame.draw.rect(
                self.screen, color,
                [(self.RECT_MARGIN + self.RECT_WIDTH) * x + self.RECT_MARGIN,
                (self.RECT_MARGIN + self.RECT_HEIGHT) * y + self.RECT_MARGIN,
                self.RECT_WIDTH, self.RECT_HEIGHT]
            )

        agent_x, agent_y = agent.here
        pygame.draw.rect(
            self.screen, self.BLUE,
            [(self.RECT_MARGIN + self.RECT_WIDTH) * agent_x + self.RECT_MARGIN,
            (self.RECT_MARGIN + self.RECT_HEIGHT) * agent_y + self.RECT_MARGIN,
            self.RECT_WIDTH, self.RECT_HEIGHT]
        )

        pygame.display.flip()