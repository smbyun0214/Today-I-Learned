import numpy as np
import pygame
from GridWorld import GridWorld
from Agent import Agent

def play(world, agent):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    agent = world.step(agent, world.UP)
                elif event.key == pygame.K_DOWN:
                    agent = world.step(agent, world.DOWN)
                elif event.key == pygame.K_LEFT:
                    agent = world.step(agent, world.LEFT)
                elif event.key == pygame.K_RIGHT:
                    agent = world.step(agent, world.RIGHT)

        world.render(agent)
            
                
if __name__ == "__main__":
    rect_size, rect_margin = [20, 20], 2
    world = GridWorld(rect_size, rect_margin)
    agent = Agent(world.START[np.random.choice(len(world.START))])

    play(world, agent)
