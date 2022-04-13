from Agent import Agent
import gym
from ShipWorld import ShipWorld
import pygame as pg
import numpy as np

# Initialize
pg.init()

# Create environment
env_size = (1000, 1000)
start = (1,400,0)
env = ShipWorld(env_size, start)
state = env.reset()
agent = Agent(env)
max_episodes = 50000
env.set_simulation_speed(40)
steps_per_second = env.step_per_second  # timing of each step
render = True
env.set_render(render)  # whether or not to render training
overtime = 20 * steps_per_second  # maximum steps per episode

# Load saved model
agent.load_model(2)
start_episode = agent.get_episode() + 1

# Obstacles
obs1 = ["rect", [(0, 0), (400, 400)]]
obs2 = ["circ", [(250, 500), 50]]
env.add_obstacle()

for episode in range(start_episode, max_episodes):
    # Reset env each episode
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    # Save rewards
    rewards = []

    # Run episode until WP reached or over runtime
    while not done:
        events = pg.event.get()  # for pygame window manipulation
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_y:
                    env.set_render(True)
                if event.key == pg.K_n:
                    env.set_render(False)

        # Timing control
        step_start = pg.time.get_ticks()

        # Step forward
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        step = step + 1

        # Overtime condition
        if step > overtime:
            done = True

        # Train agent, update total reward, and environment
        agent.train(state, action, next_state, reward, done)
        total_reward = total_reward + reward
        state = next_state

        # Timing control
        step_end = pg.time.get_ticks()
        # print(step_end-step_start)
        if render:
            pg.time.wait(int((1000 / steps_per_second / env.multiplier - (step_end - step_start))))

    # Save reward each episode
    rewards.append(total_reward)

    # Console log
    print("Episode: {}, Total reward: {:.2f}, eps: {:.2f}".format(episode, total_reward, agent.get_ep()))

    # Periodically save model
    if episode % 100 == 0 and episode > 0:
        agent.save_model(episode)

