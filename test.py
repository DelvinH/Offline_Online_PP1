from ShipWorld import ShipWorld
#from QNAgent import QNAgent
#import tensorflow.compat.v1 as tf
import pygame as pg
import time
import os
import numpy as np





#tf.disable_v2_behavior()

action = 1
size = (1000, 1000)
env = ShipWorld(size, (0, 500, 0))
#qnagent = QNAgent(env)
env.set_simulation_speed(1)
steps_per_second = env.step_per_second  # timing of each step

obs1 = ["rect", [(0, 0), (400, 400)]]
obs2 = ["polygon", [(1000, 800), (800, 1000), (1000, 1000)]]
obs3 = ['circle', [(600,600), 50]]
env.add_obstacle(obs1, obs2, obs3)

total_reward = 0
for ep in range(100):
    state = env.reset()
    env.draw_waypoint()
    done = False
    total_reward = 0

    a = pg.time.get_ticks()
    while not done:
        step_start = pg.time.get_ticks()
        events = pg.event.get()
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_1:
                    action = 0
                if event.key == pg.K_2:
                    action = 1
                if event.key == pg.K_3:
                    action = 2
                if event.key == pg.K_4:
                    action = 3
                if event.key == pg.K_5:
                    action = 4
                if event.key == pg.K_6:
                    action = 5
                if event.key == pg.K_7:
                    action = 6
            if event.type == pg.KEYUP:
                action = 1

        #action = qnagent.get_action(state)
        next_state, reward, done, info = env.step(action)
        #qnagent.train((state, action, next_state, reward, done))
        state = next_state
        #print(reward)
        total_reward = total_reward + reward
        #print(total_reward)
        step_end = pg.time.get_ticks()
        pg.time.wait(int((1000 / steps_per_second / env.multiplier - (step_end - step_start))))
        step_final = pg.time.get_ticks()
        #print(step_final - step_start)
        b = pg.time.get_ticks()
        #print((b-a)/1000)

    print("Episode: ", ep, ", Total reward: ", total_reward)

        #print('s:', state, ', a:', env.translate_action(action))
        #print("Episode: {}, Total reward: {}, eps: {}".format(ep, total_reward, qnagent.e))
        #env.render()
        #with tf.variable_scope("q_table", reuse=True):
        #    weights = qnagent.sess.run(tf.get_variable("kernel"))
        #    print(weights)

        # optimal = [0] * qnagent.state_size
        # for s in range(qnagent.state_size):
        #     action = np.argmax(weights[s])
        #     action = env.translate_action(action)
        #     optimal[s] = action
        # print(np.reshape(optimal, (size[1], size[0])))
        # print(qnagent.q_table)
        #print('Timestep:', pause)
        #time.sleep(pause)





