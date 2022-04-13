import gym
import random
import numpy as np
from QNetwork import QNetwork
from ExperienceBuffer import ExperienceBuffer
import tensorflow.compat.v1 as tf
import os

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

tf.disable_v2_behavior()


class Agent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.gamma = 0.98
        self.ep = tf.Variable(1.0, name='ep')
        self.replay_buffer = ExperienceBuffer(length=10000)
        self.model = -1
        self.episode = tf.Variable(-1.0, name='episode')

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        if random.random() < self.sess.run(self.ep):
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(q_state)
        return action

    def train(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(500)
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_next_states[dones] = np.zeros([self.action_size])  # sets q_next_state to 0 if done
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
        self.q_network.update_model(self.sess, states, actions, q_targets)

        if done:
            self.sess.run(self.ep.assign(tf.math.maximum(0.01, tf.multiply(.99, self.ep))))

    def save_model(self, ep):
        self.sess.run(self.episode.assign(ep))
        save_path = 'models/' + self.get_model_dir() + 'model'
        print('Saving to: models/' + self.get_model_dir())
        self.saver.save(self.sess, save_path, global_step=ep)

    def load_model(self, model=-1):
        self.model = model
        model_dir = 'models/' + self.get_model_dir()
        try:
            files = [f for f in os.listdir(model_dir) if f.endswith('.meta')]
            paths = [os.path.join(model_dir, basename) for basename in files]
            meta_file = max(paths, key=os.path.getctime)
            print('Loading: ' + meta_file)
            self.saver = tf.train.import_meta_graph(meta_file)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        except (ValueError, FileNotFoundError) as e:
            print('Loading: No model loaded.')

    def get_model_dir(self):
        # If model number specified, use that number
        # Else, create new model directory
        # Extract all numbers of current saved models
        if self.model == -1:
            model_dirs = [f for f in os.listdir('models/')]
            model_nums = []
            for model_dir in model_dirs:
                model_nums.append(int(model_dir.split('_')[1]))
            self.model = max(model_nums) + 1
        return 'model_' + str(self.model) + '/'

    def get_ep(self):
        return self.sess.run(self.ep)

    def get_episode(self):
        return self.sess.run(self.episode).astype(int)

    def __del__(self):
        self.sess.close()