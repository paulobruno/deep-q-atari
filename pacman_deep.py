#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from random import sample, randint, random
from time import time, sleep
from tqdm import trange

import itertools as it
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import os
import errno
import gym
#from gym import wrappers


# game parameters
gym_game = 'MsPacman-v0'
game_resolution = (105, 80)
img_channels = 1

load_model = False
save_model = True
save_log = True
skip_learning = False

log_savefile = 'log.txt'
model_savefile = 'model.ckpt'

if (gym_game == 'MsPacman-v0'):
    save_path = 'model_mspacman/'
elif (gym_game == 'Breakout-v0'):
    save_path = 'model_breakout/'
# FIXME: games below appear on screen even when env.render() is not called
#elif (gym_game == 'MountainCar-v0'):
#    save_path = 'model_mountain_car/'
#elif (gym_game == 'CartPole-v0'):
#    save_path = 'model_cart_pole/'
else:
    print('ERROR: wrong game.')

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000

# NN architecture
conv_width = 5
conv_height = 5
features_layer1 = 8
features_layer2 = 16
fc_num_outputs = 256

# NN learning settings
batch_size = 64

# training regime
num_epochs = 6
learning_steps_per_epoch = 3000
test_episodes_per_epoch = 15
episodes_to_watch = 5
is_in_training = True


# TODO: get dropout prob in a more elegant way
def get_dropout_keep_prob():
    if is_in_training:
        return 0.8
    else:
        return 1.0

# ceil of a division, source: http://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b):
    return -(-a // b)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def preprocess(image):
    img = skimage.transform.resize(image, game_resolution)
    img = skimage.color.rgb2gray(img) # convert to gray
    img = img.astype(np.float32)
    return img


# TODO: separate classes in files
class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, game_resolution[0], game_resolution[1], img_channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos+1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

# TODO: separate classes in files
def create_network(session, num_available_actions):
    """ creates the network with 
    conv_relu + max_pool + conv_relu + max_pool + fc + dropout + fc """

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    s1_ = tf.placeholder(tf.float32, [None] + list(game_resolution) + [img_channels], name='State')

    target_q_ = tf.placeholder(tf.float32, [None, num_available_actions], name='TargetQ')

    # first convolutional layer
    W_conv1 = weight_variable([conv_height, conv_width, img_channels, features_layer1])
    b_conv1 = bias_variable([features_layer1])

    h_conv1 = tf.nn.relu(conv2d(s1_, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([conv_height, conv_width, features_layer1, features_layer2])
    b_conv2 = bias_variable([features_layer2]) 

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([ceildiv(game_resolution[0],4)*ceildiv(game_resolution[1],4)*features_layer2, fc_num_outputs])
    b_fc1 = bias_variable([fc_num_outputs])

    h_pool2_flat = tf.reshape(h_pool2, [-1, ceildiv(game_resolution[0],4)*ceildiv(game_resolution[1],4)*features_layer2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([fc_num_outputs, num_available_actions])
    b_fc2 = bias_variable([num_available_actions])

    q = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    # TODO: test if softmax gives a better result
    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    train_step = optimizer.minimize(loss)

    def function_learn(s1, target_q):
        feed_dict = {s1_: s1, target_q_: target_q, keep_prob: get_dropout_keep_prob()}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={s1_: state, keep_prob: get_dropout_keep_prob()})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={s1_: state, keep_prob: get_dropout_keep_prob()})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, game_resolution[0], game_resolution[1], 1]))[0]
    
    return function_learn, function_get_q_values, function_simple_get_best_action


def perform_learning_step(epoch):
    
    def exploration_rate(epoch):
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * num_epochs
        eps_decay_epochs = 0.6 * num_epochs

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(env.render('rgb_array'))

    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        a = get_best_action(s1)

    obs, reward, done, info = env.step(a)
    global observation
    observation = obs
    global episode_reward
    episode_reward += reward
    global is_episode_finished
    is_episode_finished = done
    
    isterminal = is_episode_finished

    s2 = preprocess(env.render('rgb_array')) if not isterminal else None

    memory.add_transition(s1, a, s2, isterminal, reward)

    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)

        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1-isterminal) * q2

        learn(s1, target_q)
        

if __name__ == '__main__':
    env = gym.make(gym_game)
    #env = wrappers.Monitor(env, 'tmp/gym-results')

    if save_log:
        make_sure_path_exists(save_path)
        if load_model:        
            log_file = open(save_path+log_savefile, 'a')
        else:
            log_file = open(save_path+log_savefile, 'w')

    num_actions = env.action_space.n
    actions = np.zeros((num_actions, num_actions), dtype=np.int32)
    for i in range(num_actions):
        actions[i, i] = 1
    actions = actions.tolist()

    memory = ReplayMemory(capacity=replay_memory_size)

    sess = tf.Session()
    learn, get_q_values, get_best_action = create_network(sess, len(actions))
    saver = tf.train.Saver()

    if load_model:
        make_sure_path_exists(save_path+model_savefile)
        print('Loading model from: ', save_path+model_savefile)
        saver.restore(sess, save_path+model_savefile)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    time_start = time()

    if not skip_learning:
        print('Starting the training!')

        for epoch in range(num_epochs):
            print('\nEpoch %d\n-------' % (epoch+1))
            train_episodes_finished = 0
            train_scores = []

            print('Training...')
            is_in_training = True
            observation = env.reset()
            is_episode_finished = False
            episode_reward = 0
            
            for learning_step in trange(learning_steps_per_epoch):
                perform_learning_step(epoch)
                if is_episode_finished:
                    train_scores.append(episode_reward)
                    env.reset()
                    is_episode_finished = False
                    episode_reward = 0
                    train_episodes_finished += 1

            print('%d training episodes played.' % train_episodes_finished)
 
            train_scores = np.array(train_scores)

            print('Results: mean: %.1f±%.1f,' % (train_scores.mean(), train_scores.std()), \
                  'min: %.1f,' % train_scores.min(), 'max: %.1f,' % train_scores.max())


            print('\nTesting...')
            is_in_training = False
            test_episode = []
            test_scores = []

            for test_episode in trange(test_episodes_per_epoch):
                observation = env.reset()
                is_episode_finished = False
                episode_reward = 0

                while not is_episode_finished:
                    state = preprocess(env.render('rgb_array'))
                    #env.render()
                    best_action = get_best_action(state)
                    observation, reward, is_episode_finished, info = env.step(best_action)
                    episode_reward += reward
                test_scores.append(episode_reward)

            test_scores = np.array(test_scores)
            print('Results: mean: %.1f±%.1f,' % (test_scores.mean(), test_scores.std()), \
                  'min: %.1f,' % test_scores.min(), 'max: %.1f,' % test_scores.max())
            
            if save_model:
                make_sure_path_exists(save_path+model_savefile)
                print('Saving the nerwork weights to:', save_path+model_savefile)
                saver.save(sess, save_path+model_savefile)

            total_elapsed_time = (time() - time_start) / 60.0
            print('Total elapsed time: %.2f minutes' % total_elapsed_time)

            # log to file
            if save_log:
                print(total_elapsed_time, train_episodes_finished, 
                      train_scores.min(), train_scores.mean(), train_scores.max(), 
                      test_scores.min(), test_scores.mean(), test_scores.max(), file=log_file)
                log_file.flush()

    if save_log:
        log_file.close()

    env.close()
    print('======================================')
    print('Training finished. It\'s time to watch!')

    raw_input('Press Enter to continue...') # in python3 use input() instead

    env = gym.make(gym_game)

    is_in_training = False
    
    for _ in range(episodes_to_watch):
        observation = env.reset()
        is_episode_finished = False
        episode_reward = 0

        while not is_episode_finished:
            env.render()
            state = preprocess(env.render('rgb_array'))
            best_action = get_best_action(state)
            observation, reward, is_episode_finished, info = env.step(best_action)
            episode_reward += reward

        print('Total score: ', episode_reward)

    #gym.upload('tmp/gym-results', api_key='YOUR_API_KEY')'''
