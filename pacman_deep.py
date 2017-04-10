#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

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
from ReplayMemory import *
from DeepNetwork import *


# game parameters
gym_game = 'MsPacman-v0'
game_resolution = (70, 40)
img_channels = 1

load_model = False
model_learn = True

if (model_learn):
    save_model = True
    save_log = True
    skip_learning = False
else:
    save_model = False
    save_log = False
    skip_learning = True    

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
discount_factor = 0.99
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# training regime
num_epochs = 30
learning_steps_per_epoch = 25000
test_episodes_per_epoch = 15
episodes_to_watch = 5


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

def perform_learning_step(epoch):
    
    def exploration_rate(epoch):
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * num_epochs
        eps_decay_epochs = 0.6 * num_epochs

        if load_model:
            return end_eps
        else:
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
        a = randint(0, num_actions - 1)
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

    memory = ReplayMemory(capacity=replay_memory_size, game_resolution=game_resolution, num_channels=img_channels)

    sess = tf.Session()
    learn, get_q_values, get_best_action = create_network(sess, num_actions, game_resolution, img_channels)
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
            observation = env.reset()
            dropout_keep_prob = 0.8
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
            dropout_keep_prob = 1.0
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

    dropout_keep_prob = 1.0
    
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
