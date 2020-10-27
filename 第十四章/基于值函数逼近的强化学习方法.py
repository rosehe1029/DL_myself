#!/usr/bin/env python
# -*- coding:utf-8 -*-

# import gym
import random
import numpy as np
import tensorflow as tf


class GriDMdp:
    def __init__(s):
        s.gamma = 0.9
        s.alpha = 0.3
        s.epsilon = 0.1
        s.states = range(1, 26)
        s.actions = ['n', 'e', 's', 'w']
        s.terminate_states = {15: 1.0, 4: -1.0, 9: -1.0, \
                              11: -1.0, 12: -1.0, 23: -1.0, 24: -1.0, 25: -1.0}
        s.trans = {}
        for state in s.states:
            if not state in s.terminate_states:
                s.trans[state] = {}
        s.trans[1]['e'] = 2
        s.trans[1]['s'] = 6
        s.trans[2]['e'] = 3
        s.trans[2]['w'] = 1
        s.trans[2]['s'] = 7
        s.trans[3]['e'] = 4
        s.trans[3]['w'] = 2
        s.trans[3]['s'] = 8
        s.trans[5]['w'] = 4
        s.trans[5]['s'] = 10
        s.trans[6]['e'] = 7
        s.trans[6]['s'] = 11
        s.trans[6]['n'] = 1
        s.trans[7]['e'] = 8
        s.trans[7]['w'] = 6
        s.trans[7]['s'] = 12
        s.trans[7]['n'] = 2
        s.trans[8]['e'] = 9
        s.trans[8]['w'] = 7
        s.trans[8]['s'] = 13
        s.trans[8]['n'] = 3
        s.trans[10]['w'] = 9
        s.trans[10]['s'] = 15
        s.trans[13]['e'] = 14
        s.trans[13]['w'] = 12
        s.trans[13]['s'] = 18
        s.trans[13]['n'] = 8
        s.trans[14]['e'] = 15
        s.trans[14]['w'] = 13
        s.trans[14]['s'] = 19
        s.trans[14]['n'] = 9
        s.trans[16]['e'] = 17
        s.trans[16]['s'] = 21
        s.trans[16]['n'] = 11
        s.trans[17]['e'] = 18
        s.trans[17]['w'] = 16
        s.trans[17]['s'] = 22
        s.trans[17]['n'] = 12
        s.trans[18]['e'] = 19
        s.trans[18]['w'] = 17
        s.trans[18]['s'] = 23
        s.trans[18]['n'] = 13
        s.trans[19]['e'] = 20
        s.trans[19]['w'] = 18
        s.trans[19]['s'] = 24
        s.trans[19]['n'] = 14
        s.trans[20]['w'] = 19
        s.trans[20]['s'] = 25
        s.trans[20]['n'] = 15
        s.trans[21]['e'] = 22
        s.trans[21]['n'] = 16
        s.trans[22]['e'] = 23
        s.trans[22]['w'] = 21
        s.trans[22]['n'] = 17

        s.rewards = {}
        for state in s.states:
            s.rewards[state] = {}
            for action in s.actions:
                s.rewards[state][action] = 0
                if state in s.trans and action in s.trans[state]:
                    next_state = s.trans[state][action]
                    if next_state in s.terminate_states:
                        s.rewards[state][action] = s.terminate_states[next_state]
        s.pi = {}
        for state in s.trans:
            s.pi[state] = random.choice(s.trans[state].keys())
        s.last_pi = s.pi.copy()

        s.v = {}
        for state in s.states:
            s.v[state] = 0.0

    def get_random_action(s, state):
        s.pi[state] = random.choice(s.trans[state].keys())
        return s.pi[state]

    def transform(s, state, action):
        next_state = state
        state_reward = 0
        is_terminate = True
        return_info = {}

        if state in s.terminate_states:
            return next_state, state_reward, is_terminate, return_info
        if state in s.trans:
            if action in s.trans[state]:
                next_state = s.trans[state][action]
        if state in s.rewards:
            if action in s.rewards[state]:
                state_reward = s.rewards[state][action]
        if not next_state in s.terminate_states:
            is_terminate = False
        return next_state, state_reward, is_terminate, return_info

    def print_states(s):
        for state in s.states:
            if state in s.terminate_states:
                print
                "*",
            else:
                print
                round(s.v[state], 2),
            if state % 5 == 0:
                print
                "|"

    def get_features(s, state):
        featrues = [0.0] * 25
        featrues[state - 1] = 1.0
        return featrues


def td_Qlearning_linear_approximation(grid_mdp):
    '''action_strategy is greey'''
    # construct model
    x_ph = tf.placeholder(tf.float32, shape=[None, 25], name="input_name")
    y_ph = tf.placeholder(tf.float32, shape=[None, 4], name="output_name")
    # w = tf.Variable(tf.random_uniform([25,4], -1.0, 1.0))
    w = tf.Variable(tf.zeros([25, 4]))
    b = tf.Variable(tf.zeros([4]))
    y = tf.matmul(x_ph, w) + b
    loss = tf.reduce_mean(tf.square(y - y_ph))
    optimizer = tf.train.GradientDescentOptimizer(0.03)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    action_dic = {'e': 0, 'w': 1, 's': 2, 'n': 3}
    total_loss = 0.0
    for iter_idx in range(1, 20000):
        # print "-----------------------"
        one_sample_list = []
        state = random.choice(grid_mdp.states)
        while (state in grid_mdp.terminate_states):
            state = random.choice(grid_mdp.states)
        sample_end = False
        while sample_end != True:
            # choose epsilon_greey strategy
            action_list = grid_mdp.trans[state].keys()
            len_action = len(action_list)
            action_prob = [grid_mdp.epsilon / float(len_action)] * len_action
            input_features = grid_mdp.get_features(state)
            pred_state_action_value = sess.run(y, feed_dict={x_ph: [input_features]})
            max_idx = 0
            max_val = float("-inf")
            max_aidx = 0
            for aidx in range(len_action):
                act_idx = action_dic[action_list[aidx]]
                tmp_value = pred_state_action_value[0, act_idx]
                if tmp_value > max_val:
                    max_val = tmp_value
                    max_idx = aidx
                    max_aidx = act_idx
            action_prob[max_idx] += (1.0 - grid_mdp.epsilon)
            # action-strategy choose epsilon_greey strategy
            action = np.random.choice(action_list, p=action_prob)
            next_state, state_reward, is_terminate, return_info = grid_mdp.transform(state, action)
            # target-strategy choose greey strategy
            real_y = pred_state_action_value
            if next_state in grid_mdp.trans:
                next_action_list = grid_mdp.trans[next_state].keys()
                len_next_action = len(next_action_list)
                next_action_prob = [grid_mdp.epsilon / float(len_next_action)] * len_next_action
                next_input_features = grid_mdp.get_features(next_state)
                next_pred_state_action_value = sess.run(y, feed_dict={x_ph: [next_input_features]})
                next_max_idx = 0
                next_max_val = float("-inf")
                next_max_aidx = 0
                for next_aidx in range(len_next_action):
                    next_act_idx = action_dic[next_action_list[next_aidx]]
                    next_tmp_value = next_pred_state_action_value[0, next_act_idx]
                    if next_tmp_value > next_max_val:
                        next_max_val = next_tmp_value
                        next_max_idx = next_aidx
                        next_max_aidx = next_act_idx
                next_action_idx = next_max_aidx
                difference = state_reward + grid_mdp.gamma * next_pred_state_action_value[0, next_action_idx] - \
                             pred_state_action_value[0, max_aidx]
                real_y[0, max_aidx] += grid_mdp.alpha * difference
            else:
                difference = state_reward - pred_state_action_value[0, max_aidx]
                real_y[0, max_aidx] += grid_mdp.alpha * difference
            # train
            feed_data = {x_ph: [np.array(input_features)], y_ph: real_y}
            sess.run(train, feed_dict=feed_data)
            total_loss += sess.run(loss, feed_data)
            state = next_state
            sample_end = is_terminate

        if iter_idx % 100 == 0:
            print
            "-" * 18 + str(iter_idx) + "-" * 18
            iter_para = 0.01
            # iter_para = 0.01/(float(iter_idx/100)**0.5)
            print
            "total_loss: ", total_loss, "iter_para: ", iter_para
            total_loss = 0.0
            # optimizer = tf.train.GradientDescentOptimizer(iter_para)
            for state in grid_mdp.trans:
                input_features = grid_mdp.get_features(state)
                pred_state_action_value = sess.run(y, feed_dict={x_ph: [input_features]})
                max_idx = np.argwhere(pred_state_action_value[0,] == pred_state_action_value[0,].max())[0, 0]
                for action in action_dic:
                    if action_dic[action] == max_idx:
                        print
                        state, action, pred_state_action_value
    sess.close()


grid_mdp = GriDMdp()
td_Qlearning_linear_approximation(grid_mdp)

