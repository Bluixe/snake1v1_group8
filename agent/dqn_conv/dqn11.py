# -*- coding:utf-8  -*-
# Time  : 2021/5/27 下午3:38
# Author: Yahui Cui

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    width = observation_list[0]['board_width']
    height = observation_list[0]['board_height']
    c = observation_list[0]['controlled_snake_index']
    state = np.zeros((height, width))
    beans = observation_list[0][1]
    snakes = []
    snakes.append(observation_list[0][2])
    snakes.append(observation_list[0][3])
    for i in beans:
        state[i[0], i[1]] = 1
    for i in snakes[0]:
        state[i[0], i[1]] = 2
    for i in snakes[1]:
        state[i[0], i[1]] = 3
    agent = DQN(20, 4, 1, 256)
    agent.load('critic_3000.pth')
    info = {'snake_position': snakes, 'beans_position': beans, 'directions': observation_list['last_direction']}
    obs = get_observations(state, info, [0, c], 18, height, width)

    actions = []
    actions[:] = agent.choose_action(obs)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action

def get_surrounding(state, ctrl_agent, info, x, y):
    state = state.copy()
    state[state==ctrl_agent+2] = 4 # 是自己的身子
    state[state==3-ctrl_agent] = 5 # 是对手的身子
    # state[info['snakes_position'][ctrl_agent][0][0]][info['snakes_position'][ctrl_agent][0][1]] = 2 # 是自己的头
    # state[info['snakes_position'][1-ctrl_agent][0][0]][info['snakes_position'][1-ctrl_agent][0][1]] = 3 # 是对手的头
    # state[info['snakes_position'][ctrl_agent][-1][0]][info['snakes_position'][ctrl_agent][-1][1]] = 6  # 是自己的尾
    # state[info['snakes_position'][1-ctrl_agent][-1][0]][info['snakes_position'][1-ctrl_agent][-1][1]] = 7 # 是对手的尾


    surrounding = np.zeros((1, 6, 6, 8))
    # for i in range(8):
        # surrounding[0][i][state==i] = 1
    for i in range(6):
        for j in range(8):
            surrounding[0][state[(y+i)%6][(x+j)%8]][i][j] = 1
    surrounding[0][2][0][0] = len(info['snakes_position'][ctrl_agent])
    surrounding[0][2][(info['snakes_position'][1-ctrl_agent][0][0]-info['snakes_position'][ctrl_agent][0][0])%6][
        (info['snakes_position'][1-ctrl_agent][0][1]-info['snakes_position'][ctrl_agent][0][1])%8] = -len(info['snakes_position'][1-ctrl_agent])
    surrounding[0][3][(info['snakes_position'][ctrl_agent][-1][0] - info['snakes_position'][ctrl_agent][0][0]) % 6][
        (info['snakes_position'][ctrl_agent][-1][1] - info['snakes_position'][ctrl_agent][0][1]) % 8] = len(info['snakes_position'][ctrl_agent])
    surrounding[0][3][(info['snakes_position'][1 - ctrl_agent][-1][0] - info['snakes_position'][ctrl_agent][0][0]) % 6][
        (info['snakes_position'][1 - ctrl_agent][-1][1] - info['snakes_position'][ctrl_agent][0][1]) % 8] = -len(info['snakes_position'][1-ctrl_agent])

    # print(surrounding)
    return surrounding



# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) -- (other_x - self_x, other_y - self_y)
def get_observations(state, info, agents_index, obs_dim, height, width):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    observations = np.zeros((len(agents_index), obs_dim))
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position']).flatten()
    head_x = snakes_position[agents_index[0]][0][1]
    head_y = snakes_position[agents_index[0]][0][0]
    obs = []
    for i in agents_index:
        obs.append(get_surrounding(state, i, info, head_x, head_y))

    return obs


class Critic(nn.Module):
    def __init__(self, in_channels, input_size, output_size):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.output_size = output_size
        self.conv1 = nn.Conv2d(in_channels, 8, (7,7), padding=(3,3), padding_mode='circular')
        self.conv2 = nn.Conv2d(8, 10, (5,5), padding=(2,2), padding_mode='circular')
        self.conv3 = nn.Conv2d(10, 10, (3,3), padding=(1,1), padding_mode='circular')
        self.conv4 = nn.Conv2d(10, 6, (3,3), padding=(1,1), padding_mode='circular')
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, output_size)


    def forward(self, x):
        x = torch.tensor(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(len(x), 1, -1)
        # print(x)
        x = F.sigmoid(self.linear1(x))

        # print(x.shape)
        x = self.linear2(x)
        return x



class DQN(object):
    def __init__(self, state_dim, action_dim, num_agent, in_channels):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agent = num_agent

        self.in_channels = in_channels

        self.critic_eval = Critic(self.in_channels, self.state_dim, self.action_dim)
        self.critic_target = Critic(self.in_channels, self.state_dim, self.action_dim)

    def choose_action(self, observation):
        #print(observation)
        observation = torch.tensor(observation, dtype=torch.float)
        #print(observation)
        action = torch.argmax(self.critic_eval(observation)).item()
        return action

    def load(self, file):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(base_path, file)
        self.critic_eval.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
        self.critic_target.load_state_dict(torch.load(file, map_location=torch.device('cpu')))


def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action

agent = DQN(288, 4, 1, 6)
agent.load('it11/critic_40000.pth')

# 110 120 130 140 150 160 170 180 190 200 210  50
# 417 410 361 385 1561 1505 1485 343             422
# 460 479 527 512 2970 3003 3071 540

# 90
# 301
# 574