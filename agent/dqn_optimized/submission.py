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
    agent = DQN(136, 4, 1, 512)
    agent.load('critic_16000.pth')
    info = {'snakes_position': [observation_list[0][2], observation_list[0][3]], 'beans_position': beans,
            'directions': observation_list[0]['last_direction']}
    obs = get_observations(state, info, [0, 1], 136, height, width)

    actions = []
    actions[:] = [agent.choose_action(obs[c])]
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action

def get_surrounding(state, width, height, x, y, ctrl_agent, info):
    state = state.copy()
    state[state>=2] = 2 # 是身子
    state[info['snakes_position'][ctrl_agent][0][0]][info['snakes_position'][ctrl_agent][0][1]] = 3 # 是自己的头
    state[info['snakes_position'][1-ctrl_agent][0][0]][info['snakes_position'][1-ctrl_agent][0][1]] = 4 # 是对手的头

    state = list((np.array(state).astype(np.int)).tolist())
    surrounding = np.zeros((24,5))
    print(state)

    surrounding[0][int(state[(y - 2) % height][x])] = 1  # upup
    surrounding[1][int(state[(y + 2) % height][x])] = 1
    surrounding[2][int(state[y][(x - 2) % width])] = 1
    surrounding[3][int(state[y][(x + 2) % width])] = 1
    surrounding[4][int(state[(y - 1) % height][(x - 1) % width])] = 1
    surrounding[5][int(state[(y - 1) % height][x])] = 1
    surrounding[6][int(state[(y - 1) % height][(x + 1) % width])] = 1
    surrounding[7][int(state[y][(x - 1) % width])] = 1
    surrounding[8][int(state[y][(x + 1) % width])] = 1
    surrounding[9][int(state[(y + 1) % height][(x - 1) % width])] = 1
    surrounding[10][int(state[(y + 1) % height][x])] = 1
    surrounding[11][int(state[(y + 1) % height][(x + 1) % width])] = 1
    surrounding[12][int(state[(y - 3) % height][x])] = 1
    surrounding[13][int(state[(y + 3) % height][x])] = 1
    surrounding[14][int(state[y][(x - 3) % width])] = 1
    surrounding[15][int(state[y][(x + 3) % width])] = 1
    surrounding[16][state[(y - 2) % height][(x - 1) % width]] = 1
    surrounding[17][state[(y - 2) % height][(x + 1) % width]] = 1
    surrounding[18][state[(y + 2) % height][(x - 1) % width]] = 1
    surrounding[19][state[(y + 2) % height][(x + 1) % width]] = 1
    surrounding[20][state[(y - 1) % height][(x - 2) % width]] = 1
    surrounding[21][state[(y - 1) % height][(x + 2) % width]] = 1
    surrounding[22][state[(y + 1) % height][(x - 2) % width]] = 1
    surrounding[23][state[(y + 1) % height][(x + 2) % width]] = 1

    surrounding = list(surrounding.flatten().tolist())
    # print(surrounding)

    return surrounding


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) -- (other_x - self_x, other_y - self_y)
def get_observations(state, info, agents_index, obs_dim, height, width):
    state = np.array(state)
    # state = np.squeeze(state, axis=2)
    observations = np.zeros((len(agents_index), obs_dim))
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position']).flatten()

    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y, i, info)
        observations[i][2:122] = head_surrounding[:]

        # beans positions
        observations[i][122:132] = beans_position[:]

        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[i][0]
        observations[i][132:134] = snake_heads.flatten()[:]

        # length
        other_index = 0 if agents_index[0] == 1 else 1
        observations[i][134] = len(info['snakes_position'][agents_index[0]])
        observations[i][135] = len(info['snakes_position'][other_index])

    return observations



class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x.shape)
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        # print(x.shape)
        x = self.linear3(x)
        return x


class DQN(object):
    def __init__(self, state_dim, action_dim, num_agent, hidden_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agent = num_agent

        self.hidden_size = hidden_size

        self.critic_eval = Critic(self.state_dim, self.action_dim, self.hidden_size)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_size)

    def choose_action(self, observation):
        #print(observation)
        observation = torch.tensor(observation, dtype=torch.float).view(1, -1)
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

# agent = DQN(136, 4, 1, 512)
# agent.load('it3/critic_1000.pth')

# 110 120 130 140 150 160 170 180 190 200 210  50
# 417 410 361 385 389 364 334 343             422
# 460 479 527 512 507 522 558 540