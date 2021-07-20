import numpy as np
import torch
import torch.nn as nn

from typing import Union
from torch.distributions import Categorical

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from agent.greedy.greedy_agent import greedy_snake
from types import SimpleNamespace as SN
import yaml
import os
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def manhattan(x,y,bean_x,bean_y,width,height):
    if abs(x-bean_x)>abs(width - abs(x - bean_x)):
        d_x=abs(width - abs(x - bean_x))
        if x>bean_x:
            ind_x=bean_x+width
        else:
            ind_x=bean_x-width
    else:
        d_x=abs(x-bean_x)
        ind_x=bean_x

    if abs(y-bean_y)>abs(height - abs(y - bean_y)):
        d_y=abs(height - abs(y - bean_y))
        if y>bean_y:
            ind_y=bean_y+height
        else:
            ind_y=bean_y-height
    else:
        d_y=abs(y-bean_y)
        ind_y=bean_y
    return d_x+d_y,ind_x,ind_y

#描述bean的聚集程度，返回一个list
def aggregation(beans_position, width, height):
    result=[]
    for i in range(len(beans_position)):
        aggre=0.0
        x=beans_position[i][1]
        y=beans_position[i][0]
        for j in range (len(beans_position)):
            if i==j:
                continue
            bean_x=beans_position[j][1]
            bean_y=beans_position[j][0]
            aggre+=1/(manhattan(x,y,bean_x,bean_y,width,height)[0])
        result.append(aggre)
    return result

#以(x, y)点为起点进行BFS搜索，返回region大小。
#Note that:为保证速度，大于15将停止搜索
def get_region_size(state, width, height, x, y):
    count = 0.0
    stack=[]
    visited = np.zeros((height,width))
    if (state[y][x]<=1) and (visited[y][x]==0):
        visited[y][x] = 1
        stack.append([y,x])
        count+=1
    while (len(stack)!=0) and (count<=15):
        y_prime,x_prime = stack.pop()
        y = (y_prime - 1) % height
        x = x_prime
        if (state[y][x]<=1) and (visited[y][x]==0):
            stack.append([y,x])
            visited[y][x]=1
            count+=1

        y = (y_prime + 1) % height
        x = x_prime
        if (state[y][x]<=1) and (visited[y][x]==0):
            stack.append([y,x])
            visited[y][x]=1
            count+=1

        y = y_prime
        x = (x_prime - 1) % width
        if (state[y][x]<=1) and (visited[y][x]==0):
            stack.append([y,x])
            visited[y][x]=1
            count+=1

        y = y_prime
        x = (x_prime + 1) % width
        if (state[y][x]<=1) and (visited[y][x]==0):
            stack.append([y,x])
            visited[y][x]=1
            count+=1

    return count
def bean_region_dis(state, width, height, beans_position):
    result=[]
    for bean in beans_position:
        count=get_region_size(state,width,height,bean[1],bean[0])
        if count==0:
            result.append(10000)
        elif count<=4:
            result.append(1000/np.sqrt(count))
        elif count<=10:
            result.append(1/count**0.9)
        else:
            result.append(0)
    # print(result)
    return result

#分析对手最近两个bean
def opponent_bean(beans_position, width, height, mysnake, opsnake):
    mylength=len(mysnake)
    oplength=len(opsnake)
    op_head_x=opsnake[0][1]
    op_head_y=opsnake[0][0]
    my_head_x=mysnake[0][1]
    my_head_y=mysnake[0][0]

    mindis=10000
    result=[0.0]*len(beans_position)
    index=0
    i=-1
    for bean in beans_position:
        i+=1
        dis,tmp,tmp =manhattan(op_head_x, op_head_y, bean[1], bean[0], width, height)
        if dis<mindis:
            mindis=dis
            index=i
    bean=beans_position[index]
    mydis,tmp,tmp =manhattan(my_head_x, my_head_y, bean[1], bean[0], width, height)
    if mydis>mindis:
        result[index]=6/mindis
    elif mydis<mindis:
        result[index]=-1/mydis
    else:
        if mylength<=oplength:
            result[index]=-1/mydis
        elif mindis==1:
            result[index]=100
        else:
            result[index]=6/mydis
    return result,mindis

def get_dis(x, y, beans_position, width, height, state, snakes, ctrl_agent_index):
    min_distance = 10000
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    mysnake=snakes[ctrl_agent_index[0]]
    opsnake=snakes[1-ctrl_agent_index[0]]

    para=[0.1, 0.25, 1]
    aggre=aggregation(beans_position, width, height)
    region=bean_region_dis(state,width,height,beans_position)
    oppo=opponent_bean(beans_position, width, height, mysnake, opsnake)[0]
    dist = list(np.zeros(5).tolist())
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance_manhattan,ind_x,ind_y = manhattan(x,y,bean_x,bean_y,width,height)
        dis = distance_manhattan + para[2] * oppo[i]
        dist[i] = dis
    return dist


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(tgt_param.data * (1.0 - tau) + src_param.data * tau)


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)


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

    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y, i, info)
        observations[i][2:146] = head_surrounding[:]

        # beans positions
        observations[i][146:156] = beans_position[:]

        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[i][0]
        observations[i][156:158] = snake_heads.flatten()[:]

        # length
        other_index = 0 if agents_index[0] == 1 else 1
        observations[i][158] = len(info['snakes_position'][agents_index[0]])
        observations[i][159] = len(info['snakes_position'][other_index])

        # tail
        # observation[]

    return observations


def get_reward(pre_state, state, pre_info, info, snake_index, reward, final_result):
    state = np.array(state)
    state = np.squeeze(state, axis=2)
    pre_state = np.array(pre_state)
    pre_state = np.squeeze(pre_state, axis=2)
    step_reward = np.zeros(len(snake_index), dtype=float)
    t = 0.001
    for i in snake_index:
        for i in snake_index:
            if final_result == 1:  # done and won
                step_reward[i] += ((4000 + len(info['snakes_position'][snake_index[0]]) * 1000) * 2 * t - (
                            2000 + len(info['snakes_position'][1]) * 500) * 2 * t)
            elif final_result == 2:  # done and lose
                step_reward[i] -= ((2500 + len(info['snakes_position'][1]) * 500) * 2 * t - (
                            1500 + len(info['snakes_position'][snake_index[0]]) * 500) * 2 * t)
            elif final_result == 3:     # done and draw
                step_reward[i] -= 4000*t
        else:                       # not done
            if reward[i]:                                 # eat a bean
                step_reward[i] += 1600*t                      # just move
            else:
                snakes_position = np.array(info['snakes_position'], dtype=object)
                pre_snakes = np.array(pre_info['snakes_position'], dtype=object)
                beans_position = np.array(info['beans_position'], dtype=object)
                pre_beans = np.array(pre_info['beans_position'], dtype=object)
                snake_heads = [snake[0] for snake in snakes_position]
                pre_heads = [snake[0] for snake in pre_snakes]
                self_head = np.array(snake_heads[i])
                pre_head = np.array(pre_heads[i])
                dists = get_dis(self_head[1], self_head[0], beans_position, 8, 6, state, snakes_position, snake_index)
                pre_dists = get_dis(pre_head[1], pre_head[0], pre_beans, 8, 6, pre_state, pre_snakes, snake_index)
                # dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
                # pre_dists = [np.sqrt(np.sum(np.square(other_head - pre_head))) for other_head in pre_beans]
                step_reward[i] += (min(pre_dists)-min(dists))*800*t
            if len(pre_info['snakes_position'][snake_index[0]]) > len(info['snakes_position'][snake_index[0]]):
                step_reward[i] -= max(3000*t, (len(pre_info['snakes_position'][snake_index[0]]) - len(info['snakes_position'][snake_index[0]]))*1500*t)
            if len(pre_info['snakes_position'][1-snake_index[0]]) > len(info['snakes_position'][1-snake_index[0]]):
                step_reward[i] += max(3000 * t, (len(pre_info['snakes_position'][1-snake_index[0]]) - len(info['snakes_position'][1-snake_index[0]])) * 1200 * t)
            # if reward[i] < 0:
            #     step_reward[i] -= 20*t
            # other_index = 0 if snake_index[0] == 1 else 1
            # step_reward[i] += (len(info['snakes_position'][snake_index[0]])-len(info['snakes_position'][other_index]))*100*t
    # print(f"steprew{step_reward}")
    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def append_random(act_dim, action):
    action = torch.Tensor([action]).to(device)
    acs = [out for out in action]
    num_agents = len(action)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions

def logits_greedy(state, info, logits, height, width):
    state = np.squeeze(np.array(state), axis=2)
    beans = info['beans_position']
    snakes = info['snakes_position']

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])
    greedy_action = greedy_snake(state, beans, snakes, width, height, [1])

    action_list = np.zeros(2)
    action_list[0] = logits_action[0]
    action_list[1] = greedy_action[0]

    return action_list


def get_surrounding(state, width, height, x, y, ctrl_agent, info):
    state = state.copy()
    state[state==ctrl_agent+2] = 4 # 是自己的身子
    state[state==3-ctrl_agent] = 5 # 是对手的身子
    state[info['snakes_position'][ctrl_agent][0][0]][info['snakes_position'][ctrl_agent][0][1]] = 2 # 是自己的头
    state[info['snakes_position'][1-ctrl_agent][0][0]][info['snakes_position'][1-ctrl_agent][0][1]] = 3 # 是对手的头


    surrounding = np.zeros((24,6))

    surrounding[0][state[(y - 2) % height][x]] = 1  # upup
    surrounding[1][state[(y + 2) % height][x]] = 1
    surrounding[2][state[y][(x - 2) % width]] = 1
    surrounding[3][state[y][(x + 2) % width]] = 1
    surrounding[4][state[(y - 1) % height][(x - 1) % width]] = 1
    surrounding[5][state[(y - 1) % height][x]] = 1
    surrounding[6][state[(y - 1) % height][(x + 1) % width]] = 1
    surrounding[7][state[y][(x - 1) % width]] = 1
    surrounding[8][state[y][(x + 1) % width]] = 1
    surrounding[9][state[(y + 1) % height][(x - 1) % width]] = 1
    surrounding[10][state[(y + 1) % height][x]] = 1
    surrounding[11][state[(y + 1) % height][(x + 1) % width]] = 1
    surrounding[12][state[(y - 3) % height][x]] = 1
    surrounding[13][state[(y + 3) % height][x]] = 1
    surrounding[14][state[y][(x - 3) % width]] = 1
    surrounding[15][state[y][(x + 3) % width]] = 1
    surrounding[16][state[(y - 2) % height][(x - 1) % width]] = 1
    surrounding[17][state[(y - 2) % height][(x + 1) % width]] = 1
    surrounding[18][state[(y + 2) % height][(x - 1) % width]] = 1
    surrounding[19][state[(y + 2) % height][(x + 1) % width]] = 1
    surrounding[20][state[(y - 1) % height][(x - 2) % width]] = 1
    surrounding[21][state[(y - 1) % height][(x + 2) % width]] = 1
    surrounding[22][state[(y + 1) % height][(x - 2) % width]] = 1
    surrounding[23][state[(y + 1) % height][(x + 2) % width]] = 1

    surrounding = list(surrounding.flatten().tolist())

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()


def load_config(args, log_path):
    file = open(os.path.join(str(log_path), 'config.yaml'), "r")
    config_dict = yaml.load(file, Loader=yaml.FullLoader)
    print("@", config_dict)
    args = SN(**config_dict)
    print("@@", args)
    return args


# def set_algos():
#     with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
#         try:
#             config_dict = yaml.load(f, Loader=yaml.FullLoader)
#         except yaml.YAMLError as exc:
#             assert False, "default.yaml error: {}".format(exc)
#
#     args = SN(**config_dict)
#     return args


