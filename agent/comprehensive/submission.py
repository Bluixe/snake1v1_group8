import copy
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    width = observation_list[0]['board_width']
    height = observation_list[0]['board_height']
    mysnake = observation_list[0]['controlled_snake_index']
    state = np.zeros((height, width))
    beans = observation_list[0][1]
    snakes = [observation_list[0][2], observation_list[0][3]]
    for i in beans:
        state[i[0], i[1]] = 1
    for i in snakes[0]:
        state[i[0], i[1]] = 2
    for i in snakes[1]:
        state[i[0], i[1]] = 3
    info = {'snakes_position': [observation_list[0][2], observation_list[0][3]], 'beans_position': beans,
            'directions': observation_list[0]['last_direction']}
    actions = compre_greedy_defense_for(state, beans, snakes, width, height, [mysnake],info)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action

def get_surrounding_dqn(state, width, height, x, y, ctrl_agent, info):
    state = state.copy()

    state[state>=2] = 2 # 是身子
    state[info['snakes_position'][ctrl_agent][0][0]][info['snakes_position'][ctrl_agent][0][1]] = 3 # 是自己的头
    state[info['snakes_position'][1-ctrl_agent][0][0]][info['snakes_position'][1-ctrl_agent][0][1]] = 4 # 是对手的头


    surrounding = np.zeros((24,5))
    state = list((np.array(state).astype(np.int)).tolist())

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
        head_surrounding = get_surrounding_dqn(state, width, height, head_x, head_y, i, info)
        observations[i][2:122] = head_surrounding[:]

        # beans positions
        observations[i][122:132] = beans_position[:]

        # other snake positions
        snake_heads = [snake[0] for snake in snakes_position]
        snake_heads = np.array(snake_heads[1:])
        snake_heads -= snakes_position[i][0]
        observations[i][132:134] = snake_heads.flatten()[:]

        # length
        observations[i][134] = len(info['snakes_position'][i])
        observations[i][135] = len(info['snakes_position'][1-i])

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
        self.critic_eval.load_state_dict(torch.load(file,map_location=torch.device('cpu')))
        self.critic_target.load_state_dict(torch.load(file,map_location=torch.device('cpu')))

agent = DQN(136, 4, 1, 512)
agent.load('critic_16000.pth')

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right
    return surrounding

#控制蛇移动的核心函数，考虑穿越边界，支持bean坐标为负数
def get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state,mysnake,flag=1):
    next_distances = []
    surrounding=[[(head_y - 1) % height,head_x],[(head_y + 1) % height,head_x],
                 [head_y,(head_x - 1) % width],[head_y,(head_x + 1) % width]]

    up_distance = math.inf if head_surrounding[0] > 1 else \
        min(abs(head_x - bean_x), abs(width - abs(head_x - bean_x))) + \
        min(abs((head_y - 1) % height - bean_y), abs(height - abs((head_y - 1) % height - bean_y)))
    next_distances.append(up_distance)

    # print(head_surrounding[1])
    down_distance = math.inf if head_surrounding[1] > 1 else \
        min(abs(head_x - bean_x), abs(width - abs(head_x - bean_x))) +\
        min(abs((head_y + 1) % height - bean_y), abs(height - abs((head_y + 1) % height - bean_y)))
    next_distances.append(down_distance)

    left_distance = math.inf if head_surrounding[2] > 1 else \
        min(abs((head_x - 1) % width - bean_x), abs(width - abs(abs((head_x - 1) % width - bean_x)))) +\
        min(abs(head_y - bean_y), abs(height - abs(head_y - bean_y)))
    next_distances.append(left_distance)

    right_distance = math.inf if head_surrounding[3] > 1 else \
        min(abs((head_x + 1) % width - bean_x), abs(width - abs(abs((head_x + 1) % width - bean_x)))) + \
        min(abs(head_y - bean_y), abs(height - abs(head_y - bean_y)))
    next_distances.append(right_distance)
    tail=mysnake[-1]
    flag2=[(head_y - 1) % height,head_x]==tail or [(head_y + 1) % height,head_x]==tail or [head_y,(head_x - 1) % width]==tail or [head_y,(head_x + 1) % width]==tail
    if flag or (not flag2):
        regi=[]
        for bean in surrounding:
            count=get_region_size(state,width,height,bean[1],bean[0])
            if count==0:
                regi.append(math.inf)
            elif count<=4.0:
                regi.append(1000/np.sqrt(count))
            else:
                regi.append(0.0)
        next_distances = list(np.array(next_distances)+np.array(regi))
    # print("——————————————————————————————————————")
    # print(state)
    # print(head_y,head_x)
    # print(regi)
    # print(next_distances)
    actions.append(next_distances.index(min(next_distances)))

def manhattan(x,y,bean_x,bean_y,width=8,height=6):
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

def spfa_forall(x, y, bean_position, state, width, height):
    stack=[]
    visited = np.zeros((height,width))
    result = [10000]*len(bean_position)
    flag = [0]*len(bean_position)
    distance = -1
    visited[y][x] = 1
    stack.append([y,x])

    while len(stack)!=0 and sum(flag)!=5:
        distance+=1
        for i in range(len(stack)):
            y_prime,x_prime = stack.pop(0)
            for j in range(len(bean_position)):
                if [y_prime,x_prime]==bean_position[j]:
                    flag[j]=1
                    result[j]=distance
            y = (y_prime - 1) % height
            x = x_prime
            if (state[y][x]<=1) and (visited[y][x]==0):
                stack.append([y,x])
                visited[y][x]=1

            y = (y_prime + 1) % height
            x = x_prime
            if (state[y][x]<=1) and (visited[y][x]==0):
                stack.append([y,x])
                visited[y][x]=1

            y = y_prime
            x = (x_prime - 1) % width
            if (state[y][x]<=1) and (visited[y][x]==0):
                stack.append([y,x])
                visited[y][x]=1

            y = y_prime
            x = (x_prime + 1) % width
            if (state[y][x]<=1) and (visited[y][x]==0):
                stack.append([y,x])
                visited[y][x]=1
    # print(result)
    return result

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
            result.append(math.inf)
        elif count<=4:
            result.append(1000/np.sqrt(count))
        elif count<=10:
            result.append(1/count**0.9)
        else:
            result.append(0)
    # print(result)
    return result

#分析对手最近两个bean
def opponent_bean(beans_position, width, height, mysnake, opsnake, mydis, opdis):
    mylength=len(mysnake)
    oplength=len(opsnake)
    op_head_x=opsnake[0][1]
    op_head_y=opsnake[0][0]
    my_head_x=mysnake[0][1]
    my_head_y=mysnake[0][0]

    mindis=math.inf
    result=[0.0]*len(beans_position)
    index=0
    i=-1
    mindis=min(opdis)
    index=opdis.index(mindis)
    # for bean in beans_position:
    #     i+=1
        # dis=spfa(op_head_x, op_head_y, bean[1], bean[0], state, width, height)
        # dis=manhattan(op_head_x, op_head_y, bean[1], bean[0], width, height)[0]
        # if dis<mindis:
        #     mindis=dis
        #     index=i
    bean=beans_position[index]
    mydis_index=mydis[index]
    if mydis_index>mindis:
        result[index]=6/mindis
    elif mydis_index<mindis:
        result[index]=-1/mydis_index
    else:
        if mylength<=oplength:
            result[index]=-1/mydis_index
        elif mindis==1:
            result[index]=100
        else:
            result[index]=6/mydis_index
    return result,mindis

def get_bean_from_dis(x, y, beans_position, width, height, state, snakes, ctrl_agent_index):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    mysnake=snakes[ctrl_agent_index[0]]
    opsnake=snakes[1-ctrl_agent_index[0]]
    mydis=spfa_forall(mysnake[0][1],mysnake[0][0],beans_position,state,width,height)
    opdis=spfa_forall(opsnake[0][1],opsnake[0][0],beans_position,state,width,height)

    para=[0.1, 0.25, 1]
    aggre=aggregation(beans_position, width, height)
    region=bean_region_dis(state,width,height,beans_position)
    oppo=opponent_bean(beans_position, width, height, mysnake, opsnake, mydis, opdis)[0]

    for i, (bean_y, bean_x) in enumerate(beans_position):
        # distance_manhattan,ind_x,ind_y = manhattan(x,y,bean_x,bean_y,width,height)
        dis = mydis[i] \
              - para[0] * aggre[i] \
              + para[1] * region[i] \
              + para[2] * oppo[i]
        if dis < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = dis
    return min_x, min_y

def greedy_zhy(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []#0:上, 1:下, 2:左, 3:右
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y = get_bean_from_dis(head_x, head_y, beans_position,width, height,
                                                  state_map, snakes, ctrl_agent_index)
        get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state_map,[0])
        # print(actions)
    return actions

#loop_defense核心算法，仅利用
def get_loop(mysnake,width,height):
    length=len(mysnake)
    x=mysnake[0][1]
    y=mysnake[0][0]
    for j in range(length):
        index=length-j-1
        distance,indx,indy = manhattan(x,y,mysnake[index][1],mysnake[index][0],width,height)
        if (length%2==0) and (distance==length-index):
            return indx, indy, mysnake[index][1], mysnake[index][0]
        if(length%2!=0) and (abs(distance-length+index)==1):
            return indx, indy, mysnake[index][1], mysnake[index][0]
def defense_zhy(state_map, beans, snakes, width, height, ctrl_agent_index):
    actions = []#0:上, 1:下, 2:左, 3:右
    for i in ctrl_agent_index:
        mysnake = snakes[i]
        # mylength = len(mysnake)
        head_x = mysnake[0][1]
        head_y = mysnake[0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        #注：t为"广义尾巴"坐标，并非为真实尾巴所处位置
        bean_x, bean_y, t_x, t_y = get_loop(snakes[i],width,height)
        # if (t_x==mysnake[-1][1]) and (t_y==mysnake[-1][0]) and (mylength%2==0):
        #     if head_x==t_x:
        #         actions.append(int((bean_y-head_y)>0))
        #     else:
        #         actions.append(int((bean_x-head_x)>0)+2)
        # else:
        get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state_map,snakes[ctrl_agent_index[0]],0)

    return actions

def compre_greedy_defense_for(state_map, beans, snakes, width, height, ctrl_agent_index,info):
    #去掉蛇尾
    # print(ctrl_agent_index)
    myindex=ctrl_agent_index[0]
    mysnake=snakes[myindex]
    myhead=mysnake[0]
    mytail=mysnake[-1]
    opindex=1-myindex
    opsnake=snakes[opindex]
    opdis=get_surrounding(state_map,width,height,snakes[opindex][0][1],snakes[opindex][0][0])
    if not (1 in opdis):
        state_map[snakes[opindex][-1][0]][snakes[opindex][-1][1]]=0.
    state_map[snakes[myindex][-1][0]][snakes[myindex][-1][1]]=0.

    my_index=ctrl_agent_index[0]
    limitation=16
    # and manhattan(myhead[1],myhead[0],mytail[1],mytail[0])[0]<4
    if len(snakes[my_index])>limitation:
        actions=defense_zhy(state_map, beans, snakes, width, height, ctrl_agent_index)
    elif len(snakes[my_index])<6:
        actions=greedy_zhy(state_map, beans, snakes, width, height, ctrl_agent_index)
    else:
        obs=get_observations(state_map, info, [0, 1], 136, height, width)
        actions=[agent.choose_action(obs[ctrl_agent_index[0]])]
    return actions




def to_joint_action(actions, num_agent):
    joint_action = []
    for i in range(num_agent):
        action = actions[i]
        one_hot_action = [0] * 4
        one_hot_action[action] = 1
        one_hot_action = [one_hot_action]
        joint_action.append(one_hot_action)
    return joint_action
