import copy
import math
import numpy as np

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
    # state[snakes[0][-1][0]][snakes[0][-1][1]]=0.
    # state[snakes[1][-1][0]][snakes[1][-1][1]]=0.
    # print(state)
    actions = compre_greedy_defense(state, beans, snakes, width, height, [mysnake])
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

#控制蛇移动的核心函数，考虑穿越边界，支持bean坐标为负数
def get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state,flag=1):
    next_distances = []
    surrounding=[[(head_y - 1) % height,head_x],[(head_y + 1) % height,head_x],
                 [head_y,(head_x - 1) % width],[head_y,(head_x + 1) % width]]
    if flag:
        regi=[]
        for bean in surrounding:
            count=get_region_size(state,width,height,bean[1],bean[0])
            if count==0:
                regi.append(math.inf)
            elif count<=4.0:
                regi.append(1000/np.sqrt(count))
            else:
                regi.append(0.0)
    # print("——————————————————————————————————————")
    # print(state)
    # print(head_y,head_x)
    # print(regi)
    up_distance = math.inf if head_surrounding[0] > 1 else \
        min(abs(head_x - bean_x), abs(width - abs(head_x - bean_x))) + \
        min(abs((head_y - 1) % height - bean_y), abs(height - abs((head_y - 1) % height - bean_y)))
    next_distances.append(up_distance)

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
    if flag:
        next_distances = list(np.array(next_distances)+np.array(regi))
    # print(next_distances)
    actions.append(next_distances.index(min(next_distances)))

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
def opponent_bean(beans_position, width, height, mysnake, opsnake):
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

def get_bean_from_dis(x, y, beans_position, width, height, state, snakes, ctrl_agent_index):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    mysnake=snakes[ctrl_agent_index[0]]
    opsnake=snakes[1-ctrl_agent_index[0]]

    para=[0.1, 0.25, 1]
    aggre=aggregation(beans_position, width, height)
    region=bean_region_dis(state,width,height,beans_position)
    oppo=opponent_bean(beans_position, width, height, mysnake, opsnake)[0]
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance_manhattan,ind_x,ind_y = manhattan(x,y,bean_x,bean_y,width,height)
        dis = distance_manhattan \
              - para[0] * aggre[i] \
              + para[1] * region[i] \
              + para[2] * oppo[i]
        if dis < min_distance:
            min_x = ind_x
            min_y = ind_y
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
        get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state_map)
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
        get_actions(head_x,head_y,bean_x,bean_y,head_surrounding,width,height,actions,state_map,0)

    return actions

def compre_greedy_defense(state_map, beans, snakes, width, height, ctrl_agent_index):
    #去掉蛇尾
    myindex=ctrl_agent_index[0]
    opindex=1-myindex
    opdis=opponent_bean(beans,width,height,snakes[myindex],snakes[opindex])[1]
    if opdis!=1:
        state_map[snakes[opindex][-1][0]][snakes[opindex][-1][1]]=0.
    state_map[snakes[myindex][-1][0]][snakes[myindex][-1][1]]=0.

    my_index=ctrl_agent_index[0]
    limitation=16
    if len(snakes[my_index])>limitation:
        actions=defense_zhy(state_map, beans, snakes, width, height, ctrl_agent_index)
    else:
        actions=greedy_zhy(state_map, beans, snakes, width, height, ctrl_agent_index)
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
