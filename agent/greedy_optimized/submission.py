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
    snakes = []
    snakes.append(observation_list[0][2])
    snakes.append(observation_list[0][3])
    for i in beans:
        state[i[0], i[1]] = 1
    for i in snakes[0]:
        state[i[0], i[1]] = 2
    for i in snakes[1]:
        state[i[0], i[1]] = 3
    actions = greedy_snake_optimized(state, beans, snakes, width, height, [mysnake])
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action


def manhattan(x,y,bean_x,bean_y,width,height):
    ind_x,ind_y=0.0,0.0
    d_x,d_y=0.0,0.0
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

def dis(i,beans_position,width,height):
    result=0.0
    x=beans_position[i][1]
    y=beans_position[i][0]
    for j in range (len(beans_position)):
        if i==j:
            continue
        bean_x=beans_position[j][1]
        bean_y=beans_position[j][0]
        result+=1/(manhattan(x,y,bean_x,bean_y,width,height)[0])
    return result

def get_min_bean(x, y, beans_position, width, height):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    para=[0.1]
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance_manhattan,ind_x,ind_y = manhattan(x,y,bean_x,bean_y,width,height)
        distance=distance_manhattan-para[0]*dis(i,beans_position,width,height)
        # print(dis(i,beans_position,width,height),distance)
        if distance < min_distance:
            min_x = ind_x
            min_y = ind_y
            min_distance = distance
            index = i
    return min_x, min_y, index

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding

def greedy_snake_optimized(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []#0:上, 1:下, 2:左, 3:右
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position, width, height)
        beans_position.pop(index)

        next_distances = []
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

        actions.append(next_distances.index(min(next_distances)))
        # print(actions)
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
