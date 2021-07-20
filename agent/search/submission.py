import numpy as np
# from snakes import SnakeEatBeans


def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    width = observation_list[0]['board_width']
    height = observation_list[0]['board_height']
    c = observation_list[0]['controlled_snake_index']

    state = np.zeros((height, width))  # state是二维矩阵

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
    info = {'snakes_position': [observation_list[0][2], observation_list[0][3]], 'beans_position': beans,
            'directions': observation_list[0]['last_direction']}

    actions = []
    actions[:] = search_deep(info, [c], 2)
    player = []
    each = [0] * 4
    each[actions[0]] = 1
    player.append(each)
    joint_action.append(player)
    return joint_action


def search(info, ctrl_agent, step, length):
    # print(info)
    dire_dict = {'up': 0,
                 'down': 1,
                 'left': 2,
                 'right': 3}
    reward = np.zeros((4,4))  # axis 0: my, 1:op
    min_r = np.zeros(4)
    max_r = -1000

    # 4*4 -> 3*3 剪枝
    for i in range(4):
        if i == dire_dict[info['directions'][ctrl_agent[0]]]:
            reward[i, :] = np.nan
            continue
        for j in range(4):
            if j == dire_dict[info['directions'][1 - ctrl_agent[0]]]:
                reward[i, j] = np.nan
                continue
            joint_action = [[[0] * 4]] * 2
            joint_action[0][0][i] = 1
            joint_action[1][0][j] = 1
            n_info, death = get_next_state(info, joint_action)
            if death[ctrl_agent[0]] == 0 and death[1 - ctrl_agent[0]] == 0:
                if step == 0:
                    n_length = [len(n_info['snakes_position'][0]), len(n_info['snakes_position'][1])]
                    reward[i, j] = (n_length[ctrl_agent[0]] - length[ctrl_agent[0]])*5-(n_length[1-ctrl_agent[0]]-length[1-ctrl_agent[0]])*4
                else:
                    reward[i, j] = search(n_info, ctrl_agent, step - 1, length)
            elif death[ctrl_agent[0]] == 1 and death[1 - ctrl_agent[0]] == 0:  # 自己死了对手没死
                reward[i, j] = -10
            elif death[ctrl_agent[0]] == 0 and death[1 - ctrl_agent[0]] == 1:  # 自己没死对手死了
                reward[i, j] = 2
            else:  # 都死了
                reward[i, j] = -5
            min_r[i] = min(min_r[i], reward[i, j])
            # alpha-beta 剪枝
            if i != 0 and min_r[i] < max_r:
                break
        max_r = max(max_r, min_r[i])
    return max_r




def search_deep(info, ctrl_agent, step):
    dire_dict = {'up': 0,
                 'down': 1,
                 'left': 2,
                 'right': 3}
    length = [len(info['snakes_position'][0]), len(info['snakes_position'][1])]
    joint_action = [[[0] * 4]] * 2
    reward = np.zeros((4, 4))  # axis 0: my, 1:op
    min_r = np.zeros(4)
    max_r = -1000
    for i in range(4):
        for j in range(4):
            joint_action[0][0][i] = 1
            joint_action[1][0][j] = 1
            n_info, death = get_next_state(info, joint_action)
            if death[ctrl_agent[0]] == 0 and death[1 - ctrl_agent[0]] == 0:
                if step == 0:
                    n_length = [len(n_info['snakes_position'][0]), len(n_info['snakes_position'][1])]
                    reward[i, j] = n_length[ctrl_agent[0]] - length[ctrl_agent[0]]
                else:
                    reward[i, j] = search(n_info, ctrl_agent, step - 1, length)
            elif death[ctrl_agent[0]] == 1 and death[1-ctrl_agent[0]] == 0:  # 自己死了对手没死
                reward[i, j] = -10
            elif death[ctrl_agent[0]] == 0 and death[1-ctrl_agent[0]] == 1:  # 自己没死对手死了
                reward[i, j] = 2
            else:  # 都死了
                reward[i, j] = -5
            min_r[i] = min(min_r[i], reward[i, j])
            if i != 0 and min_r[i] < max_r:
                break
        max_r = max(max_r, min_r[i])
    act = [np.array(np.where(max_r == min_r))[0][0]]
    # print(act)
    return act

def get_next_pos(cur_pos, act, height, width):
    x = cur_pos[1]
    y = cur_pos[0]
    pos = []
    if act == 0:
        pos = [(y - 1) % height, x]
    elif act == 1:
        pos = [(y + 1) % height, x]
    elif act == 2:
        pos = [y, (x - 1) % width]
    elif act == 3:
        pos = [y, (x + 1) % width]
    return pos

def get_next_state(info, joint_action):
    dire_dict = {0: 'up',
                 1: 'down',
                 2: 'left',
                 3: 'right'}
    dire = []
    death = [0, 0]
    next_snake = []
    beans = info['beans_position'].copy()
    for i in range(2):
        act = joint_action[i][0].index(1)
        head_pos = info['snakes_position'][i][0]
        p = get_next_pos(head_pos, act, 6, 8)
        while p[0] == info['snakes_position'][i][1][0] and p[1] == info['snakes_position'][i][1][1]:
            act = np.random.randint(4)
            p = get_next_pos(head_pos, act, 6, 8)
        dire.append(dire_dict[act])
        snake = info['snakes_position'][i]
        flg = 0
        n_snake = [p]
        for a, b in beans:
            if a == p[0] and b == p[1]:
                flg = 1
                break
        n_snake = np.append(n_snake, snake, axis=0) if flg else np.append(n_snake, snake[:-1], axis=0)
        next_snake.append(n_snake)


    for i, j in next_snake[1]:
        if next_snake[0][0][0] == i and next_snake[0][0][1] == j:
            death[0] = 1
            break
    for i, j in next_snake[0]:
        if next_snake[1][0][0] == i and next_snake[1][0][1] == j:
            death[1] = 1
            break

    n_info = {'snakes_position': next_snake, 'beans_position': beans,
              'directions': dire}

    return n_info, death