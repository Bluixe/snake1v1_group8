
def my_controller(observation_list, action_space_list, is_act_continuous=False):
    joint_action = []
    for i in range(len(action_space_list)):
        player = sample(action_space_list[i], is_act_continuous)
        joint_action.append(player)
    return joint_action



def sample(action_space_list_each, is_act_continuous):
    player = []
    if is_act_continuous:
        for j in range(len(action_space_list_each)):
            each = action_space_list_each[j].sample()
            player.append(each)
    else:
        player = []
        for j in range(len(action_space_list_each)):
            # each = [0] * action_space_list_each[j]
            # idx = np.random.randint(action_space_list_each[j])
            if action_space_list_each[j].__class__.__name__ == "Discrete":
                each = [0] * action_space_list_each[j].n
                idx = action_space_list_each[j].sample()
                each[idx] = 1
                player.append(each)
            elif action_space_list_each[j].__class__.__name__ == "MultiDiscreteParticle":
                each = []
                nvec = action_space_list_each[j].high

                sample_indexes = action_space_list_each[j].sample()

                for i in range(len(nvec)):
                    dim = nvec[i] + 1
                    new_action = [0] * dim
                    index = sample_indexes[i]
                    new_action[index] = 1
                    each.extend(new_action)
                player.append(each)
    return player
