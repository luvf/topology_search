import itertools
from multiprocessing import Pool

import numpy as np
from pypownet.agent import Agent
from pypownet.environment import RunEnv


class TopologyAgent(Agent):
    def __init__(self, environment, topology):
        super(TopologyAgent, self).__init__(environment)
        self.topology = topology
        self.t = True

    def act(self, observation):
        if self.t:
            self.t = False
            return self.topology
        else:
            return self.environment.action_space.get_do_nothing_action()


class Random_Agent(Agent):
    def __init__(self, environment):
        super(Random_Agent, self).__init__(environment)

    def act(self, observation):
        return np.random.randint(0,2,76)



def eval_topology(topology, ep_len=30, directory="parameters", lvl="std"):
    environment = RunEnv(parameters_folder=directory, game_level=lvl,
                         chronic_looping_mode="natural", start_id=0,
                         game_over_mode="easy",
                         without_overflow_cutoff=True)
    
    agent = TopologyAgent(environment, topology)
    observation = environment.get_observation()  # _get_obs
    reward = list()
    for i in range(ep_len):
        action = agent.act(observation)
        observation, reward_aslist, done, info = environment.step(action, do_sum=False)
        reward.append(reward_aslist[-1])

    # print(sum(reward))
    return [float(r) for r in reward]


def eval_topologies(topologies):
    pass


def get_env():
    return RunEnv(parameters_folder="parameters", game_level="chronics_contestants",
                  chronic_looping_mode="natural", start_id=0,
                  game_over_mode="soft",
                  without_overflow_cutoff=True)


# Generators


def generator_cutoff(generator, n):
    for i, el in enumerate(generator):
        if i >= n:
            break
        yield el


def get_random_dist_action(action_space, d):
    """

    :param action_space: pypownet.environment.ActionSpace
    :param d: int
    :return:
    """
    while True:
        def substation_nb_actions(x):
            return 2 ** (x - 1) - x

        nb_lines = action_space.lines_status_subaction_length

        probas = np.array([1. for _ in range(nb_lines)] +
                          [substation_nb_actions(action_space.get_number_elements_of_substation(sub_id))
                           for sub_id in action_space.substations_ids])
        el_actions = np.random.choice(len(probas), d, replace=False, p=probas / sum(probas))
        action = action_space.get_do_nothing_action(as_class_Action=True)
        for el in el_actions:
            if el < nb_lines:
                action_space.set_lines_status_switch_from_id(action=action, line_id=el, new_switch_value=1)
            else:
                sub_id = action_space.substations_ids[el - nb_lines]
                nb_sub_els = action_space.get_number_elements_of_substation(sub_id)
                new_configuration = np.zeros(nb_sub_els)
                s = 1
                while s == 1 or (s == nb_sub_els - 1):
                    new_configuration = [0] + list(np.random.randint(0, 2, nb_sub_els - 1))
                    s = sum(new_configuration)
                action_space.set_substation_switches_in_action(action=action, substation_id=sub_id,
                                                               new_values=new_configuration)
        yield action.as_array()


def actioned_component_mask(ap):
    actions = list()

    for i in range((ap.lines_status_subaction_length)):
        action = ap.get_do_nothing_action(as_class_Action=True)
        ap.set_lines_status_switch_from_id(action=action, line_id=i, new_switch_value=1)
        actions.append(action.as_array())
    for i in ap.substations_ids:
        config = [1] * ap.get_number_elements_of_substation(i)
        action = ap.get_do_nothing_action(as_class_Action=True)
        ap.set_substation_switches_in_action(action=action, substation_id=i,
                                             new_values=config)
        actions.append(action.as_array())
    return actions


def representant(ap, topology):
    """
        Give an equivalent topology
    :param ap:
    :param topology:
    :return:
    """
    topology = ap.array_to_action(topology)
    for i in ap.substations_ids:
        config = ap.get_substation_switches_in_action(topology, i, concatenated_output=False)
        if config[0][0] == 1:
            config = np.logical_not(config)
            ap.set_substation_switches_in_action(action=topology, substation_id=i, new_values=config)
    return topology.as_array()


def get_all_dist_action(action_space, d, with_sym=False):
    # """
    nb_lines = action_space.lines_status_subaction_length
    nb_actions = nb_lines + len(action_space.substations_ids)
    yield action_space.get_do_nothing_action()
    for els in itertools.combinations(range(nb_actions), d):
        actions = list()
        for i, el in enumerate(els):
            if el < nb_lines:
                action = action_space.get_do_nothing_action(as_class_Action=True)
                action_space.set_lines_status_switch_from_id(action=action, line_id=el, new_switch_value=1)
                actions.append([action])
            else:
                sub_id = action_space.substations_ids[el - nb_lines]
                nb_sub_els = action_space.get_number_elements_of_substation(sub_id)
                acts = list()
                if with_sym:
                    iterator = itertools.product([0, 1], repeat=nb_sub_els)
                    initList = list()
                else:
                    iterator = itertools.product([0, 1], repeat=nb_sub_els - 1)
                    initList = [0]

                for configuration in list(iterator):
                    new_configuration = initList + list(configuration)
                    action = action_space.get_do_nothing_action(as_class_Action=True)
                    action_space.set_substation_switches_in_action(action=action, substation_id=sub_id,
                                                                   new_values=new_configuration)
                    acts.append(action)
                actions.append(acts)

        for elements in itertools.product(*actions):
            action = action_space.get_do_nothing_action(as_class_Action=False)
            for el in elements:
                action += el.as_array()
            yield action


def neighbor_function(action_space, d=1):
    return lambda topology: [np.logical_xor(topology, a) for a in get_all_dist_action(action_space, d)]


def neighbor_function_all(action_space, d=1):
    return lambda topology: [np.logical_xor(topology, a) for a in get_all_dist_action(action_space, d, with_sym=True)]


def topology_counter(action_space, d):
    # """
    nb_lines = action_space.lines_status_subaction_length
    nb_actions = nb_lines + len(action_space.substations_ids)
    s = 0
    for els in itertools.combinations(range(nb_actions), d):
        m = 1
        for i, el in enumerate(els):
            if el > nb_lines:
                sub_id = action_space.substations_ids[el - nb_lines]
                nb_sub_els = action_space.get_number_elements_of_substation(sub_id)
                if nb_sub_els > 3:
                    m = m * (2 ** (nb_sub_els - 1) - nb_sub_els)
        s += m
    return s


def all_topology_counter(action_space):
    # """
    nb_lines = action_space.lines_status_subaction_length
    s = 2 ** nb_lines
    for i in action_space.substations_ids:
        print(i, s)
        nb_sub_els = action_space.get_number_elements_of_substation(i)
        print(nb_sub_els)
        s = s * (2 ** (nb_sub_els - 1) - nb_sub_els + 1)
    return s


def count_topology(N):
    pool = Pool(1)

    res = pool.map(test_topologies, range(N))
    print(res, sum(res)/(N*2048))


def test_topologies(n):
    environment = RunEnv(parameters_folder="parameters", game_level="chronics_contestants",
                         chronic_looping_mode="natural", start_id=0,
                         game_over_mode="easy",
                         without_overflow_cutoff=True)

    agent = Random_Agent(environment)
    observation = environment.get_observation()  # _get_obs

    reward = 0
    for i in range(4096):
        action = agent.act(observation)
        _, _, done, _ = environment.step(action, do_sum=False)
        reward += (1-done)
        #print(reward)

    # print(sum(reward))
    return reward



''''''
#count_topology(1)


POOL = Pool(1)
