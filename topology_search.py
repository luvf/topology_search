from pypownet.environment import RunEnv
from pypownet.agent import Agent

import numpy as np

import itertools
from multiprocessing import Pool
import json
import argparse


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


class TopologySearch(object):
    def __init__(self):
        self.environment = RunEnv(parameters_folder=".", game_level="input",
                                  chronic_looping_mode="natural", start_id=0,
                                  game_over_mode="soft",
                                  without_overflow_cutoff=True)
        self.explored_topologies = dict()

    def add_evaluated_topologies(self, topologies):
        self.explored_topologies.update(topologies)

    @staticmethod
    def eval_topology(topology):
        environment = RunEnv(parameters_folder=".", game_level="input",
                             chronic_looping_mode="natural", start_id=0,
                             game_over_mode="soft",
                             without_overflow_cutoff=True)

        agent = TopologyAgent(environment, topology)

        observation = environment.get_observation()
        action = agent.act(observation)
        observation, reward_aslist, done, info = environment.step(action, do_sum=False)
        reward = [reward_aslist[-1]]
        if reward == 0.:
            return 0
        for i in range(95):
            action = agent.act(observation)
            observation, reward_aslist, done, info = environment.step(action, do_sum=False)
            reward.append(reward_aslist[-1])

        print(sum(reward))
        return tuple(topology.tolist()), [float(r) for r in reward]

    def eval_topologies(self, topologies):
        evaluated = POOL.map(self.eval_topology, topologies)
        self.add_evaluated_topologies(evaluated)
        return evaluated

    def search(self, budget):
        pass

    def load(self, filename):
        with open(filename, "r") as file:
            tops = json.load(file)
            self.add_evaluated_topologies(tops)

    def save(self, filename):
        with open(filename, "w") as file:
            json.dump(list(self.explored_topologies.items()), file)


class GeneratedTopologySearch(TopologySearch):
    def __init__(self, generator):
        super(GeneratedTopologySearch, self).__init__()
        self.generator = generator

    def search(self, budget=2 ** 100):
        evaluated = self.eval_topologies(generator_cutoff(self.generator, budget))
        return evaluated


class MCTSSearch(TopologySearch):
    def __init__(self):
        super(MCTSSearch, self).__init__()

        self.topology_graph = TopologyGraph(neighbor_function(self.environment.action_space))

        do_nothing = self.environment.action_space.get_do_nothing_action(as_class_Action=False)
        value0 = self.eval_topology(do_nothing)

        self.add_evaluated_topologies([value0])
        #self.topology_graph.add_topology(do_nothing, value0)

    def add_evaluated_topologies(self, topologies):
        self.topology_graph.add_topologies(dict(topologies))
        super(MCTSSearch, self).add_evaluated_topologies(topologies)

    def search(self, budget, batch_size):
        evaluated = list()

        values = [n.value for n in self.topology_graph.nodes.values() if n.value is not None]
        means = np.divide(values, len(values))
        for i in range( //batch_size):
            to_evaluate = self.topology_graph.get_unevaluated()
            utility = np.array([MCTSSearch.utility_function(t, means) for t in to_evaluate])
            tops = [t.topology for t in to_evaluate]

            selected = np.random.choice(len(tops), batch_size, replace=False, p= utility / sum(utility))
            topos = [tops[s] for s in selected]

            evaluated += self.eval_topologies(topos)
            self.save("tmp/topologies_save.json")

            # self.topology_graph.add_topology(topology, score, score != 0)
            # scores.append(score)
        return evaluated

    @staticmethod
    def utility_function(topology, means):
        m = np.maximum(0.1, means)  # avoyds divide 0
        ret = [np.divide(n.value, m).sum() for n in topology.neighbors if n.value is not None]
        return sum(np.square(ret)) / len(ret) if len(ret) else 0



def increasing_mean(n_els, means, new_el):
    means = (means * n_els + new_el) / (n_els + 1)
    return n_els, means


class TopologyNode:
    def __init__(self, topology, value):
        self.neighbors = set()
        self.topology = topology
        self.value = value

    def add_neighbor(self, neighbor):
        self.neighbors.add(neighbor)

    def add_neighbors(self, neighbors):
        self.neighbors.update(neighbors)


class TopologyGraph:
    def __init__(self, neighbor_func):
        self.nodes = dict()
        self.unevaluated = set()
        self.neighborFunction = neighbor_func

    def add_topology(self, topology, value, explore=True):
        tuple_topology = tuple(topology)
        cur_node = self.nodes.setdefault(tuple_topology, TopologyNode(topology, value))

        if not cur_node.value:
            cur_node.value = value

        self.unevaluated.discard(tuple_topology)

        neighbors = self.neighborFunction(topology)

        tuple_neighbor = {tuple(t) for t in neighbors}
        if explore:
            frontier = tuple_neighbor - self.nodes.keys()
            new_unevaluated = {f: TopologyNode(np.array(f), None) for f in frontier}

            self.nodes.update(new_unevaluated)
            self.unevaluated.update(frontier)
        else:
            tuple_neighbor = tuple_neighbor & self.nodes.keys()

        node_neighbors = {self.nodes[k] for k in tuple_neighbor}

        cur_node.add_neighbors(node_neighbors)

        for n in node_neighbors:
            n.add_neighbor(cur_node)

    def add_topologies(self, topologies):
        for k, v in topologies.items():
            self.add_topology(k, v)

    def get_unevaluated(self):
        return [self.nodes[t] for t in self.unevaluated]

    def dump_json(self, filename):
        evaluated_nodes = [(k, n.value) for k, n in self.nodes.items() if n.value ]
        with open(filename, 'w') as file:
            json.dump(evaluated_nodes, file)



env = RunEnv(parameters_folder=".", game_level="input",
             chronic_looping_mode="natural", start_id=0,
             game_over_mode="soft",
             without_overflow_cutoff=True)


def main(args):
    # ts =  RandomTopologySearch(dist_generator(3)
    print("searching trivial topologies")
    gen = get_all_dist_action(env.action_space, 1)
    t1 = GeneratedTopologySearch(gen)
    t1.search(1000)
    t1.save("tmp/topologies_save_dist1.json")
    print("MCTS Search")
    ts = MCTSSearch()
    ts.add_evaluated_topologies(t1.explored_topologies)

    scores = ts.search(10000, 64) #args.threads)
    ts.save("topologies_save_final.json")


def count(args):
    ts = MCTSSearch()
    ts.load("tmp/topologies_save_dist.json")

    uneval = ts.topology_graph.get_unevaluated()
    print(len(uneval))
    print(len(ts.topology_graph.nodes))





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


def get_all_dist_action(action_space, d):
    # """
    nb_lines = action_space.lines_status_subaction_length
    nb_actions = nb_lines + len(action_space.substations_ids)

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
                for configuration in list(itertools.product([0, 1], repeat=nb_sub_els - 1)):
                    new_configuration = [0] + list(configuration)
                    if sum(new_configuration) != 1 and sum(new_configuration) != nb_sub_els - 1:
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
    return lambda topology: [topology + a for a in get_all_dist_action(action_space, d)]


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



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--threads', dest='threads', type=int, default=1)
args = parser.parse_args()

POOL = Pool(args.threads)


#main(args)
uneval get_unevaluated_size():



# TODO
# tester la sauvegarde des topologies
# ameliorer le support du multprocess
# netoyer le code
