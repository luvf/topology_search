import json
import os
from functools import partial
from heapq import heappop, heappush

from tqdm import tqdm

import IBEA
from pypownet_helper import *


def fitness(x, ref, scale, indic):
    return -np.sum(np.exp([indic(i, x) / scale for i in ref]))


def get_fitness(evaluated, ref_keys, args):
    tops = evaluated.keys()
    ref_values = np.array([evaluated[tuple(s)] for s in ref_keys])

    mi = np.min(ref_values, axis=0)
    ma = np.max(ref_values, axis=0)
    mami = ma - mi
    F = dict()

    new_val = np.array([v - mi / mami for v in evaluated.values()])
    fit = partial(fitness, ref=ref_values, scale=args["fit_scale"], indic=IBEA.indicators[args["indicator"]])
    R = POOL.map(fit, new_val)

    for k, v in zip(tops, R):
        if k in ref_keys:
            v += 1
        F[k] = v

    return F


def nbest(evaluated, pop, n):
    evp = [evaluated[el] for el in pop]
    for i, el in enumerate(evp):
        evp[i] = sum(sorted(el)[:n])
    return list(zip(*sorted(zip(evp, pop))))[1]


def primary_selection(evaluated, pop):
    ap = get_env().action_space
    mask = actioned_component_mask(ap)

    def nb_actioned_components(action):
        return sum([np.logical_and(action, el).any() != 0 for el in mask])

    connex_topos = connex_component(evaluated)

    max_k = np.array([max(el[k] for el in evaluated.values() if el[k] < 199) for k in range(len(evaluated[pop[0]]))])
    min_k = np.array([min(el[k] for el in evaluated.values() if el[k] < 199) for k in range(len(evaluated[pop[0]]))])
    evaluated2 = {k: (v-min_k)/(max_k-min_k) for k, v in evaluated.items()}
    mins1 = nbest(evaluated2, connex_topos, 1)[:60]
    mins2 = nbest(evaluated2, connex_topos, 2)[:30]
    mins3 = nbest(evaluated2, connex_topos, 3)[:25]
    minsn1 = nbest(evaluated2, connex_topos, 10)[:5]
    minsn2 = nbest(evaluated2, connex_topos, 15)[:5]
    minsn3 = nbest(evaluated2, connex_topos, 20)[:5]
    minsn4 = nbest(evaluated2, connex_topos, 25)[:5]

    print("size of the main component : ", len(connex_topos))

    #nb_actioned = [nb_actioned_components(el) for el in pop]


    return list(mins1) + list(mins2) + list(mins3) + list(minsn1)+ list(minsn2)+ list(minsn3)+ list(minsn4)+[tuple(ap.get_do_nothing_action().tolist())]


def topology_selection(evaluated, pop, args):
    F = get_fitness(evaluated, pop,  args)

    def neighbors(graph, node):
        topology_node = graph.nodes[node]
        return [(n.topology, F[tuple(n.topology)]) for n in topology_node.neighbors]

    topology_graph = TopologyGraph(neighbor_function_all(get_env().action_space))
    topology_graph.init_topologies(evaluated)

    primary_selected = primary_selection(evaluated, pop)


    print("first selection", len(primary_selected))
    # print(len(nb_comp_connexes(topology_graph)))
    print("init")
    selected = set()
    for i in tqdm(range(len(primary_selected))):
        selected = selected.union(
            dijkstra(primary_selected[i], primary_selected[i:], topology_graph, neighbors))
    return np.array(list(selected)).tolist()


def extract_topologies(args, cur_ev):
    ap = get_env().action_space
    with open(os.path.join("topology_search", "evaluated.json"), "r") as f:
        evaluated = json.load(f)
        evaluated = {tuple(representant(ap, k)): v for k, v in evaluated}
        print("nb evaluated topologies : ", len(evaluated))

    pop=list()
    """for lv in args["levels"]:
        with open(os.path.join("topology_search", lv+"pop.json"), "r") as f:
            pp = json.load(f)
            pop += list(map(tuple,pp))
    pop = list(set(pop))"""

    topologies_selected = topology_selection(evaluated, list(cur_ev), args)
    print("nb selected topologies : ", len(topologies_selected))
    with open(os.path.join("topology_search", "selected_topologies.json"), "w") as ff:
        json.dump(list(topologies_selected), ff)


class TopologyNode:
    def __init__(self, topology, value):
        self.neighbors = set()
        self.topology = topology
        self.value = value
        self.visit = 0

    def add_neighbor(self, neighbor):
        self.neighbors.add(neighbor)

    def add_neighbors(self, neighbors):
        self.neighbors.update(neighbors)


class TopologyGraph:
    def __init__(self, neighbor_func):
        self.nodes = dict()
        self.neighborFunction = neighbor_func
        self.unevaluated = list()

    def add_topology(self, topology, value, neighbor_calc=False):
        if not isinstance(topology, tuple):
            tuple_topology = tuple(topology.tolist())
        else:
            tuple_topology = topology
        cur_node = self.nodes.setdefault(tuple_topology, TopologyNode(topology, value))

        if not cur_node.value:
            cur_node.value = value

        neighbors = self.neighborFunction(topology)
        if neighbor_calc:
            tuple_neighbor = {tuple(t.tolist()) for t in neighbors}

            tuple_neighbor = tuple_neighbor & self.nodes.keys()

            node_neighbors = {self.nodes[k] for k in tuple_neighbor}

            cur_node.add_neighbors(node_neighbors)

            for n in node_neighbors:
                n.add_neighbor(cur_node)

    def update_neighbors(self):
        for cur_node in self.nodes.values():
            neighbors = self.neighborFunction(cur_node.topology)
            tuple_neighbor = {tuple(t.tolist()) for t in neighbors}
            node_neighbors = {self.nodes[k] for k in tuple_neighbor if k in self.nodes}
            cur_node.add_neighbors(node_neighbors)

            for n in node_neighbors:
                n.add_neighbor(cur_node)

    def add_topologies(self, topologies):
        for k, v in topologies.items():
            self.add_topology(k, v)

    def init_topologies(self, datas):
        """
        without border
        :param datas:
        :return:
        """
        topologies = set(datas.keys())
        new_nodes = {t: TopologyNode(t, v) for t, v in datas.items()}

        el_neigbor = self.neighborFunction(get_env().action_space.get_do_nothing_action())

        for t in tqdm(new_nodes):
            nb = {tuple(np.logical_xor(new_nodes[t].topology, n).tolist()) for n in el_neigbor}
            nb = topologies.intersection(nb)
            new_nodes[t].add_neighbors([new_nodes[d] for d in nb])

        self.nodes = new_nodes

    def get_unevaluated(self):
        return [self.nodes[t] for t in self.unevaluated]

    def dump_json(self, filename):
        evaluated_nodes = [(k, n.value) for k, n in self.nodes.items() if n.value]
        with open(filename, 'w') as file:
            json.dump(evaluated_nodes, file)


def dijkstra(source, targets, graph, neighbors):
    T = set(targets)
    visited = set()
    d = {source: 0}
    p = {}
    nexts = [(0, source)]  # tas de couples (d[x],x)
    while nexts != [] and len(T) > 0:
        dx, x = heappop(nexts)
        if x in visited:
            continue
        if x in T:
            T.remove(x)
        visited.add(x)
        for y, w in neighbors(graph, x):
            if y in visited:
                continue
            dy = dx + w
            if y not in d or d[y] > dy:
                d[y] = dy
                heappush(nexts, (dy, y))
                p[y] = x
    path = [source]
    for t in targets:
        x = t
        while x != source:
            path.append(x)
            x = p[x]
    return path


def connex_component(evaluated):
    evs = set(evaluated.keys())
    ap = get_env().action_space
    source = tuple(ap.get_do_nothing_action())
    neighbors = neighbor_function_all(ap)
    visited = set()
    nexts = {source} # tas de couples (d[x],x)
    while len(nexts) != 0:
        x = nexts.pop()
        if x in visited: #useless a priori
            continue
        visited.add(x)
        for l in neighbors(x):
            el = tuple(l)
            if el in evs and el not in visited:
                nexts.add(el)
    return list(visited)


def topology_search(args):
    global POOL
    POOL = Pool(args["n_threads"])
    ev1 = IBEA.main(args, level= args["levels"][0])

    ev2 = IBEA.main(args, level= args["levels"][1], evaluated="topology_search/evaluated.json")

    extract_topologies(args,ev1.union(ev2))

    POOL.close()

if __name__ == '__main__':
    # TODO
    pass
