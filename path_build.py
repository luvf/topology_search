from pypownet_helper import *
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, Value
import collections
from itertools import product

import json
import os

n_threads = 1


def run_scenarios(args):
    global n_threads
    n_threads = args["n_threads"]
    # POOL = Pool(args["n_threads"])

    with open(os.path.join("topology_search", "selected_topologies.json"), "r") as ff:
        topologies = list(tuple(el) for el in json.load(ff))
    if isinstance(args["n_scenarios"], list):
        iteraror = args["n_scenarios"]
    else:
        iteraror = range(int(args["n_scenarios"]))

    for s in iteraror:
        evaluated_topologies = eval_tops(topologies, s, args)
        with open(os.path.join("saved_scenario", args["game_level"], "evaluated_scenario" + str(s) + ".json"),
                  "w") as ff:
            json.dump(list(evaluated_topologies.items()), ff)


def eval_tops(topologies, scenario, args):
    if isinstance(args["n_steps"], list):
        nsteps = args["n_steps"][scenario]
    else:
        nsteps = int(args["n_steps"])

    with Pool(n_threads) as pool:
        evaluation = partial(eval_top, scenario=scenario, nstep=nsteps + 1, game_level=args["game_level"])
        scores = pool.map(evaluation, topologies)
    return dict(zip(topologies, scores))


def eval_top(topology, scenario, nstep, game_level):
    environment = RunEnv(parameters_folder="parameters", game_level=game_level,
                         chronic_looping_mode="natural", start_id=scenario,
                         game_over_mode="easy",
                         without_overflow_cutoff=True)
    agent = TopologyAgent(environment, topology)
    observation = environment.get_observation()  # _get_obs

    reward = list()
    for i in range(nstep):
        action = agent.act(observation)
        observation, reward_aslist, done, info = environment.step(action, do_sum=False)
        if reward_aslist[-2] != 0:
            reward.append(200)
        else:
            reward.append(reward_aslist[-1])
    return [float(r) for r in reward]


def game_graph(scenario):
    env = get_env()
    action_space = env.action_space
    nb = partial(neighbor_function_all, action_space, d=1)
    mask = actioned_component_mask(env.action_space)

    with open(scenario, "r") as ff:
        evaluated_topologies = json.load(ff)
        evaluated_topologies = {tuple(k): v for k, v in evaluated_topologies}

        doNothing = tuple(env.action_space.get_do_nothing_action().astype(int).tolist())
        topologies = list(evaluated_topologies.keys())
        neighbors = [tuple(el.astype(int).tolist()) for el in nb()(doNothing)]
        top_dict = {t: i for i, t in enumerate(topologies)}

        for el in neighbors:
            top_dict.setdefault(el, len(top_dict))
        new_evaluated = {top_dict[k]: v for k, v in evaluated_topologies.items()}
        #print(list(top_dict.items())
        # print(neighbors_topologies)
        # nbs = [tuple(el.tolist()) for el in neighbor_function()(t)))]
        nbs = dict()
        for t in topologies:
            nn = list(filter(lambda x: x in topologies, [tuple(el.astype(int).tolist()) for el in nb()(t)]))
            nbs[top_dict[t]] = [top_dict[a] for a in nn]

        actions = {(top_dict[a], top_dict[b]):
                       top_dict.get(tuple(np.logical_xor(a, b).astype(int).tolist()), top_dict[doNothing]) for a, b
                   in product(topologies, topologies)}

        activated_comp = {
            (top_dict[t], top_dict[n]): np.array([np.logical_and(el, np.logical_xor(t, n)).any() for el in mask])
            for t, n in product(topologies, topologies)}

    return top_dict, new_evaluated, nbs, actions, activated_comp


def find_path(args, name=""):
    global n_threads
    n_threads = args["n_threads"]

    game_level = args["game_level"]

    if isinstance(args["n_scenarios"], list):
        iteraror = list(args["n_scenarios"])
    else:
        iteraror = list(range(int(args["n_scenarios"])))

    top_dict, new_evaluated, nbs, actions, activated_comp = game_graph(
        os.path.join("saved_scenario", game_level, "evaluated_scenario" + str(iteraror[0]) + ".json"))
    with open(os.path.join("paths", game_level, "graph.json"), "w") as ff:
        json.dump(list(top_dict.items()), ff)
    path = list()
    for s in iteraror:
        with open(os.path.join("saved_scenario", game_level, "evaluated_scenario" + str(s) + ".json"), "r") as ff:
            evaluated_topologies = json.load(ff)
            evaluated_topologies = {tuple(k): v for k, v in evaluated_topologies}

        # snf = partial(neighbor_function, nb=nb, epsilon=0, component_mask=comp_mask)
        if isinstance(args["n_steps"], list):
            nsteps = args["real_n_steps"][s]
        else:
            nsteps = int(args["n_steps"])
        path.append(topology_path(evaluated_topologies, nsteps - 1, s, game_level, top_dict, nbs, actions, activated_comp))
    with open("output.json", "r") as f:
        steps=json.load(f)
    with open("output.json", "w") as f:
        steps[name] = path
        json.dump(steps, f)

def topology_path(evaluated_topologies, nsteps, s, game_level, top_dict, nbs, actions, activated_comp):
    new_evaluated = {top_dict[k]: v for k, v in evaluated_topologies.items()}
    #topologies = list(evaluated_topologies.keys())

    reversed_episode = {k: list(reversed(v)) for k, v in new_evaluated.items()}
    topologies = [top_dict[el] for el in evaluated_topologies.keys()]

    h_len=len(list(activated_comp.values())[0])

    nf1 = partial(history_neighbor_function, nb=nbs, activated_components=activated_comp, actions=actions, epsilon=5, h_len=h_len, hist_len=0)
    nf2 = partial(history_neighbor_function, nb=nbs, activated_components=activated_comp, actions=actions, epsilon=5, h_len=h_len)


    path2 = custom_optimal_path(new_evaluated, nsteps, nf1, topologies)
    with open(os.path.join("paths", game_level, "optimal_" + str(s) + ".json"), "w") as ff:
        json.dump(list(path2), ff)
    print("path 1 : evaluated")


    """path2 = custom_optimal_path(reversed_episode, nsteps, nf1, topologies)
    with open(os.path.join("paths", game_level, "reversed_" + str(s) + ".json"), "w") as ff:
        json.dump(path2, ff)
    print("path 1 : evaluated")
    """
    path3 = greedy_policy(new_evaluated, nsteps+1, nbs, actions)
    with open(os.path.join("paths", game_level, "greedy_" + str(s) + ".json"), "w") as ff:
        json.dump(path3, ff)
    print("path 3 : evaluated")

    print(get_optimal_path_score(path2, nsteps))
    print(get_greedy_path_score(new_evaluated, path3, top_dict, nsteps))
    print(get_topology_score(new_evaluated, nsteps))
    print(get_optimal_topology_score(new_evaluated, nsteps))

    return {"optimal" : get_optimal_path_score(path2, nsteps),
            "greedy": get_greedy_path_score(new_evaluated, path3, top_dict, nsteps),
            "one_topology" :get_topology_score(new_evaluated, nsteps),
            "global_opptimum" :get_optimal_topology_score(new_evaluated, nsteps)
            # "reversed1": (list(path4.items()), list(values[4].items()))
            #"greedy"  : list(path3)
            }


# _______ optimal path


def history_neighbor_function(node, node_grid, history, nb, activated_components, actions, epsilon,h_len, hist_len=0):
    n, it = node
    cur_node = n
    hist = np.zeros(h_len)  # stores the number of time each component is activated in hist_len
    for i in range(hist_len):
        if it - i < 0:
            break
        next_node, action = history[it - i][cur_node][1:3]
        # history.get((cur_node, it - i), (0, do_nothing, do_nothing))[1:]
        hist += activated_components[cur_node, next_node]
        cur_node = next_node

    topos = nb[n]
    #activated_comp = {t: 0 for t in topos}
    topo_coef = [(np.array(hist) * (activated_components[n, t])).sum() for t in topos]

    ret = [None] * len(topos)  #  fast
    for i in range(len(topos)):#top, coef in zip(topos, topo_coef):
        #new_node = (top, it + 1)
        top=topos[i]
        value = node_grid[top][it + 1] * (1 + epsilon * topo_coef[i])
        action = actions[top, n]
        ret[i] =(top, value, action)
    return ret


import cProfile
import pstats, io

def custom_optimal_path(node_grid, len_episode, neighbor_function, topologies):
    # d = dict() # {s: 0}
    #cp = cProfile.Profile()
    #cp.enable()


    d = list()

    d.append([(0, node, node, node) for node in topologies])
    for i in tqdm(range(len_episode)):
        map_f = partial(map_fn, i=i, history=d, node_grid=node_grid, neighbor_function=neighbor_function)
        map_reduce_results = map_reduce(map_f, reduce_fn, topologies, i)
        # d.update(map_reduce_results)
        d.append(map_reduce_results)
    #cp.disable()
    #sortby = 'cumulative'
    #ps = pstats.Stats(cp).sort_stats(sortby)
    #ps.print_stats()

    #cp = pstats.Stats(pr).sort_stats(sortby)
    #cp.print_stats()
    return d

def get_optimal_path_score(path, nsteps):
    return min(path[nsteps], key= lambda x:x[0])[0]

def get_greedy_path_score(evaluated_topologies, path, top_dict, nsteps):
    v=0
    t = top_dict[tuple([0]*76)]
    for i in range(nsteps):
        v += evaluated_topologies[t][i]
        t = path[i][t][0]
    return v

def get_topology_score(evaluated_topologies, nsteps):
    return min([sum(ev[:nsteps]) for ev in evaluated_topologies.values()])

def get_optimal_topology_score(evaluated_topologies,nsteps):
    return sum([min([ev[i] for ev in evaluated_topologies.values()]) for i in range(nsteps)])

# ______ Greedy policy


def greedy_choice(node, i, neighbors, node_grid, actions):
    existing_nexts = [(node_grid[el][i+1], el, actions[node, el]) for el in neighbors[node]]
    v , new_node, action = min(existing_nexts, key=lambda x: x[0])
    return (new_node, action, v)


def greedy_policy(node_grid, len_episode, neighbors, actions):
    p = list()
    nodes = list(node_grid.keys())
    greedy_ch = partial(greedy_choice, neighbors=neighbors, node_grid=node_grid, actions=actions)
    #with Pool(n_threads) as pool:
    for i in tqdm(range(len_episode)):
        map_fun = partial(greedy_ch, i=i)
        ps = list(map(map_fun, nodes))
        p.append(ps)
    return p


# ______ Map Reduce


def map_fn(node, i, history, node_grid, neighbor_function):
    ret = list()
    #top_dict = {t: i for i, t in enumerate(node_grid.keys())}

    # dx = history.get((node, i), [10e100])[0]
    dx = history[i][node][0]
    for y, w, a in neighbor_function((node, i), node_grid, history):
        dy = dx + w
        ret.append((y, (dy, node, a, y)))
    return ret  # DY, P, D


def reduce_fn(item):
    k, v = item
    ret = min(v, key=lambda x: x[0])
    return k, ret


def partition(mapped_values):
    partitioned_data = collections.defaultdict(list)
    for key, value in mapped_values:
        partitioned_data[key].append(value)
    return partitioned_data.items()


def map_reduce(map_func, reduce_func, inputs, it):
    # with Pool(n_threads) as pool:
    map_responses = map(map_func, inputs)
    partitioned_data = partition(itertools.chain(*map_responses))
    reduced_values = dict(map(reduce_func, partitioned_data))
    return [reduced_values.get(t, (200, 0, 0, 0)) for t in inputs]


#  auxiliaries (to be removed

def covariate_matrix():
    with open('tmp/topologies_save_final.json', 'r') as ff:
        data = json.load(ff)
    ddata = np.array(np.transpose([d[1] for d in data]))
    cov = np.cov(ddata)
    return cov




def nb_top_connexes():
    arretes = [(1,2), (1,5),
              (2,3),(2,4),(2,5),
              (3,4),
              (4,5), (4,7), (4,9),
              (5,6),
              (6,11), (6,12), (6,13),
              (7,8), (7,9),
              (9,10), (9,14),
              (10,11),
              (12,13), (13,14)
              ]
    nb = 0
    print(nb_top_connexes_rec([], arretes))


def nb_top_connexes_rec(keeped, arretes):
    graph = collections.defaultdict(list)
    for x, y in keeped + arretes:
        graph[x].append(y)
        graph[y].append(x)
    out = union_find2(graph)
    if len({find(el)[0] for el in out.values()}) != 1:
        return 0
    elif len(arretes) != 0:
        nb = 0
        for i in range(len(arretes)):
            nb += nb_top_connexes_rec(keeped + arretes[:i], arretes[i+1:])
        return nb+1
    else :
        return 0

def find(el):
    if el[0] != el[1][0]:
        el[1] = find(el[1])
        return el[1]
    else:
        return el


def union(a, b):
    ra = find(a)
    rb = find(b)
    if ra != rb:
        if ra[2] > rb[2]:
            rb[1] = ra
        elif ra[2] < rb[2]:
            ra[1] = rb
        else:
            ra[1] = rb
            rb[2] += 1


def union_find(tgraph):
    forest = [[k.topology, [k.topology], 1] for k in tgraph.nodes.values()]
    keys = {el[0]: el for el in forest}
    for el in tgraph.nodes.values():
        for n in el.neighbors:
            union(keys[el.topology], keys[n.topology])
    return keys


def union_find2(tgraph):
    forest = [[k, [k], 1] for k in tgraph.keys()]
    keys = {el[0]: el for el in forest}
    for k,el in tgraph.items():
        for n in el:
            union(keys[k], keys[n])
    return keys



if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--threads', dest='threads', type=int, default=3)
    args = parser.parse_args()

    POOL = Pool(args.threads)
    """
