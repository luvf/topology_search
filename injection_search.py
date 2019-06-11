import os

import numpy as np


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from shutil import copyfile

from pypownet.environment import RunEnv
from pypownet.agent import DoNothing
from multiprocessing import Pool
import argparse
import json

import sys

#sys.path.append("../pypownet/")


NTHREADS = 1


def load_injections(directory, scenario_id):
    """
    Return an array of injections.

    :param directory:
    :param scenario_id:
    :return:
    """
    shape = get_shape("template")
    all_injections = np.zeros((0, sum(shape)))

    dirs = sorted(filter(lambda x: os.path.isdir(os.path.join(directory, x)), os.listdir(directory)))
    scenarios = [dirs[i] for i in scenario_id]
    for s in scenarios:
        inj = load_scenario(os.path.join(directory, s))
        all_injections = np.concatenate((all_injections, inj), axis=0)
    return all_injections, shape


def get_csv_content(csv_absolute_fpath):
    return np.genfromtxt(csv_absolute_fpath, dtype=np.float32, delimiter=';', skip_header=True)


def get_shape(directory):
    files = ['_N_loads_p.csv', '_N_loads_q.csv', '_N_prods_p.csv', '_N_prods_v.csv']
    data = [get_csv_content(os.path.join(directory, f)) for f in files]
    return [el.shape[1] for el in data]


def load_scenario(directory):
    files = ['_N_loads_p.csv', '_N_loads_q.csv', '_N_prods_p.csv', '_N_prods_v.csv']
    data = [get_csv_content(os.path.join(directory, f)) for f in files]
    return np.concatenate(data, axis=1)


def save_injections(dirname, injections, shape):
    """

    :param shape:
    :param dirname:
    :param injections:
    :return:
    """
    files = ["hazards.csv", '_N_loads_p_planned.csv', "_N_prods_v.csv", "maintenance.csv", "_N_loads_q.csv",
             "_N_prods_v_planned.csv", "_N_datetimes.csv", "_N_loads_q_planned.csv", "_N_simu_ids.csv", "_N_imaps.csv",
             "_N_prods_p.csv", "_N_loads_p.csv", "_N_prods_p_planned.csv"]
    mod_files = ['_N_loads_p.csv', '_N_loads_q.csv', '_N_prods_p.csv', '_N_prods_v.csv']
    for f in files:
        copyfile(os.path.join("template", f), os.path.join(dirname, 'chronics', '0', f))

    acc = 0
    for f, s in zip(mod_files, shape):
        to_save = injections[:, acc:acc + s]
        np.savetxt(os.path.join(dirname, 'chronics', '0', f), to_save, fmt="%.4f", delimiter=';')
        acc += s


def get_centers(injections, k):
    """

    :param injections: array of all injections to test
    :param k: number of clusters
    :return: center of clusters
    """
    pca = PCA(16)  # over 32
    injections_pca = pca.fit_transform(injections)
    print("PCA Done")
    kmeans = KMeans(n_clusters=k, algorithm="auto", verbose=1)
    kmeans.fit(injections_pca)
    centers = kmeans.cluster_centers_
    print("KMeans done")

    onenn = NearestNeighbors(1)
    onenn.fit(injections_pca)
    points = onenn.kneighbors(centers, return_distance=False)
    # KMEDIANES ....

    elements = [injections_pca[p[0]] for p in points]

    injections_centers = pca.inverse_transform(elements)
    return np.array(injections_centers)




def scenario_hard_injections(scenario_id):
    print(scenario_id)
    env = RunEnv(parameters_folder=".", game_level="chronics_contestants",
                 chronic_looping_mode="natural", start_id=scenario_id,
                 game_over_mode="soft",
                 without_overflow_cutoff=True)

    observation = env.get_observation()
    agent = DoNothing(env)
    hard_injections = list()
    injections, _ = load_injections("chronics_contestants/chronics", [scenario_id])
    for i in range(1000):
        action = agent.act(observation)

        observation, reward_aslist, done, info = env.step(action, do_sum=False)

        if reward_aslist[2] > 0:
            # print(i)
            # print(reward_aslist)
            hard_injections.append(injections[i + 1])

    return hard_injections


def select_difficult_injections(indexes, n_threads):
    pool = Pool(n_threads)
    injections = pool.map(scenario_hard_injections, indexes)

    return np.concatenate(injections)



def select_injections(outpudir, scenarios, n_centers, args):
    """

    :param inputdir:
    :param outpudir:
    :param scenarios:
    :param n_centers:
    :return:
    """
    # injections, shape = load_injections(inputdir, scenarios)
    shape = get_shape("template")

    first_injs = load_injections("chronics_contestants/chronics", [0])[0][:3]
    print("selecting difficult injections")
    hard_injections = select_difficult_injections(scenarios, args.threads)
    hard_injections.dump("output/hard_injection.npz")

    print("extracting centers")
    centers = get_centers(hard_injections, n_centers - 3)
    to_save = np.concatenate((first_injs, centers))

    print("saving")
    save_injections(outpudir, to_save, shape)



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--threads', dest='threads', type=int, default=1)
parser.add_argument('--n_chronics', dest='n_chronics', type=int, default=1)

args = parser.parse_args()


# select_injections("chronics_contestants/chronics", "output", [i for i in range(3)], 100)
select_injections("output", [i for i in range(args.n_chronics)], 100, args)
