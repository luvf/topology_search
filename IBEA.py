import argparse
import json
import logging
import math
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from pypownet_helper import get_env, get_all_dist_action, eval_topology

# import cProfile, pstats
# from pstats import SortKey

POOL = Pool(1)

DIR = "topology_search"


class IBEA:
    def __init__(self, alpha, gen, fit, objective, indic, name):
        self.alpha = alpha
        self.gen = gen  # nbde génération max
        self.fitval = fit
        self.cfit = 0
        self.indic = indic
        self.cur_gen = 0  # compteur de génération
        self.objective = objective
        self.cur_objective = list()
        self.cur_indic = np.zeros((0, 0))
        self.P = self.generate_pop(alpha)
        self.dim = len(self.P[0])
        self.outdim = len(objective(self.P[0]))
        self.actions = list(get_all_dist_action(get_env().action_space, 1))
        self.F = np.zeros(alpha)

        self.name = name
        self.trace = list()

    def fit(self):
        """
            Basic IBEA step 1, map x in P to F(x)
            calculle pour tout x_1 de p : $sum_(x_2 in P_(x1))-e^(-I(x_2,x_1)/k)$
        """
        # self.F.clear()
        # vI= (lambda x: np.vectorize(self.cur_indic)(np.array(list(self.P)),x))
        scale = self.fitval * self.cfit
        # print(self.cfit)
        if scale == 0:
            print("div par zero")
        for i in range(len(self.P)):
            self.F[i] = -np.sum(np.exp(self.cur_indic[:, i] / scale)) + 1

    def addaptive_fit(self):
        """
            Adaptive IBEA, rescale les fonctions objectif dans cur_objective
            construit cur_indic  en fonctions de ces nouvelles fonctions.

        """


        fx = self.objective.eval_list(self.P)
        mi = np.min(fx, axis=0)
        ma = np.max(fx, axis=0)
        mami = ma - mi
        cur_objective = [(self.objective(x) - mi) / mami for x in self.P]

        self.cfit = 0
        self.F = np.zeros(len(self.P))

        self.cur_indic = np.zeros((len(cur_objective), len(cur_objective)))

        for i, x in enumerate(cur_objective):
            for j, y in enumerate(cur_objective):
                cur = - self.indic(x, y)
                self.cur_indic[i, j] = cur
                self.cfit = max(self.cfit, abs(cur))

        self.fit()
        return

    def environemental_selection(self):
        """
            step 3
        """
        le = len(self.P)
        while le >= self.alpha:
            idx = np.argmin(self.F)  # self.F.items(), key=(lambda x: x[1]))[0]
            self.F[idx] = 2 ** 100
            self.P[idx] = False
            le -= 1
            self.update_f(idx)
        self.P = np.array([el for el in self.P if el is not False])
        self.trace.append((self.F.tolist(), self.P.tolist()))
        with open(os.path.join(DIR, self.name + "saveF.json"), "w") as ff:
            json.dump(self.trace, ff)

    def update_f(self, iel):
        """
            recompute F without el :
            F_1(x)= F(x)+ e^{-I(\{el\},\{x\}/(k*c)}
        """
        scale = self.fitval * self.cfit
        for j, x in enumerate(self.P):
            if x is not False:
                self.F[j] += math.exp(self.cur_indic[j, iel] / scale)

    def terminaison(self):
        """
            Sep4
        """
        return self.gen <= self.cur_gen

    def mating_selection(self, pop):
        """Mating selection aims at picking promising solutions for variation and
        usually is performed in a randomized fashion. Perform binary tournament
        selection with replacement on P in order to fill the temporary mating pool P_
        Params:
            P ---> pool of the population
        """
        mate = np.random.choice(range(len(pop)), (2, len(pop)))
        ff = np.vectorize(lambda x: self.F[x])

        f_pop = ff(mate)  # .array([[self.F[m] for m in mate[j]] for j in range(2)])
        mating_population = [mate[int(v), i] for i, v in enumerate(f_pop[0] < f_pop[1])]

        return np.array([pop[el] for el in mating_population])

    def variation(self, pop, mut_rate=0.05):
        """the mutation operator modifies individuals by changing small
        parts in the associated vectors according to a given mutation rate.

        Params:
            P_ ---> mating pool
            mut_rate ---> mutation rate by default is 1.0
            mu ---> 25
        """
        size_mutation = int(len(pop) * mut_rate)
        sample = np.random.choice(len(pop), int(size_mutation) + 1)
        out = list()
        for ind in sample:
            element = pop[ind]
            mutation = self.actions[np.random.randint(0, len(self.actions))]
            # np.random.choice([0, 1], len(element), p=[mut_rate, 1 - mut_rate])
            # if mutation.any():
            p_mut = np.logical_xor(element, mutation)
            out.append(p_mut)
        return np.array(out)

    @staticmethod
    def recombination(pop, recom_rate=1.0):
        """The recombination operator takes a certain number of parents and creates a
        predefined number of children by combining parts of the parents. To mimic the
        stochastic nature of evolution, a crossover probability is associated with this
        operator.

        Params:
            P_ ---> mating pool
            recom_rate ---> recombination rate by default is 1.0
        """
        size_recom = int(len(pop) * recom_rate)

        sample = np.random.choice(range(len(pop)), (size_recom, 2))  # Permutation
        out = list()

        for parent0, parent1 in sample:
            # Step 1: Choose a random number u 2 [0; 1).
            p0 = pop[parent0]
            p1 = pop[parent1]

            u = np.random.randint(0, 2, len(p0))
            # Step 2: Calculate  beta(q) using equation
            child0 = p0 * u + p1 * (1 - u)
            child1 = p1 * u + p0 * (1 - u)
            out.append(child0)
            out.append(child1)

        return np.array(out)

    @staticmethod
    def generate_pop(pop_size):
        """
            generate a pop_size set of random vectors in [-5, 5]^n
        """
        ap = get_env().action_space
        do_nothing = ap.get_do_nothing_action()
        pop = [do_nothing]
        d = 1
        while len(pop) < pop_size:
            elements = list(get_all_dist_action(ap, d))
            pop += elements[:pop_size]
            d += 1
        return np.array(pop)

    def run(self):
        # while not self.terminaison():
        self.addaptive_fit()
        for i in tqdm(range(self.gen)):
            mat = self.mating_selection(self.P)
            comb = self.recombination(mat)
            var = self.variation(comb)
            new_pop = {tuple(el) for el in np.concatenate((self.P, comb, var))}
            self.P = [np.array(el) for el in new_pop]

            self.addaptive_fit()
            self.environemental_selection()

            if i % 10 == 0:
                self.save(DIR, self.name)

    def save(self, directory, name):
        with open(os.path.join(directory, name + "pop.json"), "w") as savefile:
            json.dump([el.tolist() for el in self.P], savefile)
        self.objective.save(os.path.join(directory, "evaluated.json"))


def i_epsilon(al, bl):  # gerer les positifs négatifs
    m = -100
    for a, b in zip(al, bl):
        c = a - b
        if m < c:
            m = c
    return m


def I_H(el):
    return np.product(201 - el)


def I_HD(A, B):
    if (A - B > 0).any():
        return I_H(B) - I_H(np.max((A, B), axis=0))
    else:
        return I_H(B) - I_H(A)


indicators = {
    "ihd": I_HD,
    "iep": i_epsilon
}


def bin_epsilon(A, B):  # gerer les positifs négatifs
    x = A[0] - B[0]
    y = A[1] - B[1]
    if x > y:
        return x
    else:
        return y


class Function(object):
    def __init__(self, args, selected_injections="selected_injections_hard", file=None):
        self.evaluated = dict()
        self.current_evaluated=set()
        self.pareto = list()
        if file:
            self.load(file)

        self.len_evaluation = args["len_evaluation_chronic"]
        self.selected_injections = selected_injections

    def load(self, filename):
        with open(filename, 'r') as f:
            self.evaluated = {tuple(k): v for k, v in json.load(f)}

    def eval_list(self, top_list):
        eval_function = partial(self.eval_helper, len_evaluation=self.len_evaluation,
                                level=self.selected_injections)
        tlist = [tuple(t.tolist()) for t in top_list]
        unevaluated = list(filter(lambda x : x not in self.evaluated, tlist))
        #ev = filter(lambda x : x in self.evaluated, top_list)
        self.current_evaluated.update(unevaluated)

        values = POOL.map(eval_function, unevaluated)

        self.evaluated.update(values)
        return [self.evaluated[el] for el in tlist]

    def __call__(self, topology):
        top = tuple(topology.tolist())
        if top not in self.evaluated:
            value = eval_topology(top, ep_len=self.len_evaluation, lvl=self.selected_injections)
            self.evaluated[top] = value
            self.current_evaluated.add(top)
            return value
        else:
            return self.evaluated[top]

    def save(self, filename):
        with open(filename, "w") as savefile:
            json.dump(list(self.evaluated.items()), savefile)

    @staticmethod
    def eval_helper(top, len_evaluation, level):
       return top, [pow(x,1) for x in eval_topology(top, ep_len=len_evaluation, directory="parameters", lvl=level)]


def myIBEA(fun, pop_size, num_max_gen, fit_scale_fact, indic, run_name):
    # ibea1 = IBEA(pop_size, num_max_gen, fit_scale_fact, fun, i_epsilon, "iep_")
    # pr = cProfile.Profile()
    # pr.enable()
    # print('run I_epsilon')
    # ibea1.run()

    # ibea1.save(DIR, "iep_final")

    ibea = IBEA(pop_size, num_max_gen, fit_scale_fact, fun, indic, run_name)
    print("run I_HD")
    ibea.run()
    ibea.save(DIR, "run")

    return fun.current_evaluated


# pr.disable()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr).sort_stats(sortby)
# ps.print_stats()


def main(args, level, evaluated=None):
    ppn_logger = logging.getLogger("pypownet")
    ppn_logger.setLevel(logging.CRITICAL)

    global POOL
    POOL = Pool(args["n_threads"])

    opt_fun = Function(args, level, file=evaluated)  # =os.path.join(DIR, "evaluated.json")
    output = myIBEA(opt_fun,
                  int(args["pop_size"]),
                  int(args["epochs"]),
                  int(args["fit_scale"]),
                  indicators[args["indicator"]],
                  level)
    POOL.close()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--threads', dest='threads', type=int, default=2)
    args = parser.parse_args()

    main(args)
