from topology_search import actioned_component_mask
import IBEA
import matplotlib
import matplotlib.pyplot as plt

def plot_fitness(name):
    with open(os.path.join(IBEA.DIR, name + "saveF.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join("tmp2", "evaluated.json"), "r") as f:
        evaluated = json.load(f)
        devaluated = {tuple(k): v for k, v in evaluated}
    print(data[0][0][0])
    print(data[0][1][0])
    x = [i for i in range(len(data))]
    y = np.array([np.mean(d[0]) for d in data])
    yerri, yerrs = list(zip(*[(np.percentile(el[0], 10), np.percentile(el[0], 90)) for el in data]))

    # plt.errorbar(x,y, yerr=(y-yerri, yerrs-y))

    # plt.show()
    dims = len(evaluated[0][1])

    mins = [sum([min([devaluated[tuple(el)][k] for el in d[1]]) for k in range(dims)]) for d in data]
    means = np.array([np.mean([sum(devaluated[tuple(el)]) for el in d[1]]) for d in data])
    yerri = [np.percentile([sum(devaluated[tuple(el)]) for el in d[1]], 10) for d in data]
    yerrm = [np.percentile([sum(devaluated[tuple(el)]) for el in d[1]], 90) for d in data]

    plt.errorbar(x, means, yerr=(means - yerri, yerrm - means), label='Mean ' + name[:-1])
    plt.errorbar(x, mins, label="minimal " + name[:-1])
    # plt.show()



def deepth(name):
    with open(os.path.join(IBEA.DIR, name + "saveF.json"), "r") as f:
        data = json.load(f)
    ap = get_env().action_space
    mask = actioned_component_mask(ap)
    print(len(data), len(data[0]), len(data[0][1]))
    depth = [[sum([np.logical_and(top, el).any() for el in mask]) for top in deph[1]] for deph in data]
    # print(depth)
    means = np.array([np.mean(d) for d in depth])
    yerri = [np.percentile(eph, 10) for eph in depth]
    ymed = np.array([np.percentile(eph, 50) for eph in depth])
    yerrm = [np.percentile(eph, 90) for eph in depth]
    # print(data[0][1][23])
    # print(data[88][1][75])
    x = [i for i in range(len(data))]
    plt.errorbar(x, ymed, yerr=(ymed - yerri, yerrm - ymed), label="toto" + name)
    plt.plot(x, means, label="to" + name)






if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    name = IBEA.indicators[config["injection_search"]["indicator"]]
    plot_fitness(name)
    deepth(name)


