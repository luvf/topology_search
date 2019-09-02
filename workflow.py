import json

from injection_search import injection_search
from path_build import *
from path_build import run_scenarios, find_path
from topology_search import topology_search

if __name__ == '__main__':
    with open("config.json", "r") as f:
        config = json.load(f)
    for el in config.keys():
        if el != "n_threads":
            config[el]["n_threads"] = config["n_threads"]

    for i in range(0,15):
        #select_injections("chronics_contestants/chronics", "output", [i for i in range(3)], 100)
        print("select injections")
        #injection_search(config["injection_search"])  # , args.n, [i for i in range(args.n_chronics)], 35, args)
        print("injection search")
        config["topology_search"]["levels"]= config["injection_search"]["levels"]
        topology_search(config["topology_search"])
        print("train scenarios")
        run_scenarios(config["optimal_path"])
        print("find path")
        find_path(config["optimal_path"], str(i))