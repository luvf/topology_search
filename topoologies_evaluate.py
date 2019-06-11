import topology_Search
from topology_search import TopologyAgent



def topologies_evaluation(topologies):


def topology_search(topologies, K):

    topologies, np.array([el for el in toopologies.value() if len(el) > 1])

    indexes = np.maximum(topologies, axis = 1)

    primary_selected_topologies = [topologies.keys()[i] for i in indexes]+

    selected_topologies
    for t in primary_selected_topologies:




def topology_path(source, target, action_space):
    modifications = diff(source,dest)

def diff(source, dest, action_space):
    for l in action_space.lines_status_subaction_length:
        sline =  action_space.get_lines_status_switch_from_id(self.action_space.array_to_action(source), l)
        tline =  action_space.get_lines_status_switch_from_id(self.action_space.array_to_action(target), l)
        if sline == tline:
            new_el = action_space.get_do_nothing_action(True)
            new_el
            out.append()
    sline = action_space.array_to_action(source).get_lines_status_subaction()
    tline = action_space.array_to_action(tource).get_lines_status_subaction()
    line_indices  = [i for i in range(len(sline)) if sline[i] != tline[]]
    def set_line_status(action, el):
        action.lines_status_subaction[el] = 1
    [set_line_status(action_space.get_do_nothing_action(True),i)i in line_indices]

    get_lines_status_subaction()
    for n in action_space.substations_ids
        ssub = action_space.get_substation_switches_in_action(source, n)
        tsub = action_space.get_substation_switches_in_action(target, n)
        if ssub == tsub
            new_el = action_space.get_do_nothing_action()
            new_el = action_space.set_substation_switches_in_action(new_el, n, ssub-tsub)
            out.append(new_el)






class Best_policy_Search(object):
    def __init(self, topologies_score,dist_funct, sum_func=sum):
        self.topologies_score = topologies_score
        self.dist_funct = dist_funct
        self.sum_func= sum_func

    def
