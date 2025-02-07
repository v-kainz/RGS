import numpy as np

from modules.informationtheory import I0, Info
from modules.helperfunc import Counter


def fill(list, init = None):
    """takes list with nans, e.g. [nan, nan, 0.52752, 0.7426, nan, 0.6325] and fills the empty positions 
       with the number before. This example would return [X, X, 0.52752, 0.7426, 0.7426, 0.6325]. 
       The first elements X depends on the datatype -> how the quantity is initialized"""
    # 1st element:
    if np.isnan(list[0]):
        if type(init)==str:
            if init == 'info':
                list[0] = I0.mean
            elif init == 'rms':
                list[0] = I0.rms
            elif init == 'fr':
                pass
            elif init == 'kappa':
                list[0] = float(1)
            elif init == 'no_init' or init is None:
                pass
        elif type(init) == float or type(init) == int:
            list[0] = init
    # all other elements:
    for i in range(1,len(list)):
        if np.isnan(list[i]):
            list[i]=list[i-1]
        else:
            pass
    return list

def reset(a, line): 
    """gives agent a all the properties contained in dictionary line.
    Used to reset an agent when a game is continued."""
    a.id = line['id']
    a.name = line['name']
    a.x = line['x']
    a.I  =  [Info(entry[0],entry[1]) for entry in line['I']]# his beliefs
    a.Nc = Counter(line['Nc'])        # number of communications made
    a.Nt = Counter(line['Nt'])        # number of trues made
    a.Nl = Counter(line['Nl'])        # number of lies  made
    a.K  = line['K']       # surprises enCountered, initalized with 1
    a.kappa  = line['kappa']   # surprise scale sceptical
    a.fr_affinity = line['fr_affinity']
    a.friendships = [Info(entry[0],entry[1]) for entry in line['friendships']] # friendships
    a.shyness = line['shyness']
    a.relationsc = [Counter(_) for _ in line['relationsc']] # talked to someone
    a.relationsm = [Counter(_) for _ in line['relationsm']] # heard about someone
    a.Jothers = [[Info(entry[0], entry[1]) for entry in lst] for lst in line['Jothers']] # what others said last   
    a.Iothers = [[Info(entry[0], entry[1]) for entry in lst] for lst in line['Iothers']] # what others think   
    a.Cothers = [[Info(entry[0], entry[1]) for entry in lst] for lst in line['Cothers']] # what others think the agent thinks
    if 'prob_bt' in line.keys():
        assert 'prob_bl' in line.keys()
        assert len(line['prob_bt']) == len(line['prob_bl']) == 2
        a.prob_bt = Info(line['prob_bt'][0], line['prob_bt'][1])
        a.prob_bl = Info(line['prob_bl'][0], line['prob_bl'][1])
    a.openess  = line['openess']
    a.mind     = line['mind']
    if 'listening' in line.keys(): # included after update, old files might not have listening
        a.listening = line['listening']
    a.decepting    = line['decepting']
    a.disturbing = line['disturbing']
    a.strategic = line['strategic']
    a.egocentric= line['egocentric']
    a.flattering= line['flattering']
    a.aggressive= line['aggressive']
    a.shameless = line['shameless']
    if 'color' in line.keys(): # included after update, old files might not have color
        a.color = line['color']
        a.alpha = line['alpha']
    a.Buffer = line['Buffer']
    return a