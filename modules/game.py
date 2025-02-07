import json
import numpy as np
import yaml
from statistics import median

from modules.globals import event_buffer
from modules.helperfunc import Counter, Message, Comm_event
from modules.helperfunc_dependent import reset
from modules.informationtheory import Info
with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
ffl = config['parameters']['ffl']
fcaution = config['parameters']['fcaution']
NM = config['parameters']['NM']
RandomSeed = config['randomseed']['RandomSeed']
perc_one_to_one = config['parameters']['perc_one_to_one']

def make_data0(sim):
    """collects all characteristic parameters of a run and returns them as dictionary. 
    These parameters are required to check if a run with this configuration has already been made"""
    data0_constants = config

    data0_sim = {'x_est':[a.x for a in sim.A], 'fr_affinities':[a.fr_affinity for a in sim.A], 'shynesses':[a.shyness for a in sim.A], \
            'colors':sim.colors, 'alphas':sim.alphas, 'outfile':sim.filenames['outfile'], \
            'title':sim.filenames['title'], 'RSeed':sim.RSeed, 'mode':sim.mode, 'NA':sim.NA, 'NR':sim.NR}

    data0 = {**data0_sim, **data0_constants}
    return data0                

def play_game(sim): 
    data0 = make_data0(sim)  
    with open(sim.filenames['outfile'], 'w+') as f:
        json.dump(data0, f)
        f.write('\n')

    # save the initial configuration of all agents
    for a in sim.A: 
        with open(sim.filenames['outfile'], 'a') as f:
            initial_status = a.Agent_dict()
            initial_status = {**{'event_type': 'initial_status'}, **initial_status}
            json.dump(initial_status, f)
            f.write('\n')

    EventTime = Counter(0) # Time: two messages plus updates belong to one timestep
    EventNr = Counter(0) # Event-Number: each message and update has its own Event-Number
    random_dict = sim.random_dict
    for round in np.arange(sim.NR):
        shuffled_agents = np.copy(sim.NoA)
        random_dict['talking_order'].shuffle(shuffled_agents)
        for agent in shuffled_agents: 
            conversation(agent, sim, EventTime, EventNr)

    # write events from buffer to outputfile
    with open(sim.filenames['outfile'], 'a') as f:
        for event in event_buffer:
            # Serialize each event dictionary as a JSON string
            json_str = json.dumps(event)  # Convert the dictionary to a valid JSON string
            f.write(json_str + '\n')  # Write the JSON string followed by a newline
        event_buffer.clear()
    # save the final configuration of all agents
    for a in sim.A: 
        with open(sim.filenames['outfile'], 'a') as f:
            final_status = a.Agent_dict()
            final_status = {**{'event_type': 'final_status'}, **final_status}
            json.dump(final_status, f)
            f.write('\n')

def continue_game(arg_list):
    """reset simulation_config to current status and continue the game.
    Random number sequences are NOT continued but started again"""
    print(f"A previous game is continued.")

    sim, remaining_rounds, params, honesties, mindI, Ks, prob_bts, prob_bls, final_stati, content = arg_list

    # copy content
    with open(sim.filenames['outfile'], 'a') as f:
        for d in content:
            json.dump(d, f)
            f.write('\n')

    # reset counter
    EventTime = Counter(params['last_time']) # Time: two messages plus updates belong to one timestep
    EventNr = Counter(params['last_EventNr']) # Event-Number: each message and update has his own Event-Number

    # reset agents
    for a in sim.A:
        id = int(a.id)
        final_status = [_ for _ in final_stati if int(_['id'])==id]
        assert len(final_status) == 1
        a = reset(a, final_status.pop())

    # overwrite agents' stati with specifications
    if len(honesties) > 0:
        raise NotImplementedError('Changing honesties in continuation of run is not implemented yet. Specification of honesties must be an empty list.')
    if len(mindI) > 0:
        assert len(mindI) == len(sim.A) 
        for i in range(len(sim.A)):
            if not mindI[i] == 'default':
                assert len(mindI[i]) == len(sim.A) # list of [mu, lambda]-lists
                for j in range(len(sim.A)):
                    sim.A[i].I[j] = Info(mindI[i][j][0], mindI[i][j][1])
    if len(Ks) > 0:
        assert len(Ks) == len(sim.A) 
        for _ in range(len(sim.A)):
            if not Ks[_] == 'default':
                assert len(Ks[_]) == len(sim.A[_].K) # length of K array is the same
                sim.A[_].K = Ks[_]
                sim.A[_].kappa = median(sim.A[_].K)/np.sqrt(np.pi)
    if len(prob_bts) > 0 and content[0]['switches']['EBEH']:
        assert len(prob_bts) == len(sim.A) 
        for _ in range(len(sim.A)):
            if not prob_bts[_] == 'default':
                assert len(prob_bts[_]) == 2 # list of mu and lambda
                sim.A[_].prob_bt = Info(prob_bts[_][0], prob_bts[_][1])
    if len(prob_bls) > 0 and content[0]['switches']['EBEH']:
        assert len(prob_bls) == len(sim.A) 
        for _ in range(len(sim.A)):
            if not prob_bls[_] == 'default':
                assert len(prob_bls[_]) == 2 # list of mu and lambda
                sim.A[_].prob_bl = Info(prob_bls[_][0], prob_bls[_][1])

    # continue the game
    for round in np.arange(remaining_rounds): # get-together rounds have already been subtracted
        shuffled_agents = np.copy(np.array(sim.NoA))
        np.random.shuffle(shuffled_agents)
        for agent in shuffled_agents: 
            conversation(agent, sim, EventTime, EventNr)

    # write events from buffer to outputfile
    with open(sim.filenames['outfile'], 'a') as f:
        for event in event_buffer:
            # Serialize each event dictionary as a JSON string
            json_str = json.dumps(event)  # Convert the dictionary to a valid JSON string
            f.write(json_str + '\n')  # Write the JSON string followed by a newline
        event_buffer.clear()
    # save the final configuration of all agents
    for a in sim.A: 
        with open(sim.filenames['outfile'], 'a') as f:
            final_status = a.Agent_dict()
            final_status = {**{'event_type': 'final_status'}, **final_status}
            json.dump(final_status, f)
            f.write('\n')

def conversation(a, simulation, EventTime, EventNr):
    """Each round consists of one conversation (back and forth communication of an opinion) plus 
       the participating agents' updates.
       Here, agent a will talk to the agent b about c.

    Args:
        a (Agent): the agent initiating the conversation
        simulation (Simulation): the instance of the Simulation class, defining
            the simulation parameters
        EventTime (Counter): The simulation time counter
        EventNr (int): Unique event number in the simulation.

    Returns:
        _type_: _description_
    """
    agent_list = simulation.A
    random_dict = simulation.random_dict
    filenames = simulation.filenames

    one_to_one = random_dict['one_to_one'].choice([True, False], p=[perc_one_to_one, 1-perc_one_to_one])

    # forward random variables
    a_now_egocentric = agent_list[a].egocentric > random_dict['ego'].uniform() or (agent_list[a].strategy=='MoLG' and not one_to_one)
    a_now_aggressive = agent_list[a].aggressive > random_dict['aggressive'].uniform()
    a_now_flattering = agent_list[a].flattering > random_dict['flattering'].uniform() or (agent_list[a].strategy=='MoLG' and one_to_one)
    forward_now_honest = agent_list[a].x >= random_dict['honests'].uniform()
    if not forward_now_honest:
        forward_now_blush = random_dict['blush'].uniform()<ffl
    else:
        forward_now_blush = False
    forward_now_KL_target = random_dict['lies'].exponential(agent_list[a].kappa)*fcaution
    
    # Choose Communication Partner 
    N_rec_weights = agent_list[a].draw_number_of_receivers(one_to_one)
    N_rec = random_dict['N_recipients'].choice([_ for _ in range(1,len(agent_list))], p=N_rec_weights)
    b_weights = agent_list[a].draw_communication_partner(agent_list, one_to_one)
    b_set = tuple(random_dict['recipients'].choice([i for i in range(len(agent_list))], size = N_rec, replace=False, p=b_weights))

    # Choose Topic
    if a_now_flattering:
        possible_flattering_targets_indices = agent_list[a].get_flattering_target(b_set)
        flattering_target_index = random_dict['flattering'].choice(possible_flattering_targets_indices)
        flattering_target = b_set[flattering_target_index]
    else:
        flattering_target = None
    c_weights = agent_list[a].draw_topic(agent_list, b_set, a_now_egocentric, a_now_flattering, a_now_aggressive, flattering_target, one_to_one)
    c = random_dict['topic'].choice([i for i in range(len(agent_list))], p=c_weights)

    EventTime.inc() 

    # Forward communication
    forward   = agent_list[a].talk(c, b_set, b_weights, agent_list, forward_now_honest, forward_now_blush, forward_now_KL_target, a_now_aggressive, one_to_one, filenames, EventTime, EventNr)  # initial  communication
    for recipient in b_set:
        agent_list[recipient].Buffer.append(forward)

    EventTime.inc()
    
    # Backward
    if one_to_one: # recipient answers  
        assert len(b_set) == 1
        b = b_set[0]

        # backward random variables
        b_now_aggressive = agent_list[b].aggressive > random_dict['aggressive'].uniform()
        backward_now_honest = agent_list[b].x >= random_dict['honests'].uniform()
        if not backward_now_honest:
            backward_now_blush = random_dict['blush'].uniform()<ffl
        else:
            backward_now_blush = False
        backward_now_KL_target = random_dict['lies'].exponential(agent_list[b].kappa)*fcaution
        
        # backward communication
        backward_b_weights = np.array([1]*len(agent_list)) / len(agent_list)  
        backward_b_set = (a,)   
        backward = agent_list[b].talk(c, backward_b_set, backward_b_weights, agent_list, backward_now_honest, backward_now_blush, backward_now_KL_target, b_now_aggressive, one_to_one, filenames, EventTime, EventNr)    # response communication, b_weights is just [1] here
        agent_list[a].Buffer.append(backward)

    # Update knowledge for all who received a message
    for i in range(len(agent_list)):
        if len(agent_list[i].Buffer) == NM:
            agent_list[i].update_from_Buffer(agent_list, filenames, EventNr)
    
def play_propaganda(sim, Ip):
    mode_str = sim.mode['all']
    sim.filenames['title']="isolated "+ mode_str + " agents under propaganda"

    EventTime = Counter(0) # Time: two messages plus updates belong to one timestep
    EventNr = Counter(0) # Event-Number: each message and update has his own Event-Number
    # initial status of agents:
    sim.A[0].x    = 0
    sim.A[0].shameless = True
    sim.A[1].I[0] = Info(3,0) #unsceptical start
    sim.A[2].I[0] = Info(0,0) #neutral start
    sim.A[3].I[0] = Info(0,3) #sceptical start
    # save parameters
    data0 = make_data0(sim)
    with open(sim.filenames['outfile'], 'w+') as f:
        json.dump(data0, f)
        f.write('\n')
    # save initial status of agents
    for i in range(1, sim.NA):
        data = {'event_type':'initial_status', 'id': str(i), 'I_0':sim.A[i].I[0].to_list()}
        with open(sim.filenames['outfile'], 'a') as f:
            json.dump(data, f)
            f.write('\n')
    # play the game
    for round in np.arange(sim.NR):
        print('Round ', round)
        for b in range(1, sim.NA):
            propaganda(0,b,0,Ip, sim, EventTime, EventNr)

    # write events from buffer to outputfile
    with open(sim.filenames['outfile'], 'a') as f:
        data_save = []
        for i in range(len(event_buffer)):
            data_save.append(str(event_buffer[i]).replace("'", '"').replace('True', 'true').replace('False', 'false'))
            data_save.append('\n')
        f.writelines(data_save)
        event_buffer.clear() # empty for next run


def play_antipropagangda(sim, Ip):
    mode_str = sim.mode['all']
    sim.filenames['title']="honestly communicating "+ mode_str +" agents under propaganda"
    # use different seeds
    if mode_str.lower() == 'smart':
        sim.filenames['title'] = "honestly communicating smart and ordinary agents under propaganda"
        sim.A[2].mind = 'smart'
    
    EventTime = Counter(0) # Time: two messages plus updates belong to one timestep
    EventNr = Counter(0) # Event-Number: each message and update has his own Event-Number
    # initial status for agents:
    sim.A[0].x    = 0
    sim.A[0].shameless = True
    sim.A[1].I[0] = Info(3,0) #unsceptical start
    sim.A[2].I[0] = Info(0,0) #neutral start
    sim.A[3].I[0] = Info(0,3) #sceptical start
    # save parameters
    data0 = make_data0(sim)
    with open(sim.filenames['outfile'], 'w+') as f:
        json.dump(data0, f)
        f.write('\n')
    # save initial status of agents
    for i in range(1, sim.NA):
        data = {'event_type':'initial_status', 'id': str(i), 'I_0':sim.A[i].I[0].to_list()}
        with open(sim.filenames['outfile'], 'a') as f:
            json.dump(data, f)
            f.write('\n')
    # play the game    
    for round in np.arange(sim.NR):
        print('Round ', round)
        for b in range(1, sim.NA):
            propaganda(0, b, 0, Ip, sim, EventTime, EventNr)
            r = b+1 # recipient
            if r not in range(sim.NA):
                r = r%sim.NA +1
            antipropaganda(b, r, 0, sim, EventTime, EventNr)
            r = b+2
            if r not in range(sim.NA):
                r = r%sim.NA +1
            antipropaganda(b, r, 0, sim, EventTime, EventNr)
    # write events from buffer to outputfile
    with open(sim.filenames['outfile'], 'a') as f:
        data_save = []
        for i in range(len(event_buffer)):
            data_save.append(str(event_buffer[i]).replace("'", '"').replace('True', 'true').replace('False', 'false'))
            data_save.append('\n')
        f.writelines(data_save)
        event_buffer.clear() # empty for next run


def propaganda(a, b, c, Ip, sim, EventTime, EventNr):
    EventTime.inc()
    EventNr.inc()
    if sim.random_dict['blush'].uniform()<ffl and not sim.A[a].shameless:
        blush = True
        print('BLUSH')
    else:
        blush = False
    m = Message(c,a,[b],Ip, False, blush, EventTime, EventNr) # c, a, b, J, honest, blush
    c = Comm_event()
    c.event_type = 'propaganda'
    c.comm = m.message_dict()
    c.save(sim.filenames)
    sim.A[a].awarness(m, sim.A, sim.filenames, EventNr)
    if sim.parameters['listening']:
        EventNr.inc()
        sim.A[b].listen(m, sim.A, sim.filenames, EventNr)
    
def antipropaganda(a, b, c, sim, EventTime, EventNr):
    EventTime.inc()
    EventNr.inc()
    blush = False
    m = Message(c,a,[b], sim.A[a].I[c], True, blush, EventTime, EventNr)
    c = Comm_event()
    c.event_type = 'antipropaganda'
    c.comm = m.message_dict()
    c.save(sim.filenames)
    sim.A[a].awarness(m, sim.A, sim.filenames, EventNr)
    if sim.parameters['listening']:
        EventNr.inc()
        sim.A[b].listen(m, sim.A, sim.filenames, EventNr)
