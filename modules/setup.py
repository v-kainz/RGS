import os
import numpy as np
import yaml
from statistics import median

from modules.globals import names
from modules.agents import Agent
from modules.informationtheory import Info
with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
RandomSeed = config['randomseed']['RandomSeed']
Q = config['parameters']['Q']
RANDOM_HONESTIES = config['switches']['RANDOM_HONESTIES']
EBEH = config['switches']['EBEH']

class simulation_config:
    """All setup parameters for a simulation run. Agents get created and their personalities adapted according to the mode."""
    def __init__(self, NA, NR, mode, honesties, mindI, Ks, prob_bts, prob_bls, RSeed, folder, subfolder, sort_by_honesty=True, prop = None): 
        self.RSeed = RSeed
        self.NA = NA
        self.NR = NR
        self.NoA = np.arange(NA)
        self.mode = mode
        self.basis_mode = mode['all'] if 'all' in mode.keys() else 'ordinary'
        self.colors = []
        self.Q = Q
        # check if mode is suitable for propaganda runs
        if prop:
            assert len(mode.keys())==1 and list(mode.keys())[0] == 'all', 'propaganda simulations are only implemented for equal agents. Please specify the modes like {"all": "some_strategy"}'
            if prop == 1:
                assert list(mode.values())[0] in ["uncritical","ordinary"], f'Invalid mode for propaganda: {mode["all"]}. Please choose one of ["uncritical","ordinary"]'
            elif prop == -1:
                assert list(mode.values())[0] in ["ordinary","smart","Smart","SMart"], f'Invalid mode for antipropaganda: {mode["all"]}. Please choose one of ["ordinary","smart","Smart","SMart"]'
        # check if mode and NA are compatible
        basis_mode_overwritten = False # default. may change a few lines below
        if not 'all' in mode.keys():
            assert len(mode.keys()) <= self.NA, f'you have specified {len(mode.keys())} characters but only {self.NA} participating agents.'
            if len(mode.keys()) == self.NA:
                basis_mode_overwritten = True
        else:
            if len(mode.keys()) == self.NA + 1:
                print(f'WARNING: no agent will be {self.basis_mode} as you specified special characters for all agents individually.')
                basis_mode_overwritten = True
            elif len(mode.keys()) > self.NA + 1:
                raise ValueError(f'you have specified {len(mode.keys())-1} characters but only {self.NA} participating agents. \nAlso consider that the character specified for all agents will not be implemented when special characters for all indvidual agents are given.')
        # make filename according to participating agents
        if 'all' in mode.keys() and mode['all']=='ordinary' and len(mode)>1: mode.pop('all', None)
        mode_name_snippets = [f"{names[strategy]}_{'_'.join([k for k,v in mode.items() if v == strategy])}" for strategy in set(mode.values())]
        mode_name_snippets.sort()
        mode_name = '__'.join(mode_name_snippets)
        self.name = mode_name
        self.name += '_NA'+str(NA)
        self.name += '_RS'+str(RSeed)
        
        self.parameters = {'basis_mode': self.basis_mode, 'decepting': True, 'listening': True, 'Disturbing': False} # dictionary for semi-constant global parameters
        self.filenames = {'folder': folder, 'subfolder': subfolder, 'title': '', 'outfile': ''} # dictionary for filenames
        
        # Random Number Chains
        self.random_honesties = np.random.seed(RandomSeed)  # determines honsesties (constant wrt RSeed)
        self.varying_random_honesties = np.random.RandomState()
        self.varying_random_honesties.seed(RandomSeed+RSeed) # determines honesties (varying with RSeed)
        self.talking_order = np.random.RandomState()
        self.talking_order.seed(RandomSeed+RSeed) # in which order they initiate a conversation in each round
        self.random_fr_affinities = np.random.RandomState()
        self.random_fr_affinities.seed(RandomSeed+RSeed+1) # determines friendship affinities
        self.random_shynesses = np.random.RandomState()
        self.random_shynesses.seed(RandomSeed+RSeed+2) # determines shyness
        self.random_one_to_one = np.random.RandomState() # determines when a one-to-one-conversation takes places (or one-to-many else
        self.random_one_to_one.seed(RandomSeed+10*RSeed-1) 
        self.random_N_recipients = np.random.RandomState() # random stream determining recipients (constant wrt RSeed)
        self.random_N_recipients.seed(RandomSeed+10*RSeed+0) 
        self.random_recipients = np.random.RandomState() # random stream determining recipients (constant wrt RSeed)
        self.random_recipients.seed(RandomSeed+10*RSeed+1)  
        self.random_N_recipients = np.random.RandomState() # random stream determining recipients (constant wrt RSeed)
        self.random_N_recipients.seed(RandomSeed+10*RSeed+0)  
        self.random_topic = np.random.RandomState() # random stream determining resipients
        self.random_topic.seed(RandomSeed+10*RSeed+2)  
        self.random_honests = np.random.RandomState() # random stream determining honest statements
        self.random_honests.seed(RandomSeed+10*RSeed+3) 
        self.random_lies = np.random.RandomState() # random stream determining lies
        self.random_lies.seed(RandomSeed+10*RSeed+4) # separate stream, so that switching on/off lies does not affect sequence
        self.random_blush = np.random.RandomState() # random stream determining lies
        self.random_blush.seed(RandomSeed+10*RSeed+5) # separate stream, so that switching on/off lies does not affect sequence
        self.random_ego = np.random.RandomState() # random stream determining egocentric
        self.random_ego.seed(RandomSeed+10*RSeed+6) # separate stream, so that switching on/off ego does not affect sequence
        self.random_strategic = np.random.RandomState() # random stream determining strategic
        self.random_strategic.seed(RandomSeed+10*RSeed+7) # separate stream, so that switching on/off strategic does not affect sequence
        self.random_flattering = np.random.RandomState() # random stream determining flattering
        self.random_flattering.seed(RandomSeed+10*RSeed+8) # separate stream, so that switching on/off flattering does not affect sequence
        self.random_aggressive = np.random.RandomState() # random stream determining aggressive behaviour
        self.random_aggressive.seed(RandomSeed+10*RSeed+9) # separate stream, so that switching on/off does not affect sequence
        
        self.random_dict = {'honesties':self.random_honesties, 'varying_honesties':self.varying_random_honesties, 'talking_order': self.talking_order, 'fr_affinities':self.random_fr_affinities, 'shynesses':self.random_shynesses, \
                        'one_to_one':self.random_one_to_one, 'N_recipients':self.random_N_recipients, 'recipients':self.random_recipients, 'topic':self.random_topic, \
                        'honests':self.random_honests, 'lies':self.random_lies, 'blush':self.random_blush, \
                        'ego':self.random_ego, 'strategic':self.random_strategic, 'flattering':self.random_flattering, \
                        'aggressive':self.random_aggressive}

        # set up agents
        if len(honesties) > 0:
            assert len(honesties) == len(self.NoA)
            if all([type(_)==int or type(_)==float for _ in honesties]):
                self.A = np.array([Agent(self.NoA, j, self.random_dict, x = honesties[j], decepting = self.parameters['decepting']) for j in self.NoA]) # array with agents
            elif 'random' in honesties:
                assert not 'default' in honesties, 'combination of random and default honesties has not been implemented yet.'
                # make all random, then fill given numbers
                variable_random_honesties = self.varying_random_honesties.uniform(0,1,size=len(self.NoA))
                for _ in range(len(honesties)):
                    if type(honesties[_])==int or type(honesties[_])==float:
                        variable_random_honesties[_] = honesties[_]
                self.A = np.array([Agent(self.NoA, j, self.random_dict, x = variable_random_honesties[j], decepting = self.parameters['decepting']) for j in self.NoA]) # array with agents
            elif 'default' in honesties:
                assert not 'random' in honesties, 'combination of random and default honesties has not been implemented yet.'
                # depends on what is default
                if RANDOM_HONESTIES:
                    self.A = np.array([Agent(self.NoA, j, self.random_dict, decepting = self.parameters['decepting']) for j in self.NoA]) # array with agents
                    for _ in range(len(honesties)):
                        if type(honesties[_])==int or type(honesties[_])==float:
                            self.A[_].x = honesties[_]
                            self.A[_].original_x = honesties[_]
                else:
                    lin_honesties = np.linspace(0,1,len(self.NoA))
                    for _ in range(len(honesties)):
                        if type(honesties[_])==int or type(honesties[_])==float:
                            lin_honesties[_] = honesties[_]
                    self.A = np.array([Agent(self.NoA, j, self.random_dict, x = lin_honesties[j], decepting = self.parameters['decepting']) for j in self.NoA]) # array with agents         
        elif RANDOM_HONESTIES:
            self.A = np.array([Agent(self.NoA, j, self.random_dict, decepting = self.parameters['decepting']) for j in self.NoA]) # array with agents
        else:
            lin_honesties = np.linspace(0,1,len(self.NoA))
            self.A = np.array([Agent(self.NoA, j, self.random_dict, x = lin_honesties[j], decepting = self.parameters['decepting']) for j in self.NoA]) # array with agents
        
        if sort_by_honesty:
            # sort everythig by honesty of agents
            self.A = sorted(self.A, key=lambda a: a.x, reverse=False)

        identities = [a.id for a in self.A] # either sorted or not
        for j in self.NoA:
            self.A[j].id = j # fixes mixed up identities of sorting
            self.A[j].name = str(j)
            self.A[j].friendships[j] = Info(1,0) # initialize self-friendship
        self.colors = [a.color for a in self.A]
        self.alphas = [a.alpha for a in self.A]
        if len(mindI)>0: 
            mindI = [mindI[_] for _ in identities]
            for _ in self.NoA:
                if type(mindI[_])==list:
                    assert len(mindI[_])==len(self.NoA)
                    mindI[_] = [mindI[_][__] for __ in identities]
        if len(Ks)>0: Ks = [Ks[_] for _ in identities]
        if len(prob_bts)>0: prob_bts = [prob_bts[_] for _ in identities]
        if len(prob_bls)>0: prob_bls = [prob_bls[_] for _ in identities]

        def make_character(a, mode):
            """changes the agent's character to mode"""
            def reset(a):
                # store original random features
                original_id = a.id
                original_x = a.original_x
                original_fr_affinity = a.original_fr_affinity
                original_shyness = a.original_shyness
                original_friendships = a.friendships
                # make default
                a = Agent(self.NoA, original_id, self.random_dict, decepting = self.parameters['decepting'])
                # give back original random features + copies (because it might be overwritten again)
                a.x = original_x
                a.original_x = original_x
                a.fr_affinity = original_fr_affinity
                a.original_fr_affinity = original_fr_affinity
                a.shyness = original_shyness
                a.original_shyness = original_shyness
                a.friendships = original_friendships
                a.original_friendships = original_friendships
                return a

            def make_deaf(a):
                a = reset(a)
                a.mind = 'deaf'
                a.listening = False
                a.strategy = 'deaf'
                return a
                
            def make_naive(a):
                a = reset(a)
                a.mind = 'naive'
                a.strategy = 'naive'
                return a
            def make_uncritical(a):
                a = reset(a)
                a.mind = 'uncritical'
                a.strategy = 'uncritical'
                return a
            def make_honest(a):
                a = reset(a)
                a.x = 1
                a.strategy = 'honest'
                return a
            def make_ordinary(a):
                a = reset(a)
                a.strategy = 'ordinary'
                return a
            def make_strategic(a):
                a = reset(a)
                a.strategic = 1
                # random for the moment
                # when this is not random anymore, care about setting fr_affinity and shyness back to original when character is overwritten!
                #a.fr_affinity = 10**(-3*fr_affinity_log_std) # 3 sigma
                #a.shyness = 10**(-3*shyness_log_std) # 3 sigma
                a.strategy = 'strategic'
                return a
            def make_antistrategic(a):
                a = reset(a)
                a.strategic = -1
                a.strategy = 'antistrategic'
                return a
            def make_egocentric(a):
                a = reset(a)
                a.egocentric = 0.5
                # random for the moment
                #a.fr_affinity = 10**(+3*fr_affinity_log_std) # high
                #a.shyness = 10**(-3*shyness_log_std) # low
                a.strategy = 'egocentric'
                return a
            def make_deceptive(a):
                a = reset(a)
                a.x = 0
                a.strategy = 'deceptive'
                return a
            def make_flattering(a):
                a = reset(a)
                a.flattering = 1
                a.strategy = 'flattering'
                return a
            def make_aggressive(a):
                a = reset(a)
                a.aggressive = 1
                a.strategy = 'aggressive'
                return a
            def make_shameless(a):
                a = reset(a)
                a.shameless = True
                a.strategy = 'shameless'
                return a
            def make_smart(a):
                a = reset(a)
                a.mind = 'smart'
                a.strategy = 'smart'
                return a
            def make_clever(a):
                a = reset(a)
                a.mind = 'smart'
                a.x = 0
                a.strategy = 'clever'
                return a
            def make_manipulative(a):
                a = reset(a)
                a.x = 0 # deceptive agent
                a.strategic  = -1 #antistrategic agent 
                a.flattering = 1 # ingratuating agent 
                # random for the moment
                #a.fr_affinity = 10**(-3*fr_affinity_log_std) # 3 sigma
                #a.shyness = 10**(-3*shyness_log_std) # 3 sigma
                #a.shyness = 0.3
                a.mind = 'smart'  # smart agent  
                a.strategy = 'manipulative'
                return a
            def make_dominant(a):
                a = reset(a)
                a.x = 0 # deceptive agent
                a.strategic  = 1 # strategic agent 
                a.egocentric = 0.5 # egocentric agent
                # random for the moment
                #a.fr_affinity = 10**(-3*fr_affinity_log_std) # 3 sigma
                #a.shyness = 10**(-3*shyness_log_std) # 3 sigma
                #a.shyness = 0.3
                a.mind = 'smart'  # smart agent 
                a.strategy = 'dominant'
                return a
            def make_destructive(a):
                a = reset(a)
                a.x = 0 # deceptive agent
                a.strategic  = 1 # strategic agent 
                a.shameless = True
                a.aggressive = 1
                # random for the moment
                #a.fr_affinity = 10**(-3*fr_affinity_log_std) # 3 sigma
                #a.shyness = 10**(-3*shyness_log_std) # 3 sigma
                #a.shyness = 0.3
                a.mind = 'smart'  # smart agent   
                a.strategy = 'destructive'
                return a 
            def make_good(a):
                a = reset(a)
                a.x = 1
                a.mind = 'smart'
                a.strategic = 1 
                a.strategy = 'good'
                return a
            def make_MoLG(a):
                # manipulative agent in one2one conversations, dominant agent in one2many conversations
                a = reset(a)
                a.x = 0 # deceptive agent
                a.mind = 'smart'  # smart agent  
                ##a.mind = 'ordinary'
                a.shyness = 0.3
                a.strategy = 'MoLG'
                a.use_acquaintance = False # for the choice of communication partner and topic
                return a

            if mode == "deaf":
                a = make_deaf(a)            
            elif mode == "naive":
                a = make_naive(a)    
            elif mode == "uncritical+honest":
                a = make_uncritical(a)
                a = make_honest(a)
            elif mode == "uncritical":
                a = make_uncritical(a) 
            elif mode == "ordinary":
                a = make_ordinary(a)
            elif mode == "strategic":
                a = make_strategic(a)  
            elif mode == 'antistrategic':
                a = make_antistrategic(a)  
            elif mode == "egocentric":
                a = make_egocentric(a)        
            elif mode == "deceptive":
                a = make_deceptive(a)            
            elif mode == "flattering":
                a = make_flattering(a)
            elif mode == "aggressive":
                a = make_aggressive(a)
            elif mode == "shameless":
                a = make_shameless(a)        
            elif mode == "smart":
                a = make_smart(a)            
            elif mode.lower() == 'smart':
                a = make_smart(a)
            elif mode == "clever":
                a = make_clever(a)
            elif mode == "manipulative":
                a = make_manipulative(a)    
            elif mode == "dominant":
                a = make_dominant(a)
            elif mode == "destructive":
                a = make_destructive(a)                
            elif mode == "good":
                a = make_good(a)
            elif mode == 'MoLG':
                a = make_MoLG(a)
            elif mode == 'disturbing':
                a.disturbing = True
                self.name = self.name + '_disturb'
            else:
                raise NotImplementedError(f'mode {mode} not implemented yet.')
            return a

        # give agents their characters
        key_list = list(mode.keys())
        if 'all' in key_list: # do that first
            for ii, a in enumerate(self.A):
                self.A[ii] = make_character(a, mode['all'])
            key_list.remove('all')
        for key in key_list: # all other keys
            self.A[int(key)] = make_character(self.A[int(key)], mode[key])

        # initialize agents according to specification in config file
        if len(mindI) > 0:
            assert len(mindI) == len(self.NoA) 
            for i in range(len(self.NoA)):
                if not mindI[i] == 'default':
                    assert len(mindI[i]) == len(self.NoA) # list of [mu, lambda]-lists
                    for j in range(len(self.NoA)):
                        self.A[i].I[j] = Info(mindI[i][j][0], mindI[i][j][1])
        if len(Ks) > 0:
            assert len(Ks) == len(self.NoA) 
            for _ in range(len(self.NoA)):
                if not Ks[_] == 'default':
                    assert len(Ks[_]) == len(self.A[_].K) # length of K array is the same
                    self.A[_].K = Ks[_]
                    self.A[_].kappa = median(self.A[_].K)/np.sqrt(np.pi)
        if len(prob_bts) > 0 and EBEH:
            assert len(prob_bts) == len(self.NoA) 
            for _ in range(len(self.NoA)):
                if not prob_bts[_] == 'default':
                    assert len(prob_bts[_]) == 2 # list of mu and lambda
                    self.A[_].prob_bt = Info(prob_bts[_][0], prob_bts[_][1])
        if len(prob_bls) > 0 and EBEH:
            assert len(prob_bls) == len(self.NoA) 
            for _ in range(len(self.NoA)):
                if not prob_bls[_] == 'default':
                    assert len(prob_bls[_]) == 2 # list of mu and lambda
                    self.A[_].prob_bl = Info(prob_bls[_][0], prob_bls[_][1])

        # make title for plots
        normal = self.basis_mode
        all_modes = list(mode.values())
        if normal in all_modes:
            all_modes.remove(normal)
        for m in all_modes:
            self.filenames['title'] += m + ','
        self.filenames['title'] = self.filenames['title'][:-1] # remove last comma
        if all_modes == []: # only ordinary agents
            self.filenames['title'] = normal + ' agents'
        elif all([_==normal for _ in all_modes]):
            self.filenames['title'] = normal + ' agents'
        elif len(all_modes) == self.NA:
            self.filenames['title'] += ' agents'
        elif len(all_modes) == self.NA - 1:
            self.filenames['title'] += ' among '+ normal +' agent'
        else:
            self.filenames['title'] += ' among '+ normal +' agents'
        
        # name output file
        self.filenames['outfile'] = self.filenames['folder'] + "/" + self.filenames['subfolder'] + "/" + self.name + '.json'
        if prop == 1:
            self.filenames['outfile'] = self.filenames['folder'] + "/" + self.filenames['subfolder'] + "/" + self.name + '_propaganda.json'
        elif prop == -1:
            self.filenames['outfile'] = self.filenames['folder'] + "/" + self.filenames['subfolder'] + "/" + self.name + '_antipropaganda.json'



def simulation_config_from_parameters(NR, folder, subfolder, params, RSeed_cont):
    NA = params['NA']
    NR = NR # not from previous file, but new one
    mode = params['mode']
    RSeed = params['RSeed'] + RSeed_cont
    title = params['title']
    outfile = params['outfile']
    if 'antipropaganda' in outfile:
        prop = -1
    elif 'propaganda' in outfile:
        prop = 1
    else: 
        prop = 0

    sim = simulation_config(NA, NR, mode, [], [], [], [], [], RSeed, folder, subfolder, prop=prop)
    return sim