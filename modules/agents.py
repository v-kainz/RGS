import yaml
import numpy as np
from numpy import exp
from statistics import median
from scipy.optimize import root

from modules.informationtheory import I0, KL, Info, DeltaInfo, Sc_dist_ratio, match, make_average_opinion
from modules.helperfunc import Counter, Message, Comm_event, Update_event, convert_name
with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
ffl = config['parameters']['ffl']
fcaution = config['parameters']['fcaution']
verbose = config['verbosity']['verbose']
ultraverbose = config['verbosity']['ultraverbose']
MaxCount = float(config['constants']['MaxCount'])
Q = config['parameters']['Q']
Acolor = config['constants']['Acolor']*100
Aalpha = config['constants']['Aalpha']*100
tiny = float(config['constants']['tiny'])
EBEH = config['switches']['EBEH']
EBEH_self_awareness = config['switches']['EBEH_self_awareness']
CONTINUOUS_FRIENDSHIP = config['switches']['CONTINUOUS_FRIENDSHIP']
ACTIVE_SELF_FRIENDSHIP = config['switches']['ACTIVE_SELF_FRIENDSHIP']
SCALED_FLATTERING = config['switches']['SCALED_FLATTERING']
FRIENDSHIP_AFFECTS_B = config['switches']['FRIENDSHIP_AFFECTS_B']
FRIENDSHIP_AFFECTS_C = config['switches']['FRIENDSHIP_AFFECTS_C']
RELATION_AFFECTS_B = config['switches']['RELATION_AFFECTS_B']
RELATION_AFFECTS_C = config['switches']['RELATION_AFFECTS_C']
HOMOPHOLY_AFFECTS_B = config['switches']['HOMOPHOLY_AFFECTS_B']
HOMOPHOLY_AFFECTS_C = config['switches']['HOMOPHOLY_AFFECTS_C']
fr_affinity_log_mean = config['parameters']['fr_affinity_log_mean'] if config['switches']['RANDOM_FRIENDSHIP_AFFINITIES'] else 0
fr_affinity_log_std = config['parameters']['fr_affinity_log_std'] if config['switches']['RANDOM_FRIENDSHIP_AFFINITIES'] else 0
shyness_log_mean = config['parameters']['shyness_log_mean'] if config['switches']['RANDOM_SHYNESSES'] else 0
shyness_log_std = config['parameters']['shyness_log_std'] if config['switches']['RANDOM_SHYNESSES'] else 0


class Agent:
    """The brain of an agent"""
    def __init__(self, NoA, id, random_dict, name = "", x = None, openess = 1, mind = "ordinary",\
                 listening = True, decepting = True, disturbing = False, \
                 strategic = 0, egocentric = 0, flattering=0, aggressive=0, shameless=False,):
        self.NoA = NoA
        self.id = id           # remember identity
        if name == "":
            name = str(id)
        self.name = name
        if x is None:
            self.x = I0.draw() # random honesty. special agents get their honesty later
        else:
            self.x = x
        self.original_x = self.x   
        self.strategy = 'ordinary'
        self.use_acquaintance = True # if set to True in config, else acquaintance is not used
        self.I  = [I0 for i in NoA] # his beliefs 
        self.Nc = Counter(0)        # number of communications made
        self.Nt = Counter(0)        # number of trues made
        self.Nl = Counter(0)        # number of lies  made
        self.K  = [np.sqrt(np.pi) for t in np.arange(config['parameters']['Klength'])]       # surprises encountered 
        self.kappa  = median(self.K)/np.sqrt(np.pi)   # surprise scale sceptical, initalized with 1
        self.fr_affinity = 10**(random_dict['fr_affinities'].normal(fr_affinity_log_mean, fr_affinity_log_std)) # random friendship affinity in [0,1]
        self.original_fr_affinity = self.fr_affinity
        self.friendships = [I0 for i in NoA] # friendship distributions: mu = # positiv statements, la = # negativ statements, self-friendship initialized in setup.py
        self.original_friendships = self.friendships
        self.shyness = 10**(random_dict['shynesses'].normal(shyness_log_mean, shyness_log_std)) 
        self.original_shyness = self.shyness
        self.relationsc = [Counter(0) for i in NoA] # how often one talked with others
        self.relationsm = [Counter(1) for i in NoA] # how often one heard about others
        self.Jothers = [[I0 for i in NoA] for i in NoA] # what others said last   
        self.Iothers = [[I0 for i in NoA] for i in NoA] # what others think   
        self.Cothers = [[I0 for i in NoA] for i in NoA] # what others want self to believe
        if EBEH:
            self.prob_bt = Info(0,30) # propability of blushing in true message # 0,300
            self.prob_bl = Info(3,27) # propability of blushing in lie # 30,270
        self.openess  = openess
        self.mind     = mind
        self.listening = listening
        self.decepting = decepting
        self.disturbing = disturbing
        self.strategic = strategic
        self.egocentric= egocentric
        self.flattering= flattering
        self.aggressive= aggressive
        self.shameless = shameless
        
        #Acolor = ['red', 'black', 'cyan', 'yellow', 'blue', 'grey', 'green']*100
        self.color = Acolor[id]
        #Aalpha = [1,0.85,1.5,1.5,1,1,0.85]*100
        self.alpha = Aalpha[id] # transparency correction factor

        self.Buffer = [] # memory for recent statements


    def __str__(self):
        ret = "Agent " + str(self.name) + ":\n"
        ret +="honesty     \tx = " + str(self.x) 
        ret +=" ("+str(self.Nt)+", "+str(self.Nl)+")\n" 
        ret +="self image  \t" + str(self.I[self.id]) + "\n" 
        #ret +="x_obs = " + str(self.Nt.n)+ "/" + str(self.Nc.n) + " \t  = " 
        #ret +=str(self.Nt.n/(self.Nc.n+1e-50)) + "\n" 
        ret +="x_est = " + str(self.Nt.n+1)+ "/" + str(self.Nc.n+2) + " \t  = "
        ret +=str((self.Nt.n+1)/(self.Nc.n+2)) + "\n"
        ret +="kappa        = " +str(self.kappa) + "\n"
        ret +="openess      = " +str(self.openess) + "\n"
        ret +="strategic    = " +str(self.strategic) + "\n"
        ret +="egocentric   = " +str(self.egocentric) + "\n"
        ret +="mind = "  +str(self.mind) + "\n"
        ret +="decepting = " +str(self.decepting) + "\n"
        ret +="flattering = " +str(self.flattering) + "\n"
        ret +="aggressive = " +str(self.aggressive) + "\n"
        ret +="shameless = " +str(self.shameless) + "\n"
        # friendships
        ret += 'fr_affinity = ' +str(self.fr_affinity) + '\n'
        friendships = []
        for i in self.friendships:
            friendships.append(i.to_list())
        ret += 'friendships = ' +str(friendships) +'\n'
        ret += 'shyness = ' +str(self.shyness) + '\n'
        ret += 'relationsc = ' +str([relation.n for relation in self.relationsc]) +'\n'
        ret += 'relationsm = ' +str([relation.n for relation in self.relationsm]) +'\n'

         # theory of mind
        str_Jothers_1 = []
        for i in self.Jothers:
            str_Jothers_2 = []
            for j in i:
                str_Jothers_2.append(j.to_list())
            str_Jothers_1.append(str_Jothers_2)
        str_Iothers_1 = []
        for i in self.Iothers:
            str_Iothers_2 = []
            for j in i:
                str_Iothers_2.append(j.to_list())
            str_Iothers_1.append(str_Iothers_2)
        str_Cothers_1 = []
        for i in self.Cothers:
            str_Cothers_2 = []
            for j in i:
                str_Cothers_2.append(j.to_list())
            str_Cothers_1.append(str_Cothers_2)
        ret += 'J_others = ' +str(str_Jothers_1) +'\n'
        ret += 'I_others = ' +str(str_Iothers_1) +'\n'
        ret += 'C_others = ' +str(str_Cothers_1) +'\n'

        # EBEH
        if EBEH:
            ret += 'prob_bt = ' + str(self.prob_bt) + '\n'
            ret += 'prob_bl = ' + str(self.prob_bl) + '\n'
        return ret


    def set(self, attrname, value, Event):
        """changes the agent's property by replacing its old value by the new one (given by value).
           At the same time, the change is stored in the event-object which is later on saved
           to the output file."""

        if type(value) == Info: 
            if 'others' in attrname: # Jothers, Iothers, Cothers
                if attrname[0] == 'J':
                    self.Jothers[convert_name(attrname)[0]][convert_name(attrname)[1]] = value
                if attrname[0] == 'I':
                    self.Iothers[convert_name(attrname)[0]][convert_name(attrname)[1]] = value
                if attrname[0] == 'C':
                    self.Cothers[convert_name(attrname)[0]][convert_name(attrname)[1]] = value
            elif attrname.startswith('I_'): # I_0, I_1, ...
                self.I[convert_name(attrname)[0]] = value
            elif 'prob_' in attrname: # EBEH parameters prob_bt, prob_bl, ...
                if attrname == 'prob_bt':
                    self.prob_bt = value
                elif attrname == 'prob_bl':
                    self.prob_bl = value
                else:
                    raise ValueError(f'EBEH parameter {attrname} not found.')
            if Event!=None: Event.update[attrname] = value.to_list()
        elif 'friendship' in attrname:
            if attrname == 'friendship+':
                if CONTINUOUS_FRIENDSHIP:
                    self.friendships[value.id] = self.friendships[value.id] + Info(1,0)
                else:
                    self.friendships[value.id] = Info(1,0) # forget everything what happened before
            elif attrname == 'friendship-':
                if CONTINUOUS_FRIENDSHIP:
                    self.friendships[value.id] = self.friendships[value.id] + Info(0,1)
                else:
                    self.friendships[value.id] = Info(0,1)
            if Event!=None: Event.update[attrname+'_to_'+str(value.id)] = self.friendships[value.id].to_list()
        elif 'relations' in attrname: # count meetings and narrations # relationsc_12, relationsm_12
            if 'relationsc' in attrname: # talked with somebody
                if value == +1:
                    self.relationsc[convert_name(attrname)[0]].inc()
                if Event!=None: Event.update[attrname] = self.relationsc[convert_name(attrname)[0]].n
            elif 'relationsm' in attrname: # heard of somebody
                if value == +1:
                    self.relationsm[convert_name(attrname)[0]].inc()
                if Event!=None: Event.update[attrname] = self.relationsm[convert_name(attrname)[0]].n
        elif attrname == 'K':
            assert np.isfinite(value)
            self.K.append(value) # remember surprise
            if len(self.K) > config['parameters']['Klength']:
                self.K.pop(0)
            if Event!=None: Event.update['new_'+attrname] = value 
        elif attrname == 'kappa':
            self.kappa = value
            if Event!=None: Event.update[attrname] = value 
        elif attrname == 'Nt':
            self.Nt.inc()
            if Event!=None: Event.update[attrname] = int(value) 
        elif attrname == 'Nl':
            self.Nl.inc()
            if Event!=None: Event.update[attrname] = int(value)   
        else:
            raise ValueError('the attribute you want to set was not found!')

    def get_flattering_target(self, b_set):
        # returns recipient(s) with lowest reputation
        reputations = [self.I[b].mean for b in b_set] # reputation of conversation parnters, i.e. potential flattering objects
        indices_with_min_reputation = [index for index, item in enumerate(reputations) if item == min(reputations)] # the least honest agent in b_set
        return indices_with_min_reputation

    def draw_topic(self, A, b_set, now_egocentric, now_flattering, now_aggressive, flattering_target, one_to_one): # b is the recipient
        def f(x, const=20):
            return np.exp(-const*x) + 1e-100

        if self.strategy == 'MoLG' and one_to_one: # behave manipulative
            self.flattering = 1
            now_flattering = True
            self.strategic = -1
            self.egocentric = 0
            now_egocentric = False
        elif self.strategy == 'MoLG' and not one_to_one: # hehave dominant
            self.flattering = 0
            now_flattering = False
            self.strategic = 1
            self.egocentric = 1
            now_egocentric = True
        
        if now_egocentric:
            norm_weights = [0 for _ in range(len(A))]
            norm_weights[self.id] = 1
        elif now_flattering:
            assert flattering_target is not None
            norm_weights = [0 for _ in range(len(A))]
            norm_weights[flattering_target] = 1
        else:
            possible_topics = [i for i in range(len(A))]
            opinion_self = np.array([self.I[_].mean - 0.5 for _ in range(len(A))]) 
            opinion_b_set = np.array([np.nanmean([self.Iothers[b][_].mean - 0.5 for b in b_set]) for _ in range(len(A))])
            opinion_similarities_about_c = [opinion_self[c]*opinion_b_set[c] for c in possible_topics]
            sorted_opinion_similarities_about_c = sorted(opinion_similarities_about_c, reverse=True)
            homopholoy_weights = [f(sorted_opinion_similarities_about_c.index(sim)) for sim in opinion_similarities_about_c]
            # calculate weights
            weights = []
            for c in possible_topics:
                if c == self.id: # substitution for number of conversations with itself
                    group_relations = [rel.n for rel in self.relationsc if rel.n!=0]
                    r_c_aa = 0 if group_relations == [] else np.mean(group_relations) # approximation of how well the group members know each other
                    relation_strength = self.relationsm[c].n + Q*r_c_aa
                else:
                    relation_strength = self.relationsm[c].n + Q*self.relationsc[c].n
              

                friendship_strength = self.friendships[c].mean if CONTINUOUS_FRIENDSHIP else self.friendships[c].mu # 1 if friend, 0 if enemy

                fr_weight = friendship_strength**(self.fr_affinity)
                rel_weight = relation_strength**(self.shyness)
                # aggressive weight
                if CONTINUOUS_FRIENDSHIP:
                    aggressive_weight = (1-self.friendships[c].mean)**self.fr_affinity
                else: 
                    enemies = [_ for _ in range(len(A)) if self.friendships[_].mu==0 and self.friendships[_].la==1]
                    aggressive_weight = 1 if c in enemies else 0 


                weight = 1
                if now_aggressive: weight *= aggressive_weight# aggressive=0,1 # TODO: may need finetuning to live on same range then rel_weight
                elif FRIENDSHIP_AFFECTS_C: weight *= fr_weight # friendships should only affect the topic choice if not agressive
                if RELATION_AFFECTS_C: weight *= rel_weight # also agressive agents should be affected by relation strength
                if HOMOPHOLY_AFFECTS_C: weight *= homopholoy_weights[c]
                weights.append(weight)
            
            norm_weights = np.array(weights)/sum(weights)
        return norm_weights

    def draw_number_of_receivers(self, one_to_one):
        if one_to_one: # one-to-one communication
            norm_weights = [1 if _==1 else 0 for _ in range(1,len(self.NoA))]
        else: # determine number of recipients
            weights = [0 if _==1 else _**(-self.shyness) for _ in range(1,len(self.NoA))]
            norm_weights = [w/sum(weights) for w in weights]
        return norm_weights
    
    def draw_communication_partner(self, A, one_to_one):
        """The agent draws the communication partner

        Args:
            A (_type_): _description_
            random_dict (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.strategy == 'MoLG' and one_to_one: # behave manipulative
            self.flattering = 1
            self.strategic = -1
            self.egocentric = 0
        elif self.strategy == 'MoLG' and not one_to_one: # hehave dominant
            self.flattering = 0
            self.strategic = 1
            self.egocentric = 1

        possible_receivers = [i for i in range(len(A))] # ids of all the agents
        # calculate weights
        weights = []
        #print('self:     ', [(self.I[_].mu, self.I[_].la) for _ in range(len(A))])
        opinion_self = np.array([self.I[_].mean for _ in range(len(A))]) - 0.5
        similarities = [np.dot(opinion_self, np.array([self.Iothers[b][_].mean for _ in range(len(A))]) - 0.5)/len(A) for b in possible_receivers]
        def f(x, const=20):
            return np.exp(-const*x) + 1e-100
        sorted_similarities = sorted(similarities, reverse=True)
        homopholy_weight = [f(sorted_similarities.index(sim)) for sim in similarities]
        for b in possible_receivers:
            weight = 1
            if FRIENDSHIP_AFFECTS_B: 
                friendship_strength = self.friendships[b].mean if CONTINUOUS_FRIENDSHIP else self.friendships[b].mu # 1 if friend, 0 if enemy
                fr_weight = friendship_strength**(self.fr_affinity)
                weight *= fr_weight
            if RELATION_AFFECTS_B and self.use_acquaintance:
                relation_strength = self.relationsm[b].n + Q*self.relationsc[b].n
                rel_weight = relation_strength**(self.shyness)
                weight *= rel_weight
            if HOMOPHOLY_AFFECTS_B:
                #print('agent', b, ': ', [(self.Iothers[b][_].mu, self.Iothers[b][_].la) for _ in range(len(A))])   
                weight *= homopholy_weight[b]


            weights.append(weight)

        if self.strategic != 0:     # strategic agent 
            # distribute according to weights
            if self.strategic > 0: # strategic agent
                strategic_weights = [(self.I[k].mean)**self.strategic for k in range(len(A))]
            else: # anti-strategic agent
                strategic_weights = [(1-self.I[k].mean)**(-self.strategic) for k in range(len(A))]

            if self.use_acquaintance:
                weights = [weights[_]*strategic_weights[_] for _ in range(len(weights))]
            else:
                weights = strategic_weights # use only strategic weights, without acquaintance

        weights[self.id] = 0 # no self talking
        if sum(weights) == 0: weights = [weights[_]+tiny if _!=self.id else 0 for _ in range(len(weights))]
        norm_weights = np.array(weights)/sum(weights)
        #print('weights: ', norm_weights)
        return norm_weights
          
    def update(self, c, a, Iu, Ic, yc, Event, ToMupdate = True): 
        Itopic = self.I[c]           #initial belief
        if (np.isnan(Iu.mu) or np.isnan(Iu.la)): #check for NaNs
            Iu = Itopic #don't update
            print("WARNING: update resulted in NaN")
        else:
            sum = Iu.mu+Iu.la
            if sum > MaxCount:
                Iu.mu = Iu.mu* MaxCount/sum
                Iu.la = Iu.la* MaxCount/sum
            self.set('I_'+str(c), Iu, Event)    
        if ToMupdate:
            self.update_ToM(c, a, Ic, yc, Event)
            
    def update_ToM(self, c, a, Ic, yc, Event): # theory of mind update
        attrname = 'Iothers_'+str(a)+'_'+str(c)
        new_Iothers = yc*Ic+(1-yc)*self.Iothers[a][c]
        assert np.isfinite(new_Iothers.mu) and np.isfinite(new_Iothers.la)
        self.set(attrname, new_Iothers, Event)

        attrname = 'Cothers_'+str(a)+'_'+str(c)
        new_Cothers = (1-yc)*Ic+yc*self.Cothers[a][c]
        assert np.isfinite(new_Cothers.mu) and np.isfinite(new_Cothers.la)

        self.set(attrname, new_Cothers, Event)

    def update_EBEH_params(self, Iu, I_old, param, Event):
        if (np.isnan(Iu.mu) or np.isnan(Iu.la)): #check for NaNs
            Iu = I_old #don't update
            print("WARNING: EBEH parameter update resulted in NaN")
        else:
            sum = Iu.mu+Iu.la
            if sum > MaxCount:
                Iu.mu = Iu.mu* MaxCount/sum
                Iu.la = Iu.la* MaxCount/sum
        self.set(param, Iu, Event)

    def update_kappa(self, KLc, Event): # update kappa
        if KLc>0:                                
            self.set('K', KLc, Event) # remember surprise up to Klength entries
            self.set('kappa', median(self.K)/np.sqrt(np.pi), Event) #  adjust surprise scale
    
    def update_from_Buffer(self, A, filenames, EventNr): # later: update in a fancier way here
        """ at the moment awareness-updates are independent from Buffer and instantaneous """
        for m in self.Buffer:
            if self.listening: 
                EventNr.inc()
                self.listen(m, A, filenames, EventNr)  
            else: # watch only
                EventNr.inc()
                self.watch(m, filenames, EventNr)
            self.Buffer.remove(m)
        assert len(self.Buffer) == 0

    def watch(self, message, filenames, EventNr):
        a = message.a
        c = message.c
        Event = Update_event(int(self.id), int(message.time.n), int(EventNr.n))
        blush = message.blush 
        Ia = self.I[a]  # info on speaker
        xa = Ia.mean    # average assumed honesty of a
        # update relations: once per conversation
        self.set('relationsc_'+str(a), +1, Event) # b talked to a
        #self.set('relationsc_'+str(self.id), +1, Event) # b will observe talking himself in the answer
        self.set('relationsm_'+str(c), +1, Event) # b heard about c
        # estimate reliability of message and update
        if self.mind == 'naive':
            pass
        elif not EBEH and blush: # obvious lie
            self.update(a, a, Ia+Info(0,1), I0, 0, Event, ToMupdate=False) # remember lie
        
        else:
            if EBEH:
                Rb = self.prob_bl.mean/self.prob_bt.mean if blush else  (1-self.prob_bl.mean)/(1-self.prob_bt.mean)
            else:
                Rb = np.inf if blush else (1-ffl)
            yc = xa/(xa+Rb*(1-xa))
            assert np.isfinite(yc)
            #update belief on speaker
            Iau  = match(yc, Ia + Info(1,0), Ia + Info(0,1), Istart = Ia) # updated belief
            self.update(a, a, Iau, I0, yc, Event, False)            # update  belief

        if EBEH: # update evidence parameters          
            # prob_bt
            Itruth_bt = self.prob_bt + Info(1,0) if blush else self.prob_bt + Info(0,1)
            Ilie_bt = self.prob_bt
            Iu_bt = match(yc, Itruth_bt, Ilie_bt, self.prob_bt)
            self.update_EBEH_params(Iu_bt, self.prob_bt, 'prob_bt', Event) 
            # prob_bl
            Itruth_bl = self.prob_bl
            Ilie_bl = self.prob_bl + Info(1,0) if blush else self.prob_bl + Info(0,1)
            Iu_bl = match(yc, Itruth_bl, Ilie_bl, self.prob_bl)
            self.update_EBEH_params(Iu_bl, self.prob_bl, 'prob_bl', Event) 

        Event.update['yc'] = yc
        Event.save(filenames)
    
    def listen(self, message, A, filenames, EventNr):
        Event = Update_event(int(self.id), int(message.time.n), int(EventNr.n))
        b = self.id     # receiver 
        b_set = message.b_set # all receivers of this message
        c = message.c   # topic 
        Itopic = self.I[c]  # inital belief on topic c
        a = message.a   # speaker
        Ia = self.I[a]  # info on speaker
        xa = Ia.mean    # average assumed honesty of a
        Ic = message.J  # statement of a on c
        blush = message.blush # whether a blushed
        confession = (a == c and Ic.mean < xa) and not blush
        attrname = 'Jothers_'+str(a)+'_'+str(c)
        self.set(attrname, Ic, Event) # remember the statement made
        DeltaI = DeltaInfo(Ic, self.Iothers[a][c]) # new information
        Iprime = Itopic + DeltaI*self.openess           # I of naive update
        # update relations: once per conversation
        self.set('relationsc_'+str(a), +1, Event) # b talked to a
        #self.set('relationsc_'+str(self.id), +1, Event) # b will observe himself talking in the answer
        self.set('relationsm_'+str(c), +1, Event) # b heard about c

        # update knowledge
        if self.mind == 'naive':
            self.update(c, a, Iprime, Ic, 1, Event, ToMupdate=True)
            return
        
        elif self.mind == 'uncritical':
            if EBEH:
                raise NotImplementedError('Experience based evidence handling is not implemented for uncritical minds.')
            else:
                if confession:
                    yc = 1
                    Iprime = Ic # adapt, what was confessed
                else:
                    yc = int(not blush)* xa/(xa+(1-ffl)*(1-xa))
        
        elif self.mind in ['ordinary', 'smart']:
            KLc = KL(Ic, Itopic)      # KL of adapting position
            Sc = KLc/self.kappa  # rescaled surprise
            if EBEH:
                Rb = self.prob_bl.mean/self.prob_bt.mean if blush else  (1-self.prob_bl.mean)/(1-self.prob_bt.mean)
            else:
                Rb = np.inf if blush else (1-ffl)
            
            if self.mind == 'smart':
                I_a  = self.Iothers[a][c]  # k's belief on j's believe on i
                I_b  = self.Cothers[a][c]  # k's belief on what j wants k to believe on i
                KLca = KL(Ic, I_a)     # distance of message and apparent speaker's belief
                KLcb = KL(Ic, I_b)     # distance of message to apparent speaker's intention
                Sch = KLca/self.kappa  # surpriese of honest communication  
                Scl = KLcb/self.kappa  # surpriese of lie

            if self.mind == 'ordinary':
                if Rb == np.inf: # safety for some numpy/python versions
                    yc = 0
                else:
                    yc = xa/(xa+ Sc_dist_ratio(Sc)*Rb*(1-xa)) 
            elif self.mind == 'smart':
                Rc  = exp(Sch-Scl)
                if not np.isfinite(Rc) or np.isnan(Rc) or not np.isfinite(Rb):
                    yc = 0
                else:
                    yc = xa/(xa+ Rc*Sc_dist_ratio(Sc)*Rb*(1-xa))

            if confession:
                yc = 1
                Iprime = Ic # adapt, what was confessed

            if a != c:
                #update belief on topic
                Iu  = match(yc, Iprime, Itopic, Itopic)         # updated belief
                self.update(c, a, Iu, Ic, yc, Event, True) # update  belief
                #update belief on speaker
                Iau  = match(yc, Ia + Info(1,0), Ia + Info(0,1), Ia) # updated belief
                self.update(a, a, Iau, Ic, yc, Event, False)            # update  belief
            else:
                Iu = match(yc, Iprime + Info(1,0), Itopic + Info(0,1), Itopic)
                self.update(c, a, Iu, Ic, yc, Event, True) 
            self.update_kappa(KLc, Event) # update KL stack and kappa

        if EBEH: # update evidence parameters
            assert self.mind in ['ordinary', 'smart']            
            # prob_bt
            Itruth_bt = self.prob_bt + Info(1,0) if blush else self.prob_bt + Info(0,1)
            Ilie_bt = self.prob_bt
            Iu_bt = match(yc, Itruth_bt, Ilie_bt, self.prob_bt)
            self.update_EBEH_params(Iu_bt, self.prob_bt, 'prob_bt', Event) 
            # prob_bl
            Itruth_bl = self.prob_bl
            Ilie_bl = self.prob_bl + Info(1,0) if blush else self.prob_bl + Info(0,1)
            Iu_bl = match(yc, Itruth_bl, Ilie_bl, self.prob_bl)
            self.update_EBEH_params(Iu_bl, self.prob_bl, 'prob_bl', Event) 

        # maintain friendshipsS
        if c==b:         # reciever is the topic
            speaker = A[a] 
            x_of_speaker = Ic.mean # opinion expressed
            x_of_others = []  # opinions of other agents
            for aa in np.arange(len(A)):
                if aa!=b and aa!=a:
                    x_of_others.append(self.Jothers[aa][b].mean)
            x_median =  median(x_of_others)
            if x_of_speaker > x_median:       # talked positively
                #print('positive statement of', speaker.id, ' to ', b)
                # add positive statement in b's friendship to a
                for _ in range(len(b_set)): # stronger reward for 1toX conversations
                    self.set('friendship+', speaker, Event)
            elif x_of_speaker < x_median:     # talked in an insulting way ...
                # add negative statement in b's friendship to a
                #print('negative statement of', speaker.id, ' to ', b)
                for _ in range(len(b_set)): # stronger punishment in 1toX conversations
                    self.set('friendship-', speaker, Event)

        Event.update['yc'] = yc
        Event.save(filenames)
    
    def awarness(self, message, A, filenames, EventNr):
        """self awareness"""
        EventNr.inc()
        Event = Update_event(int(self.id), int(message.time.n), int(EventNr.n))
        Event.event_type = 'self_update'
        if self.id == message.a: # remember your own statements
            attrname = 'Jothers_'+str(message.a)+'_'+str(message.c)
            self.set(attrname, message.J, Event) # remember the statement made
        # update self-esteem
        honest = message.honest
        a  = self.id
        mu = self.I[a].mu
        la = self.I[a].la
        if honest:              # count a truth
            Iu = Info(mu+1,la)
            self.set('Nt', self.Nt.n+1, Event)
        else:                   # count a lie
            Iu = Info(mu,la+1)
            self.set('Nl', self.Nl.n+1, Event)
        self.update(a, a, Iu, I0, 0, Event, False)
        Event.save(filenames)

        if EBEH and EBEH_self_awareness: 
            # update EBEH parameters based on own statement
            blush = message.blush
            
            if message.honest:
                if blush: # should not happen in the current setup
                    self.update_EBEH_params(self.prob_bt + Info(1,0), self.prob_bt, 'prob_bt', Event) 
                else:
                    self.update_EBEH_params(self.prob_bt + Info(0,1), self.prob_bt, 'prob_bt', Event) 
            else:
                if blush: 
                    self.update_EBEH_params(self.prob_bl + Info(1,0), self.prob_bl, 'prob_bl', Event) 
                else:
                    self.update_EBEH_params(self.prob_bl + Info(0,1), self.prob_bl, 'prob_bl', Event)

        if ACTIVE_SELF_FRIENDSHIP:
            # maintain self-friendship
            assert self.id == message.a
            c = message.c
            Ic = message.J
            if c==a: # somebody's talking about himself
                speaker = A[a]
                x_of_speaker = Ic.mean # opinion expressed
                x_on_a = []  # opinions of other agents
                for aa in np.arange(len(A)):
                    if aa!=a:
                        x_on_a.append(self.Jothers[aa][a].mean) # what others said last (to a) about a
                x_median =  median(x_on_a)
                if x_of_speaker > x_median:       # talked positively
                    #print('SELF: positive statement of', speaker.id, ' about ', a)
                    # add positive statement in b's friendship to a
                    self.set('friendship+', speaker, Event)
                elif x_of_speaker < x_median:     # talked in an insulting way ...
                    # add negative statement in b's friendship to a
                    #print('SELF: negative statement of', speaker.id, ' about ', a)
                    self.set('friendship-', speaker, Event)
    
    def talk(self, c, b_set, b_weights, A, now_honest, now_blush, now_KL_target, now_aggressive, one_to_one, filenames, EventTime, EventNr):
        EventNr.inc()
        self.Nc.inc() 
        honest = now_honest
        blush = now_blush

        if self.strategy == 'MoLG' and one_to_one: # behave manipulative
            self.flattering = 1
            now_flattering = True
            self.strategic = -1
            self.egocentric = 0
            now_egocentric = False
        elif self.strategy == 'MoLG' and not one_to_one: # hehave dominant
            self.flattering = 0
            now_flattering = False
            self.strategic = 1
            self.egocentric = 1
            now_egocentric = True

        if self.flattering and (c in b_set): # flatter
            honest = False

        # decide what to say   
        if honest:
            Ic = self.I[c] # true belief on c
        elif not honest and self.decepting: # trying to manipulate at all, not just white lies
            # start with weighted assumed opinions of all recipients
            weights = [b_weights[_] for _ in b_set] # same weights as for choosing b <- "importance of each recipient"
            norm_weights = [_/sum(weights) for _ in weights]
            J = make_average_opinion([self.Iothers[b][c] for b in b_set], weights = norm_weights)
            mu = J.mu
            la = J.la
            KL_target =  now_KL_target

            if self.disturbing:
                KL_target = 2*KL_target
            if self.id == c: # maximal positive 
                if now_aggressive:
                    Ic = J # about oneself what the recipient believes
                else:
                    func = lambda x: KL(J + Info(mu+x*x,la),J)-KL_target
                    x = root(func, 1).x[0]
                    delta_mu_max = x*x 
                    delta_mu = delta_mu_max
                    new_mu = mu + delta_mu
                    Ic = Info(new_mu,la)
            elif self.flattering and c in b_set: # in backward communications, c might not be in b_set
                func = lambda x: KL(J + Info(mu+x*x,la),J)-KL_target
                x = root(func, 1).x[0]
                delta_mu_max = x*x 
                if SCALED_FLATTERING:
                    delta_mu = (1-self.I[c].mean)*delta_mu_max # flatter according to honesty: the more c lies the more effect has flattering
                else:
                    delta_mu = delta_mu_max
                new_mu = mu + delta_mu
                Ic = Info(new_mu,la)
            elif self.friendships[c].mean > 0.5: # regarded as a friend, adapted lie size
                if now_aggressive: 
                    Ic = J # about friends what the recipient believes
                else:
                    func = lambda x: KL(J + Info(mu+x*x,la),J)-KL_target
                    x = root(func, 1).x[0]
                    delta_mu_max = x*x
                    if CONTINUOUS_FRIENDSHIP:
                        delta_mu = 2*(self.friendships[c].mean-0.5)*delta_mu_max
                    else:
                        delta_mu = delta_mu_max
                    new_mu = mu + delta_mu
                    if new_mu<mu or new_mu<0: 
                        print("WARNING:",mu, new_mu)
                    Ic = Info(new_mu,la)
            elif self.friendships[c].mean < 0.5: # regarded as an enemy
                func = lambda x: KL(J + Info(mu,la+x*x),J)-KL_target
                x = root(func, 1).x[0]
                delta_la_max = x*x
                if CONTINUOUS_FRIENDSHIP:
                    delta_la = -2*(self.friendships[c].mean-0.5)*delta_la_max
                else:
                    delta_la = delta_la_max
                new_la = la + delta_la
                if new_la<la  or new_la<0 : 
                    print("WARNING:",la, new_la)
                Ic = Info(mu,new_la)
            else: # tell b its own belief
                Ic = J 
            if ultraverbose: 
                print(self.id, "Lies to", b_set, 'about', c, ' at ', EventTime, ":", J.mu, J.la,"->", Ic.mu, Ic.la,\
                      "\n       KL =", KL(J + DeltaInfo(Ic, self.Iothers[self.id][c])\
                               *self.openess,J))
        else: # make uninformative statement
            Ic = I0 
        m = Message(c, self.id, [int(_) for _ in b_set], Ic, bool(honest), bool(blush), EventTime, EventNr) # c, a, b, J, honest, blush
        Event = Comm_event()
        Event.comm = m.message_dict()
        if verbose:
            print(Event)
        Event.save(filenames)
        
        self.awarness(m, A, filenames, EventNr)
        return m

    def Agent_dict(self): # transforms properties in a dict containing json serializable objects
        properties = {
        'id':int(self.id), 'name':self.name, 'x': float(self.x), \
        'I':[self.I[i].to_list() for i in np.arange(len(self.I))], \
        'Nc':int(self.Nc.n), 'Nt':int(self.Nt.n), 'Nl':int(self.Nl.n), \
        'K':self.K, 'kappa':float(self.kappa),\
        'fr_affinity': float(self.fr_affinity), 'shyness':float(self.shyness), \
        'friendships': [self.friendships[i].to_list() for i in np.arange(len(self.friendships))], \
        'relationsc':[self.relationsc[i].n for i in np.arange(len(self.relationsc))],\
        'relationsm':[self.relationsm[i].n for i in np.arange(len(self.relationsm))],\
        'Jothers':[[self.Jothers[i][j].to_list() for j in np.arange(len(self.Jothers[0]))] for i in np.arange(len(self.Jothers))], \
        'Iothers':[[self.Iothers[i][j].to_list() for j in np.arange(len(self.Iothers[0]))] for i in np.arange(len(self.Iothers))], \
        'Cothers':[[self.Cothers[i][j].to_list() for j in np.arange(len(self.Cothers[0]))] for i in np.arange(len(self.Cothers))], \
        'openess':float(self.openess), 'mind':self.mind, 'listening': self.listening, 'decepting': self.decepting, 'strategic':float(self.strategic), \
        'egocentric':float(self.egocentric), 'flattering':float(self.flattering), 'aggressive':float(self.aggressive), \
        'shameless':self.shameless, 'disturbing':self.disturbing, 'color': self.color, 'alpha':self.alpha, 'Buffer':self.Buffer
        }
        if EBEH:
            properties['prob_bt'] = self.prob_bt.to_list()
            properties['prob_bl'] = self.prob_bl.to_list()
        return properties