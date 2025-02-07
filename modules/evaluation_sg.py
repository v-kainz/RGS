import os
import json
import numpy as np
import pandas as pd
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
nice_fonts = {'text.usetex': False, "pgf.texsystem": "pdflatex"}
plt.rcParams.update(nice_fonts)
from matplotlib.patches import Patch, Ellipse
from matplotlib.lines import Line2D
import yaml
import gc


from modules.globals import names
from modules.informationtheory import Info
from modules.helperfunc import convert_name, get_folder, get_name, clean, overview, rotate, make_latex_str, get_numbers
from modules.helperfunc_dependent import fill
#from modules.density_estimator import estimate_density
with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
verbose = config['verbosity']['verbose']
Q = config['parameters']['Q']
compatibility = config['switches']['compatibility']
Acolor = config['constants']['Acolor']

import time
startTime = time.time()

class DataCollector:
    """collects overarching data for multiple strategies. Every data asset it specified together with its source, 
    such that if the data asset is missing, the right plot can be suggested to generate it.
    data is of the form
    {'strategy': , 'strategy_data': ...}"""
    def __init__(self):
        self.BADEnotBADE_means = {'data': {}, 'source': 'plot_BADEnotBADE_hist'}
        self.likelihoodRatio = {'data': {}, 'source': 'plot_likelihood_ratio_hist'}

    def _get_strategy_from_name(self, name):
        potential_names = [n for n in names.keys() if n in name]
        assert len(potential_names) == 1
        return potential_names[0]

    def collect_from_strategy(self, name, attribute, data):
        strategy = self._get_strategy_from_name(name)
        if attribute == 'BADEnotBADE_means':
            self.BADEnotBADE_means['data'][strategy] = data
        elif attribute == 'likelihoodRatio':
            self.likelihoodRatio['data'][strategy] = data
        else:
            raise ValueError(f'could not find the correct attribute with name {attribute}.')
        
        

class Rec:
    def __init__(self, NA, NR, mode, title, RSeed, colors, alphas, x_est, fr_affinities, shynesses, CONTINUOUS_FRIENDSHIP):
        self.NA = NA
        self.NR = NR
        self.mode = mode
        self.title = title
        self.RSeed = RSeed
        self.CONTINUOUS_FRIENDSHIP = CONTINUOUS_FRIENDSHIP
        # Idata[who thinks][about whom][time]
        self.Idata = [[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)]
        self.Idata_rms = [[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)]
        self.awareness = [] # final awareness (mean) for all agents
        self.awareness_std = []
        self.reputation = [] # final reputation (mean) for all agents
        self.reputation_std = []
        self.Nt = [0 for i in range(NA)] # keep only the final values here
        self.Nl = [0 for i in range(NA)] # keep only the final values here
        self.colors = colors
        self.alphas = alphas
        self.x_est = x_est
        self.fr_affinities = fr_affinities
        self.shynesses = shynesses
        self.comm = [] 
        #friendship[who considers][whom a friend][time] # to what degree (1: friend, 0: enemy)
        self.friendships = [[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)]
        self.friendships_rms = [[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)]
        #relationsc[who has talked][with whom][time] how often
        self.relationsc = [[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)]
        #relationsm[who has heard][about whom][time] how often
        self.relationsm = [[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)]
        #self.Nfr = [] # how many agents consider i a friend
        #self.Nen = [] # how many agents consider i an enemy
        #self.Timefr = [] # [who's friendship][to whom]  fraction of time this friendship lasts
        #self.Timeen = [] # [who's enemyship][to whom] fraction of time this friendship lasts
        # theory of mind
        # Iothers[who's mind stores][that who thinks][about whom][time]
        self.Iothers = [[[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
        self.Iothers_rms = [[[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
        # Cothers[]
        self.Cothers = [[[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
        self.Cothers_rms = [[[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
        # Jothers[]
        self.Jothers = [[[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
        self.Jothers_rms = [[[[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
        # kappa
        self.kappa = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's kappa][time]
        # K
        self.K = [[np.nan for i in range(NA*NR*2+1)] for i in range(NA)]
        # y_c
        self.yc = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who assigned this yc][time when message was sent]
        # honesty 0,1
        self.honesty = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who was honest/dishonest][time of conversation]
        # EBEH parameters
        self.prob_bt = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's mean parameter][time]
        self.prob_bt_rms = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's std parameter][time]
        self.prob_bl = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's mean parameter][time]
        self.prob_bl_rms = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's std parameter][time]
        self.prob_ct = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's mean parameter][time]
        self.prob_ct_rms = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's std parameter][time]
        self.prob_cl = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's mean parameter][time]
        self.prob_cl_rms = [[np.nan for i in range(NA*NR*2+1)]for i in range(NA)] # [who's std parameter][time]


    def complete(self, cont_fr):
        for i in range(self.NA):
            self.kappa[i] = fill(self.kappa[i], init = 'kappa')
            self.prob_bt[i] = fill(self.prob_bt[i])
            self.prob_bl[i] = fill(self.prob_bl[i])
            self.prob_ct[i] = fill(self.prob_ct[i])
            self.prob_cl[i] = fill(self.prob_cl[i])
            self.prob_bt_rms[i] = fill(self.prob_bt_rms[i])
            self.prob_bl_rms[i] = fill(self.prob_bl_rms[i])
            self.prob_ct_rms[i] = fill(self.prob_ct_rms[i])
            self.prob_cl_rms[i] = fill(self.prob_cl_rms[i])
            
            # no completion for yc
            self.K[i] = fill(self.K[i], init = 'K')
            for j in range(self.NA):
                self.Idata[i][j] = fill(self.Idata[i][j], init = 'info')
                self.Idata_rms[i][j] = fill(self.Idata_rms[i][j], init = 'rms')
                if cont_fr:
                    self.friendships[i][j] = fill(self.friendships[i][j], init = 'info')
                    self.friendships_rms[i][j] = fill(self.friendships_rms[i][j], init = 'rms')
                else:
                    if i == j:
                        self.friendships[i][j] = fill(self.friendships[i][j], init = 1)
                    else:
                        self.friendships[i][j] = fill(self.friendships[i][j], init = None)
                self.relationsc[i][j] = fill(self.relationsc[i][j], init = 0)
                self.relationsm[i][j] = fill(self.relationsm[i][j], init = 1)
                for k in range(self.NA):
                    self.Iothers[k][i][j] = fill(self.Iothers[k][i][j], init = 'info')
                    self.Iothers_rms[k][i][j] = fill(self.Iothers_rms[k][i][j], init = 'rms')
                    self.Cothers[k][i][j] = fill(self.Cothers[k][i][j], init = 'info')
                    self.Cothers_rms[k][i][j] = fill(self.Cothers_rms[k][i][j], init = 'rms')
                    self.Jothers[k][i][j] = fill(self.Jothers[k][i][j], init = 'info')
                    self.Jothers_rms[k][i][j] = fill(self.Jothers_rms[k][i][j], init = 'rms')
                

    def calculate_awa_rep(self): # call after rec.complete
        awareness = [np.mean(self.Idata[i][i]) for i in range(self.NA)]
        awareness_std = [np.std(self.Idata[i][i]) for i in range(self.NA)]
        reputations = []
        for i in range(self.NA):
            array = np.array([])
            for j in range(self.NA):
                if j!=i:
                    array = np.concatenate((array, self.Idata[j][i]))
            reputations.append(array)
        reputation = [np.mean(reputations[i]) for i in range(self.NA)]
        reputation_std = [np.std(reputations[i]) for i in range(self.NA)]
        self.awareness = awareness
        self.awareness_std = awareness_std
        self.reputation = reputation
        self.reputation_std = reputation_std
    
    def calculate_awa_rep_half(self): 
        """Used only if compatibility is True"""
        reputations = []
        for i in range(self.NA):
            array = np.array([])
            for j in range(self.NA):
                if j!=i:
                    array = np.concatenate((array, self.Idata[j][i][::2][1:]))
            reputations.append(array)   
        # use every second entry only and leave out starting value   
        awareness = [np.mean(self.Idata[i][i][::2][1:]) for i in range(self.NA)]
        awareness_std = [np.std(self.Idata[i][i][::2][1:]) for i in range(self.NA)]
        reputation = [np.mean(reputations[i]) for i in range(self.NA)]
        reputation_std = [np.std(reputations[i]) for i in range(self.NA)]
        self.awareness = awareness
        self.awareness_std = awareness_std
        self.reputation = reputation
        self.reputation_std = reputation_std
    
    def plot_dynamics(self, name, dc, ignore = {}, prop = 0):

        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([0,self.NA*self.NR*2])
        if prop == +1: # shorter xrange
            ax.set_xlim([0,(self.NA-1)*self.NR])
        if prop == -1: # longer xrange
            ax.set_xlim([0,(self.NA-1)*self.NR*3+1])
            timeline = np.arange(0,(self.NA-1)*self.NR*3+1)
        #if listening:
        for i in np.arange(self.NA):
            for j in np.arange(self.NA):
                if (i,j) not in ignore and i!=j:
                    ax.step(timeline, self.Idata[i][j], where='post', color = Acolor[j], lw=3) #@ timeline
                    ax.step(timeline, self.Idata[i][j], where='post', color = Acolor[i], dashes=[1,4], lw=3)
                    #ax.fill_between(timeline, \
                    #                np.array(self.Idata[i][j])-np.array(self.Idata_rms[i][j]), \
                    #                np.array(self.Idata[i][j])+np.array(self.Idata_rms[i][j]), \
                    #                step = 'post', color = Acolor[j], alpha = 0.05*Aalpha[j])
        for i in np.arange(self.NA):
            if (i,i) not in ignore:
                pass
                #ax.step(timeline, self.Idata[i][i], where='post', color = Acolor[i], linewidth = 3)
                #ax.fill_between(timeline, np.array(self.Idata[i][i])-np.array(self.Idata_rms[i][i]), \
                #                np.array(self.Idata[i][i])+np.array(self.Idata_rms[i][i]), \
                #                step = 'post', color = Acolor[i], alpha = 0.15*Aalpha[i])
            if (i) not in ignore:
                #ax.set_xlim([0,self.NA*self.NR*2*(1.01+(self.NA+self.NA)/100)])
                if self.Nt[i]+self.Nl[i]>0:
                    x_est = (self.Nt[i])/(self.Nt[i]+self.Nl[i])
                    ax.plot([0,self.NA*self.NR*2], [x_est, x_est],  color = Acolor[i],\
                            linewidth = 1, linestyle = 'solid')
                    #ax.errorbar(self.NA*self.NR*2*(1.01+i/100), self.reputation[i], self.reputation_std[i],\
                    #            marker='s', markersize=9, color = Acolor[i], capsize = 5, capthick = 1.5)
                    #ax.errorbar(self.NA*self.NR*2*(1.01+(self.NA+i)/100), self.awareness[i], self.awareness_std[i],\
                    #            marker='o', markersize=9, color = Acolor[i], capsize = 5, capthick = 1.5) 
                else:
                    print('WARNING: x_est is not defined, true x not implemented yet')
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        Ticks = [_ for _ in timeline if _%1000==0]
        ax.set_xticks(Ticks)
        #plt.xlabel(make_latex_str('time $t$'), fontsize = 20) #20
        #plt.ylabel(r'$\mathrm{perceived\ honesty\ }\overline{x}_{ab}$', fontsize = 20) 
        plt.xlabel(make_latex_str('time'), fontsize = 20) #20
        plt.ylabel('perceived honesties', fontsize = 20) 
        plt.title(make_latex_str(self.title), fontsize = 24)
        plt.xticks(fontsize=18)#12
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'.png')
        plt.close() 
        if verbose: print("Plot in", name + '.png')

    def plot_dynamics_simple(self, name, dc, ignore = {}, prop = 0):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([0,self.NA*self.NR*2])
        #if listening:
        for i in np.arange(self.NA):
            if (i,i) not in ignore:
                # self esteem
                #ax.step(timeline, self.Idata[i][i], where='post', color = Acolor[i], linestyle='--', lw=3)
                # reputation
                av_rep = np.array([0]*len(self.Idata[i][i]))
                for j in np.arange(self.NA):
                    if i != j:
                        av_rep = av_rep + self.Idata[j][i]
                av_rep = av_rep/(self.NA-1)
                ax.step(timeline, av_rep, where='post', color = Acolor[i], lw=3)
                # intrinsic honesties
                if self.Nt[i]+self.Nl[i]>0:
                    x_est = (self.Nt[i])/(self.Nt[i]+self.Nl[i])
                    ax.plot([0,self.NA*self.NR*2], [x_est, x_est],  color = Acolor[i],\
                            linewidth = 1, linestyle = 'solid')
            if False:
                ax.set_xlim([0,self.NA*self.NR*2*(1.01+(self.NA+self.NA)/100)])
                if self.Nt[i]+self.Nl[i]>0:
                    x_est = (self.Nt[i])/(self.Nt[i]+self.Nl[i])
                    #ax.plot([0,self.NA*self.NR*2], [x_est, x_est],  color = Acolor[i],\
                    #        linewidth = 3, linestyle = '--')
                    ax.errorbar(self.NA*self.NR*2*(1.03+i/100), x_est, 0,\
                                marker='s', markersize=9, color = Acolor[i], capsize = 0, capthick = 0)
                else:
                    print('WARNING: x_est is not defined, true x not implemented yet')
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        Ticks = [_ for _ in timeline if _%2000==0]
        ax.set_xticks(Ticks)
        plt.xlabel(make_latex_str('time'), fontsize = 20) #20
        #plt.xlabel(make_latex_str('time $t$'), fontsize = 20) #20
        plt.ylabel(make_latex_str('reputation'), fontsize = 20) 
        plt.title(make_latex_str(self.title), fontsize = 24)
        plt.xticks(fontsize=18)#12
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_s.png')
        plt.close() 
        if verbose: print("Plot in", name + '_s.png')

    def plot_communications(self, name, dc):
        Acolor = self.colors
        timeline = np.arange(0, self.NA*self.NR*2+1)
        fig, ax = plt.subplots(figsize=(50, 6))
        # x_honest[who speaks][about whom] is a tuple of dicts
        x_honest = [[() for i in range(self.NA)] for j in range(self.NA)] 
        x_lie = [[() for i in range(self.NA)] for j in range(self.NA)]
        for C in self.comm:
            if C['honest']:
                marker = 'o'
                x_honest[C['a']][C['c']] = x_honest[C['a']][C['c']] + (C,)
            else:
                marker = 'v'
                x_lie[C['a']][C['c']] = x_lie[C['a']][C['c']] + (C,)
            ax.scatter(C['time'], C['J_mean'], color = Acolor[C['a']], s=100, marker = marker)
            if len(C['b_set']) > 1: 
                raise NotImplementedError('communication plots with more than 1 recipient have not been implemenetd yet.')
            else:
                b = int(C['b_set'][0])
            ax.scatter(C['time'], C['J_mean'], color = Acolor[b], s=60, marker = marker)
            ax.scatter(C['time'], C['J_mean'], color = Acolor[C['c']], s=25, marker = marker)
        for i in range(self.NA):
            for j in range(self.NA):
                # i talks about j
                if x_honest[i][j] != ():                
                    x_mean = np.mean([com['J_mean'] for com in x_honest[i][j]])
                    x_std  =  np.std([com['J_mean'] for com in x_honest[i][j]])
                    t = self.NA*self.NR*2*(1.02+4*i/100)
                    marker = 'o'
                    ax.errorbar(t, x_mean, x_std, marker=marker, markersize=6, color= Acolor[j],\
                                ecolor=Acolor[i], capsize = 5, capthick = 1.5)
                    ax.scatter(t, x_mean, color = Acolor[i], s=80, marker = marker)      
                    ax.scatter(t, x_mean, color = Acolor[j], s=40, marker = marker)
                if x_lie[i][j] != ():
                    x_mean = np.mean([com['J_mean'] for com in x_lie[i][j]])
                    x_std  =  np.std([com['J_mean'] for com in x_lie[i][j]])
                    t  = self.NA*self.NR*2*(1.04+4*i/100)
                    marker = 'v'
                    ax.errorbar(t, x_mean, x_std, marker=marker, markersize=6, color=Acolor[j],\
                                ecolor=Acolor[i], capsize = 5, capthick = 1.5)
                    ax.scatter(t, x_mean, color = Acolor[i], s=80, marker = marker)      
                    ax.scatter(t, x_mean, color = Acolor[j], s=40, marker = marker)
        
        ax.set_ylim([0,1])
        ax.set_xlim([0,self.NA*self.NR*2*(1.02+4*self.NA/100)])
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        ax.set_xticks(Ticks)
        plt.xlabel(make_latex_str('time $t$'), fontsize = 20)
        plt.ylabel(r'$\mathrm{communicated\ honesty\ }\overline{x}_{J}$', fontsize = 20) 
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+"_com.png")
        if verbose: print("Plot in", name+"_com.png")
        for i in range(self.NA):
            for j in range(self.NA):
                ax.step(timeline, self.Idata[i][j], where='post', color = Acolor[j])
                ax.step(timeline, self.Idata[i][j], where='post', color = Acolor[i], dashes=[1,4])
        plt.savefig(name+"_comL.png")
        plt.close() 
        if verbose: print("Plot in", name+"_comL.png")

    def plot_friendship(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([0,1])
        ax.set_xlim([0,self.NA*self.NR*2])
        for i in np.arange(self.NA):
            for j in np.arange(self.NA):
                if i!=j:
                    ax.step(timeline, self.friendships[i][j], where='post', color = Acolor[j]) #@ timeline
                    ax.step(timeline, self.friendships[i][j], where='post', color = Acolor[i], linestyle = ':')
                    ax.fill_between(timeline, \
                                    np.array(self.friendships[i][j])-np.array(self.friendships_rms[i][j]), \
                                    np.array(self.friendships[i][j])+np.array(self.friendships_rms[i][j]), \
                                    step = 'post', color = Acolor[j], alpha = 0.05*Aalpha[j])
        #for i in np.arange(self.NA):
        #    ax.step(timeline, self.friendships[i][i], where='post', color = Acolor[i], linewidth = 3)
        #    ax.fill_between(timeline, np.array(self.friendships[i][i])-np.array(self.friendships_rms[i][i]), \
        #                    np.array(self.friendships[i][i])+np.array(self.friendships_rms[i][i]), \
        #                    step = 'post', color = Acolor[i], alpha = 0.15*Aalpha[i])
            #if (i) not in ignore:
                #ax.set_xlim([0,self.NA*self.NR*2*(1.01+(self.NA+self.NA)/100)])
                #ax.errorbar(self.NA*self.NR*2*(1.01+i/100), self.reputation[i], self.reputation_std[i],\
                #            marker='s', color = Acolor[i], capsize = 5)
                #ax.errorbar(self.NA*self.NR*2*(1.01+(self.NA+i)/100), self.awareness[i], self.awareness_std[i],\
                #            marker='o', color = Acolor[i], capsize = 5) 
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        ax.set_xticks(Ticks)
        plt.xlabel('time', fontsize=20)
        plt.ylabel('friendship', fontsize=20)
        plt.title(self.title, fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_fr'+'.png')
        plt.close() 
        if verbose: print("Plot in", name+'_fr.png')
    
    def plot_dyn_fr(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([0,1])
        ax.set_xlim([0,self.NA*self.NR*2])
        delta = 0.01
        d = 0.05 # distance between friend- and enemy-area
        D = (1-d)/(2*self.NA) # width of area for one agent
        s = (1-d)/(2*self.NA*self.NA) # distance between single lines
        for i in np.arange(self.NA):
            for j in np.arange(self.NA):
                if i!=j:
                    ax.step(timeline, self.Idata[i][j], where='post', color = Acolor[j])
                    ax.step(timeline, self.Idata[i][j], where='post', color = Acolor[i], dashes=[1,4])
                    # adapt friendship data to honesty values
                    fr_dyn = [[[np.nan for i in range(self.NA*self.NR*2+1)]for i in range(self.NA)] for i in range(self.NA)]
                    for t in timeline:
                        if self.friendships[i][j][t] > 0.5: # upper half in friendship plot
                            fr_dyn[i][j][t] = self.Idata[i][j][t]
                    ax.fill_between(timeline, \
                                    np.array(fr_dyn[i][j])-np.array([delta for t in timeline]), \
                                    np.array(fr_dyn[i][j])+np.array([delta for t in timeline]), \
                                    step = 'post', color = Acolor[j], alpha = 0.5*Aalpha[j])
        legend = [Line2D([0], [0], color='red', lw=1, label="'s reputation in the eyes of"),
                Patch(facecolor='red', edgecolor=None, label='is considered a friend', alpha = 0.15*1.5),
                Line2D([0], [0], color='black', lw=1, label="", dashes=[1,4])]
        ax.legend(handles=legend, loc='best', frameon=True, ncol=2)
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        ax.set_xticks(Ticks)
        plt.xlabel(make_latex_str('time $t$'), fontsize = 20)
        plt.ylabel(make_latex_str('perceived honesty'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_dyn_fr.png')
        plt.close() 
        if verbose: print("Plot in", name+'_dyn_fr.png')
    
    def plot_ToMI(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-1,1])
        for k in range(self.NA):
            all_Iothers = []  
            for i in range(self.NA):
                for j in range(self.NA):              
                    if not (i==j and i==k):
                        ax.step(timeline, np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j]), where='post', color = Acolor[k], linewidth = 2)
                        ax.step(timeline, np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j]), where='post', color = Acolor[j], linewidth = 1)
                        ax.step(timeline, np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j]), where='post', color = Acolor[i], linewidth = 1, linestyle = (0,(1,2)))
                        # ticks
                        ticks = [0,250,500,750,1000,1250,1500,1750]
                        ax.set_xticks(ticks)
                        dic = {0 : "0", 250 : "250", 500 : "500", 750 : "750", 1000 : "1000", 1250 : "1250", 1500 : "1500", 1750 : "1750"}
                        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
                        # errorbars, very differentiated
                        ax.set_xlim([0,self.NA*self.NR*2*(1.02+(self.NA-1)/20+(self.NA-1)/80+(self.NA-1)/100)])
                        ax.errorbar(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j])), \
                                    np.std(np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j])),\
                                    marker='o', color = Acolor[k], capsize = 2, markersize = 0, elinewidth = 1, capthick = 1.5, zorder = -1)
                        ax.scatter(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j])), \
                                    color = Acolor[k], s=14, zorder = 5)
                        ax.scatter(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j])), \
                                    color = Acolor[i], s=6, zorder = 10)
                        ax.scatter(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Iothers[k][i][j])-np.array(self.Idata[i][j])), \
                                    color = Acolor[j], s=2, zorder = 20)

        legend1 = [Line2D([0], [0], color='black', lw=6, label="black's view on what white thinks about grey")]
        legend2 = [Line2D([0], [0], color='grey', lw=3, label="black's view on what white thinks about grey")]
        legend3 = [Line2D([0], [0], color='white', lw=3, label="black's view on what white thinks about grey", linestyle=(0,(1,2)))]
        leg1 = ax.legend(handles=legend1, loc='best', frameon=True, framealpha = 0)   
        leg2 = ax.legend(handles=legend2, loc='best', frameon=True, framealpha = 0) 
        leg3 = ax.legend(handles=legend3, loc='best', frameon=True, framealpha = 0) 
        plt.setp(leg1.get_texts(), color='w')
        plt.setp(leg2.get_texts(), color='w')
        ax.add_artist(leg1)
        ax.add_artist(leg2)   
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        ax.set_xticks(Ticks)           
        plt.xlabel(make_latex_str('time $t$'), fontsize = 20)
        plt.ylabel(make_latex_str('misinformation'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_ToMI.png')
        plt.close() 
        if verbose: print("Plot in", name+'_ToMI.png')

    def plot_ToMC(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-1,1])
        for k in range(self.NA):
            all_Cothers = []  
            for i in range(self.NA):
                for j in range(self.NA):              
                    if not ((i==j and i==k) or  k==i):
                        ax.step(timeline, np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j]), where='post', color = Acolor[k], linewidth = 2)
                        ax.step(timeline, np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j]), where='post', color = Acolor[j], linewidth = 1)
                        ax.step(timeline, np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j]), where='post', color = Acolor[i], linewidth = 1, linestyle = (0,(1,2)))
                        # ticks
                        ticks = [0,250,500,750,1000,1250,1500,1750]
                        ax.set_xticks(ticks)
                        dic = {0 : "0", 250 : "250", 500 : "500", 750 : "750", 1000 : "1000", 1250 : "1250", 1500 : "1500", 1750 : "1750"}
                        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
                        # errorbars, differentiated
                        ax.set_xlim([0,self.NA*self.NR*2*(1.02+(self.NA-1)/20+(self.NA-1)/80+(self.NA-1)/100)])
                        ax.errorbar(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j])), \
                                    np.std(np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j])),\
                                    marker='o', color = Acolor[k], capsize = 2, markersize = 0, elinewidth = 1, capthick = 1.5, zorder = -1)
                        ax.scatter(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j])), \
                                    color = Acolor[k], s=14, zorder = 5)
                        ax.scatter(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j])), \
                                    color = Acolor[i], s=6, zorder = 10)
                        ax.scatter(self.NA*self.NR*2*(1.02+k/20+i/80+j/100), np.mean(np.array(self.Cothers[k][i][j])-np.array(self.Iothers[i][k][j])), \
                                    color = Acolor[j], s=2, zorder = 20)

        legend1 = [Line2D([0], [0], color='black', lw=6, label="black's view on what white assumes that black thinks about grey")]
        legend2 = [Line2D([0], [0], color='grey', lw=3, label="black's view on what white assumes that black thinks about grey")]
        legend3 = [Line2D([0], [0], color='white', lw=3, label="black's view on what white assumes that black thinks about grey", linestyle = (0,(1,2)))]
        leg1 = ax.legend(handles=legend1, loc='best', frameon=True, framealpha = 0)   
        leg2 = ax.legend(handles=legend2, loc='best', frameon=True, framealpha = 0) 
        leg3 = ax.legend(handles=legend3, loc='best', frameon=True, framealpha = 0) 
        plt.setp(leg1.get_texts(), color='w')
        plt.setp(leg2.get_texts(), color='w')
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        ticks = ax.get_xticks()
        Ticks = []
        for tick in ticks:
            if tick <= timeline[-1]:
                Ticks.append(tick)
        ax.set_xticks(Ticks)
        plt.xlabel(make_latex_str('time $t$'), fontsize = 20)
        plt.ylabel(make_latex_str('misinformation'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_ToMC.png')
        plt.close() 
        if verbose: print("Plot in", name+'_ToMC.png')

    def plot_kappa(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-5, 5])
        ax.set_xlim([0,self.NA*self.NR*2*(1.02+4/100)])
        ax.plot([0,timeline[-1]],[0,0], color='grey', linewidth = 0.5)        
        for i in range(self.NA):
            data = np.log10(np.array(self.kappa[i]))
            ax.step(timeline, data, where='post', color = Acolor[i], linewidth = 2)
            ax.scatter([self.NA*self.NR*2*(1.02+i/100)], [np.mean(data)], color = Acolor[i], s = 70)
            ax.errorbar(self.NA*self.NR*2*(1.02+i/100), np.mean(data), np.std(data), \
                                 color = Acolor[i], capsize = 5, capthick = 1.5, markersize=15)
            
        
        # ticks
        ticks = [-3, -2, -1, 0, 1, 2, 3]
        ax.set_yticks(ticks)
        dic = {t : '10$^{'+str(t)+'}$' for t in ticks}
        #dic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ax.set_yticklabels(labels)
        
        #plt.yscale('log')
        plt.xlabel(make_latex_str('time $t$'), fontsize = 30) #20
        #plt.ylabel('surprise scale $log_{10}\kappa)$', fontsize=14)
        plt.ylabel(make_latex_str('surprise scale $\kappa_{a}$'), fontsize = 30)
        plt.title(make_latex_str(self.title), fontsize = 30)
        plt.xticks(fontsize=18) #12
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_kappa.png', dpi=200)
        plt.close() 
        if verbose: print("Plot in", name + '_kappa.png')

    def plot_rep_comparison(self, name, dc):
        Acolor = self.colors
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        av_rep0 = np.average([self.Idata[j][0] for j in range(self.NA) if j != 0], axis=0)
        for i in range(1,self.NA): # all but red
            av_rep = np.average([self.Idata[j][i] for j in range(self.NA) if j != i], axis=0)
            ax.plot(av_rep, av_rep0, color = Acolor[i])
        ax.plot([0,1],[0,1], color = 'black')
        
        plt.xlabel(make_latex_str("other's reputation"), fontsize = 20)
        plt.ylabel(make_latex_str("red's reputation"), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_rep_comp.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rep_comp.png')

    def plot_relations(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim([0,self.NA*self.NR*2])        
        for i in range(self.NA):
            for j in range(self.NA):
                if i!=j:
                    #print(i,j)
                    #print(self.relationsc[i][j])
                    ax.step(timeline, self.relationsc[i][j], where='post', color = Acolor[j], linewidth=3) 
                    ax.step(timeline, self.relationsc[i][j], where='post', color = Acolor[i], linewidth=3, linestyle = ':')
                    ax.step(timeline, self.relationsm[i][j], where='post', color = Acolor[j], linewidth=1) 
                    ax.step(timeline, self.relationsm[i][j], where='post', color = Acolor[i], linewidth=1, linestyle = ':')        
        plt.xlabel('time', fontsize=20)
        plt.ylabel('relation', fontsize=20)
        plt.title(self.title, fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_rel.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rel.png')
    
    def plot_relation_network(self, name, dc):
        fig, ax = plt.subplots()
        G = nx.Graph()
        weights = []
        edge_colors = []       
        for i in range(self.NA):
            G.add_node(i)
            for j in range(self.NA):
                if i != j and i<j:
                    weight_c = 0.5*(self.relationsc[i][j][-1] + self.relationsc[j][i][-1])
                    weight_m = 0.5*(self.relationsm[i][j][-1] + self.relationsm[j][i][-1])
                    weight = (Q*weight_c+weight_m)/(Q + 1)
                    weights.append(weight)
                    asymetry = (Q*self.relationsc[i][j][-1] + self.relationsm[i][j][-1]) - (Q*self.relationsc[j][i][-1] + self.relationsm[j][i][-1])
                    if self.relationsc[i][j][-1] != self.relationsc[j][i][-1]:
                        raise ValueError('number of communications must be symmetric!')
                    if asymetry>0: # relation is mostly from i to j
                        edge_colors.append(self.colors[i])
                        color=self.colors[i]
                    else: # relation mostly from j to i
                        edge_colors.append(self.colors[j])
                        color=self.colors[j]
                    G.add_edge(i, j, weight=weight, color=color)
        label_dict = {}
        for node in G.nodes:
            label_dict[node]=str(node)
        sizes = [1000*self.reputation[i] for i in range(self.NA)]
        colors = [self.colors[i] for i in range(self.NA)]            
        pos = nx.circular_layout(G)
        nx.drawing.nx_pylab.draw_networkx_nodes(G, pos=pos, node_size=sizes, node_color=colors)
        #nx.draw(G, pos=pos, node_size=sizes, node_color=colors, font_color='white', with_labels=True, font_weight='bold')
        nx.drawing.nx_pylab.draw_networkx_edges(G, pos=pos, width=np.array(weights)/100, edge_color=edge_colors)
        nx.drawing.nx_pylab.draw_networkx_labels(G, pos=pos, labels=label_dict, font_color='white', font_weight='bold')

        plt.axis('off')
        plt.title(self.title, fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_rel_net_asym_circ.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rel_net_asym_circ.png')

    def plot_relation_network_c(self, name, dc):
        fig, ax = plt.subplots()
        G = nx.Graph()
        weights = []
        #edge_colors = []       
        for i in range(self.NA):
            G.add_node(i)
            for j in range(self.NA):
                if i != j and i<j:
                    weight = self.relationsc[i][j][-1] # symmetric between i and j
                    weights.append(weight)
                    #asymetry = (rel_factor*self.relationsc[i][j][-1] + self.relationsm[i][j][-1]) - (rel_factor*self.relationsc[j][i][-1] + self.relationsm[j][i][-1])
                    #print(self.relationsc[i][j][-1], self.relationsm[i][j][-1], self.relationsm[j][i][-1])
                    #if asymetry>0: # relation is mostly from i to j
                    #    edge_colors.append(self.colors[i])
                    #    color=self.colors[i]
                    #else: # relation mostly from j to i
                    #    edge_colors.append(self.colors[j])
                    #    color=self.colors[j]
                    G.add_edge(i, j, weight=weight, color='black')
        label_dict = {}
        for node in G.nodes:
            label_dict[node]=str(node)
        sizes = [1000*self.reputation[i] for i in range(self.NA)]
        colors = [self.colors[i] for i in range(self.NA)]            
        pos = nx.circular_layout(G)
        nx.drawing.nx_pylab.draw_networkx_nodes(G, pos=pos, node_size=sizes, node_color=colors)
        #nx.draw(G, pos=pos, node_size=sizes, node_color=colors, font_color='white', with_labels=True, font_weight='bold')
        nx.drawing.nx_pylab.draw_networkx_edges(G, pos=pos, width=np.array(weights)/70)
        nx.drawing.nx_pylab.draw_networkx_labels(G, pos=pos, labels=label_dict, font_color='white', font_weight='bold')

        plt.axis('off')
        plt.title(self.title, fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_rel_net_t_circ.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rel_net_t.png')

    def plot_EBEH_parameter_evolution(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.colors
        Aalpha = self.alphas
        fig, ax = plt.subplots(figsize=(10, 6))
        lower_lim = -0.01
        upper_lim = 0.2
        ax.set_ylim([lower_lim, upper_lim])
        #ax.set_xlim([0,self.NA*self.NR*2])
        ax.set_xlim([0,self.NA*self.NR*2])

        ax.plot(timeline, [0]*len(timeline), color='black', linestyle='solid', lw=1)
        ax.plot(timeline, [0.1]*len(timeline), color='black', linestyle='solid', lw=1)

        # just for labeling:
        ax.plot(timeline, self.prob_bt[0], color = 'black', linestyle='dashed', lw=3, label='P(blush|dishonest)')
        ax.plot(timeline, self.prob_bl[0], color = 'black', linestyle='solid', lw=3, label='P(blush|honest)')

        #ax.scatter(self.NA*self.NR*2*(1.01), 0.1, c='lightgray', s=10, marker='X')
        #ax.scatter(self.NA*self.NR*2*(1.01), 0, c='dimgray', s=50, marker='X')

        for i in np.arange(self.NA):
            
            bt = np.array(self.prob_bt)[i]
            bt_rms = np.array(self.prob_bt_rms)[i]
            bl = np.array(self.prob_bl)[i]
            bl_rms = np.array(self.prob_bl_rms)[i]
            
            # plot time evolution of parameters
            ax.plot(timeline, bt, color = Acolor[i], linestyle='solid', lw=3)
            ax.plot(timeline, bl, color = Acolor[i], linestyle='dashed', lw=3)

            #ax.plot(timeline, bt+0.1*bt, color = 'dimgray', linestyle='solid', lw=1)
            #ax.plot(timeline, bl-0.1*bl, color = 'lightgray', linestyle='solid', lw=1)
            #ax.plot(timeline, bt+0.1*bt, color = Acolor[i], linestyle=(0,(1,10)), lw=1)
            #ax.plot(timeline, bl-0.1*bl, color = Acolor[i], linestyle=(0,(1,10)), lw=1)

            # mark BADE
            upper = bl - 0.1*bl
            lower = bt  + 0.1*bt
            crit_indices = list(np.where((upper-lower)<0)[0])
            lower_limits = [min(bl[_], bt[_])-0.0125 for _ in crit_indices]
            upper_limits = [max(bl[_], bt[_])+0.0125 for _ in crit_indices]
            ax.fill_between(crit_indices, upper_limits, lower_limits, color='yellow', alpha=0.8)

        if False: # plot perfect curves
            ax.plot(timeline, [0.1]*len(timeline), color = 'black', linestyle='dashed', lw=3, label='P(blush|dishonest)')
            ax.plot(timeline, [0]*len(timeline), color = 'black', linestyle='solid', lw=3, label='P(blush|honest)')
            #for i in np.arange(self.NA):            
            #    ax.plot(timeline, [0]*len(timeline), color = Acolor[i], linestyle=(3*i,(1,10)), lw=3)
            #    ax.plot(timeline, [0.1]*len(timeline), color = Acolor[i], linestyle=(3*i,(1,10)), lw=3)
        
        yticks = [0,0.05, 0.1, 0.15, 0.2]
        ax.set_yticks(yticks)
        xticks = [_ for _ in timeline if _%1000==0]
        ax.set_xticks(xticks)

        plt.xlabel(make_latex_str('time'), fontsize = 20) #20
        #plt.xlabel(make_latex_str('time $t$'), fontsize = 20) #20
        plt.ylabel(make_latex_str('perceived blushing frequencies'), fontsize = 20) 
        plt.title(make_latex_str(self.title), fontsize = 24)
        plt.xticks(fontsize=18)#12
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_EBEH_params.png')
        plt.close() 
        if verbose: print("Plot in", name + '_EBEH_params.png')


def create_data_stat(mode, NA, NR, Nstat, recs):
    rel_recs = recs
    # data (honesy)
    IdataArray = [rec.Idata for rec in rel_recs] # IdataArray[which rec][who thinks][about whom][time]
    IdataArray_rms = [rec.Idata_rms for rec in rel_recs]
    # friendships
    friendships = [rec.friendships for rec in recs]
    friendships_rms = [rec.friendships_rms for rec in recs]
    # relations
    relationsc = [rec.relationsc for rec in recs]
    relationsm = [rec.relationsm for rec in recs]
    # kappa
    kappaArray = [rec.kappa for rec in rel_recs]
    # K
    KArray = [rec.K for rec in rel_recs]
    # yc
    ycArray = [rec.yc for rec in rel_recs]
    # honesty
    honestyArray = [rec.honesty for rec in rel_recs]

    # ToM
    IothersArray = [rec.Iothers for rec in rel_recs]

    # EBEH parameters
    prob_btArray = [rec.prob_bt for rec in rel_recs] # [rec][who's parameter][time]
    prob_btArray_rms = [rec.prob_bt_rms for rec in rel_recs]
    prob_blArray = [rec.prob_bl for rec in rel_recs] # [rec][who's parameter][time]
    prob_blArray_rms = [rec.prob_bl_rms for rec in rel_recs]
    prob_ctArray = [rec.prob_ct for rec in rel_recs] # [rec][who's parameter][time]
    prob_ctArray_rms = [rec.prob_ct_rms for rec in rel_recs]
    prob_clArray = [rec.prob_cl for rec in rel_recs] # [rec][who's parameter][time]
    prob_clArray_rms = [rec.prob_cl_rms for rec in rel_recs]    

    all_x_est = np.array([rel_recs[_].x_est for _ in range(len(rel_recs))])
    av_x_est = list(np.mean(all_x_est, axis=0))

    d = {}
    d['mode'] = mode
    d['Nstat'] = Nstat
    d['NA'] = NA
    d['NR'] = NR
    d['Acolor'] = rel_recs[0].colors
    d['Aalpha'] = rel_recs[0].alphas
    d['x_est'] = av_x_est
    d['title'] = rel_recs[0].title
    d['CONTINUOUS_FRIENDSHIP'] = rel_recs[0].CONTINUOUS_FRIENDSHIP
    d['IdataArray'] = IdataArray
    d['IdataArray_rms'] = IdataArray_rms
    d['friendships'] = friendships
    d['friendships_rms'] = friendships_rms
    d['relationsc'] = relationsc
    d['relationsm'] = relationsm
    d['kappaArray'] = kappaArray
    d['KArray'] = KArray
    d['ycArray'] = ycArray
    d['honestyArray'] = honestyArray
    d['IothersArray'] = IothersArray
    d['prob_btArray'] = prob_btArray
    d['prob_blArray'] = prob_blArray
    d['prob_ctArray'] = prob_ctArray
    d['prob_clArray'] = prob_clArray
    d['prob_btArray_rms'] = prob_btArray_rms
    d['prob_blArray_rms'] = prob_blArray_rms
    d['prob_ctArray_rms'] = prob_ctArray_rms
    d['prob_clArray_rms'] = prob_clArray_rms

    data_stat = Data_Stat(d)

    return data_stat
   
class Data_Stat:
    """contains all recs belonging to one stat evaluation: only 1 mode and NA."""
    def __init__(self, d):
    #def __init__(self, mode, Nstat, NA, NR, Acolor, Aalpha, x_est, title, CONTINUOUS_FRIENDSHIP, IdataArray, IdataArray_rms, friendships, friendships_rms, relationsc, relationsm, kappaArray, KArray, ycArray, honestyArray,\
    #def __init__(self, mode, Nstat, NA, NR, Acolor, Aalpha, x_est, title, IdataArray, IdataArray_rms, friendships, kappaArray,\
                    #IothersArray, #prob_btArray, prob_blArray, prob_ctArray, prob_clArray, prob_btArray_rms, prob_blArray_rms, prob_ctArray_rms, prob_clArray_rms
                    #prob_btArray, prob_blArray, prob_ctArray, prob_clArray, prob_btArray_rms, prob_blArray_rms, prob_ctArray_rms, prob_clArray_rms
                    #):
        self.mode = d['mode']
        self.Nstat = d['Nstat']
        self.NA = d['NA']
        self.NR = d['NR']
        self.Acolor = d['Acolor']
        self.Aalpha = d['Aalpha']
        self.x_est = d['x_est']
        self.title = d['title']
        if 'CONTINUOUS_FRIENDSHIP' in d.keys(): self.CONTINUOUS_FRIENDSHIP = d['CONTINUOUS_FRIENDSHIP']
        self.IdataArray = d['IdataArray'] # IdataArray[which rec][who thinks][about whom][time]
        self.IdataArray_rms = d['IdataArray_rms'] # IdataArray_rms[which rec][who thinks][about whom][time]
        self.Ireph = []
        self.bin_edges = []
        self.friendships = d['friendships']
        if 'friendships_rms' in d.keys(): self.friendships_rms = d['friendships_rms']
        if 'relationsc' in d.keys(): self.relationsc = d['relationsc']
        if 'relationsm' in d.keys(): self.relationsm = d['relationsm']
        self.kappaArray = d['kappaArray'] # [which rec][who's kappa][time]
        if 'KArray' in d.keys(): self.KArray = d['KArray']
        if 'ycArray' in d.keys(): self.ycArray = d['ycArray'] # [which rec][who assigned this yc][time when message was sent]
        if 'honestyArray' in d.keys(): self.honestyArray = d['honestyArray'] # [which rec][who was honest][at what time]
        if 'IothersArray' in d.keys(): self.IothersArray = d['IothersArray']
        if 'prob_btArray' in d.keys():
            self.prob_btArray = d['prob_btArray'] 
            self.prob_blArray = d['prob_blArray']
            self.prob_ctArray = d['prob_ctArray']
            self.prob_clArray = d['prob_clArray']
            self.prob_btArray_rms = d['prob_btArray_rms']
            self.prob_blArray_rms = d['prob_blArray_rms']
            self.prob_ctArray_rms = d['prob_ctArray_rms']
            self.prob_clArray_rms = d['prob_clArray_rms']

    def save(self, name):
        # make dictionary
        data_stat_dict = self.__dict__
        # make everything json seriazable
        for k,v in data_stat_dict.items():
            if isinstance(v, np.ndarray):
                data_stat_dict[k] = v.tolist()



        #data_stat_dict = {'mode': self.mode, 'Nstat':self.Nstat, 'NA':self.NA, 'NR':self.NR, 'Acolor':self.Acolor, 'Aalpha':self.Aalpha, 'x_est':self.x_est, \
                #'title':self.title, 'CONTINUOUS_FRIENDSHIP': self.CONTINUOUS_FRIENDSHIP, 'IdataArray':self.IdataArray, 'IdataArray_rms':self.IdataArray_rms, 'Ireph':self.Ireph.tolist(), 'bin_edges':self.bin_edges.tolist(), \
                #'title':self.title, 'IdataArray':self.IdataArray, 'IdataArray_rms':self.IdataArray_rms, 'Ireph':self.Ireph.tolist(), 'bin_edges':self.bin_edges.tolist(), \
                #'friendships':self.friendships, 'friendships_rms':self.friendships_rms, 'relationsc':self.relationsc, 'relationsm':self.relationsm, 'kappaArray':self.kappaArray, 'KArray': self.KArray, 'ycArray': self.ycArray, 'honestyArray': self.honestyArray, 'IothersArray':self.IothersArray, \
                #'friendships':self.friendships, 'kappaArray':self.kappaArray, \
                #'prob_btArray': self.prob_btArray, 'prob_blArray': self.prob_blArray, 'prob_ctArray': self.prob_ctArray, 'prob_clArray': self.prob_clArray, \
                #'prob_btArray_rms': self.prob_btArray_rms, 'prob_blArray_rms': self.prob_blArray_rms, 'prob_ctArray_rms': self.prob_ctArray_rms, 'prob_clArray_rms': self.prob_clArray_rms
                #}
        # save it
        with open(name+'_data_stat.json', 'w+') as f:
            json.dump({'name':name}, f)
            f.write('\n')
            json.dump(data_stat_dict, f)
            f.write('\n')

    def plot_statistics(self, name, dc, ignore = {}):
        NA = self.NA
        NR = self.NR
        Nstat = self.Nstat
        Acolor = self.Acolor 
        Aalpha = self.Aalpha 
        x_est = self.x_est 
        # prepare stat plot
        title = self.title
        timeline = np.arange(0, NA*NR*2)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([0, NA*NR*2])
        # prepare histogram
        Nbins = 20
        Idata_mean_h = np.zeros((NA,NA,Nbins))
        Ireph = np.zeros((NA,Nbins))
        Iawah = np.zeros((NA,Nbins))
        IA = np.zeros((Nstat,NA,NA,NA*NR*2)) # IdataArray without values for t=0
        for n in range(Nstat):
            #print('n: ', n)
            for i in range(NA):
                #print('i: ', i)
                for j in range(NA):
                    #print('j: ', j)
                    IA[n][i][j] = self.IdataArray[n][i][j][1:] # remove t=0
        for i in range(NA):
            a = np.append(IA[:,:i,i,:], IA[:,i+1:,i,:]) #other agents than j
            Ireph[i], bin_edges = np.histogram(a = a, bins=Nbins,\
                                                    range=(0,1), density=True)
            Iawah[i], bin_edges = np.histogram(a = IA[:,i,i,:], bins=Nbins,\
                                                    range=(0,1), density=True)
            for j in range(NA):
                Idata_mean_h[j][i], bin_edges = np.histogram(a = IA[:,j,i,:], bins=Nbins,\
                                                    range=(0,1), density=True)
        self.bin_edges = bin_edges
        self.Ireph = Ireph
        self.Iawah = Iawah
        self.IA = IA
        ####################################################
        # calculate some quantities
        Idata_mean = np.mean(self.IdataArray, axis=0)
        Idata_std = np.std(self.IdataArray, axis=0)
        awa_stat = [np.mean(Idata_mean[i][i]) for i in range(NA)]
        awa_stat_std = [np.std(Idata_mean[i][i]) for i in range(NA)]
        reps_stat = []
        for i in range(NA):
            array = np.array([])
            for j in range(NA):
                if j!=i:
                    array = np.concatenate((array, Idata_mean[j][i]))
            reps_stat.append(array)
        rep_stat = [np.mean(reps_stat[i]) for i in range(NA)]
        rep_stat_std = [np.std(reps_stat[i]) for i in range(NA)]
        # make stat plot
        for i in np.arange(NA):
            for j in np.arange(NA):
                if (i,j) not in ignore and i!=j:
                    if compatibility: # diagonal lines
                        ax.plot(timeline, Idata_mean[i][j][1:], color = Acolor[j], lw=3) #where='post', 
                        ax.plot(timeline, Idata_mean[i][j][1:], color = Acolor[i], dashes=[1,4], lw=3) #where='post',
                        ax.fill_between(timeline, \
                                        np.array(Idata_mean[i][j][1:])-np.array(Idata_std[i][j][1:]), \
                                        np.array(Idata_mean[i][j][1:])+np.array(Idata_std[i][j][1:]), \
                                        color = Acolor[j], alpha = 0.05*Aalpha[j]) #step = 'post'
                    else: # rectengular lines
                        ax.step(timeline, Idata_mean[i][j], where='post', color = Acolor[j]) #where='post', 
                        ax.step(timeline, Idata_mean[i][j], where='post', color = Acolor[i], dashes=[1,4]) #where='post',
                        ax.fill_between(timeline, \
                                        np.array(Idata_mean[i][j])-np.array(Idata_std[i][j]), \
                                        np.array(Idata_mean[i][j])+np.array(Idata_std[i][j]), \
                                        step = 'post', color = Acolor[j], alpha = 0.05*Aalpha[j]) #step = 'post'
        for i in np.arange(NA):
            if (i,i) not in ignore:
                pass
                #if compatibility: # diagonal lines
                #    ax.plot(timeline, Idata_mean[i][i][1:], color = Acolor[i], linewidth = 3)
                #    ax.fill_between(timeline, np.array(Idata_mean[i][i][1:])-np.array(Idata_std[i][i][1:]), \
                #                    np.array(Idata_mean[i][i][1:])+np.array(Idata_std[i][i][1:]), \
                #                    color = Acolor[i], alpha = 0.15*Aalpha[i])
                #else:
                #    ax.step(timeline, Idata_mean[i][i][1:], where='post', color = Acolor[i], linewidth = 3)
                #    ax.fill_between(timeline, np.array(Idata_mean[i][i][1:])-np.array(Idata_std[i][i][1:]), \
                #                    np.array(Idata_mean[i][i][1:])+np.array(Idata_std[i][i][1:]), \
                #                    step = 'post', color = Acolor[i], alpha = 0.15*Aalpha[i])
            
            if (i) not in ignore:
                #ax.set_xlim([0,NA*NR*2*(1.01+(NA+NA)/100)])
                ax.plot([0,NA*NR*2], [x_est[i], x_est[i]],  color = Acolor[i],\
                        linewidth = 1, linestyle = 'solid')
                #ax.errorbar(NA*NR*2*(1.01+i/100), rep_stat[i], rep_stat_std[i],\
                #            marker='s', markersize=9, color = Acolor[i], capsize = 5, capthick = 1.5)
                #ax.errorbar(NA*NR*2*(1.01+(NA+i)/100), awa_stat[i], awa_stat_std[i],\
                #            marker='o', markersize=9, color = Acolor[i], capsize = 5, capthick = 1.5) 

        Ticks = [_ for _ in timeline if _%1000==0] #+ [timeline[-1]+1]
        ax.set_xticks(Ticks)
                 
        plt.xlabel(make_latex_str('time'), fontsize = 20)
        #plt.ylabel(r'$\mathrm{perceived\ honesty\ }\overline{x}_{ab}$', fontsize = 20)
        plt.ylabel('perceived honesties', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'.png')
        plt.close() 
        if verbose: print("Plot in", name + '.png')

        return Ireph[0]

    def plot_statistics_simple(self, name, dc, ignore = {}):
        NA = self.NA
        NR = self.NR
        Nstat = self.Nstat
        Acolor = self.Acolor 
        Aalpha = self.Aalpha 
        x_est = self.x_est 
        # prepare stat plot
        title = self.title
        timeline = np.arange(0, NA*NR*2)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([0, NA*NR*2])
        # prepare histogram
        Nbins = 20
        Idata_mean_h = np.zeros((NA,NA,Nbins))
        Ireph = np.zeros((NA,Nbins))
        Iawah = np.zeros((NA,Nbins))
        IA = np.zeros((Nstat,NA,NA,NA*NR*2)) # IdataArray without values for t=0
        for n in range(Nstat):
            #print('n: ', n)
            for i in range(NA):
                #print('i: ', i)
                for j in range(NA):
                    #print('j: ', j)
                    IA[n][i][j] = self.IdataArray[n][i][j][1:] # remove t=0
        for i in range(NA):
            a = np.append(IA[:,:i,i,:], IA[:,i+1:,i,:]) #other agents than j
            Ireph[i], bin_edges = np.histogram(a = a, bins=Nbins,\
                                                    range=(0,1), density=True)
            Iawah[i], bin_edges = np.histogram(a = IA[:,i,i,:], bins=Nbins,\
                                                    range=(0,1), density=True)
            for j in range(NA):
                Idata_mean_h[j][i], bin_edges = np.histogram(a = IA[:,j,i,:], bins=Nbins,\
                                                    range=(0,1), density=True)
        self.bin_edges = bin_edges
        self.Ireph = Ireph
        self.Iawah = Iawah
        self.IA = IA
        ####################################################
        # calculate some quantities
        Idata_mean = np.mean(self.IdataArray, axis=0)
        Idata_std = np.std(self.IdataArray, axis=0)
        awa_stat = [np.mean(Idata_mean[i][i]) for i in range(NA)]
        awa_stat_std = [np.std(Idata_mean[i][i]) for i in range(NA)]
        reps_stat = []
        for i in range(NA):
            array = np.array([])
            for j in range(NA):
                if j!=i:
                    array = np.concatenate((array, Idata_mean[j][i]))
            reps_stat.append(array)
        rep_stat = [np.mean(reps_stat[i]) for i in range(NA)]
        rep_stat_std = [np.std(reps_stat[i]) for i in range(NA)]
        # make stat plot
        for i in np.arange(NA):
            rep = np.array([0.0 for _ in range(len(timeline))])
            for j in np.arange(NA):
                if i!=j:
                    rep += np.array(Idata_mean[j][i][1:])
            rep = rep/(NA-1)

            ax.plot(timeline, rep, color = Acolor[i], lw=3) #where='post', 
            ax.plot([0,NA*NR*2], [x_est[i], x_est[i]],  color = Acolor[i], linewidth = 1, linestyle = 'solid')
            

        Ticks = [_ for _ in timeline if _%2000==0]# + [timeline[-1]+1]
        ax.set_xticks(Ticks)
                 
        plt.xlabel(make_latex_str('time'), fontsize = 20)
        #plt.ylabel(r'$\mathrm{perceived\ honesty\ }\overline{x}_{ab}$', fontsize = 20)
        plt.ylabel('reputation', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_s.png')
        plt.close() 
        if verbose: print("Plot in", name + '_s.png')
        
    def plot_histrogram(self, name, dc, ignore={}): # plot histogram - FONT: Check
        NA = self.NA
        NR = self.NR
        Nstat = self.Nstat
        Acolor = self.Acolor 
        Aalpha = self.Aalpha 
        x_est = self.x_est 
        # prepare stat plot
        title = self.title
        timeline = np.arange(0, NA*NR*2)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([0.02,20])
        ax.set_xlim([0,1])
        ax.set_yscale('log')
        ax.plot([0,1],[1,1], color='grey', linewidth = 0.5)
        for i in range(NA): # plot awareness and reputation
            if (i,i) not in ignore:
                ax.plot([x_est[i], x_est[i]],  [0.001,100], color = Acolor[i],\
                        linewidth = 3, linestyle = '--')            
                ax.hist(self.bin_edges[:-1], self.bin_edges, weights=self.Ireph[i], histtype='step',\
                        color = Acolor[i], linewidth = 1)
                ax.hist(self.bin_edges[:-1], self.bin_edges, weights=self.Iawah[i], histtype='step',\
                        color = Acolor[i], linewidth = 3)    
        for i in range(NA): # plot awareness and reputation
            if (i,i) not in ignore:           
                ax.hist(self.bin_edges[:-1], self.bin_edges, weights=self.Ireph[i], histtype='stepfilled',\
                        color = Acolor[i], linewidth = 1, alpha=0.10*Aalpha[i])
        plt.xlabel(make_latex_str('perceived honesty'), fontsize = 20)
        plt.ylabel(make_latex_str('frequency density'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+"h.png")
        plt.close() 
        if verbose: print("Histogram plot in",name+"h.png")
    
    def plot_rep_rep_nifty(self, name, dc, ignore={}): # - NEW ESTIMATOR: Check (lower contour) - FONT: Check
        NA = self.NA
        NR = self.NR
        Nstat = self.Nstat
        Acolor = self.Acolor 
        Aalpha = self.Aalpha 
        x_est = self.x_est 
        # prepare stat plot
        title = self.title
        timeline = np.arange(0, NA*NR*2)
        """reputation comparison of agent red against others"""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_ylim([0,100])
        ax.set_xlim([0,100])
        
        RepA = [i for i in range(NA)]
        ax.plot([0,100], [100*x_est[0], 100*x_est[0]],   color = Acolor[0],\
                        linewidth = 3, linestyle = '--')
        ax.plot([0,100],[0,100], color='black', linewidth = 1)

        densities = {}
        for i in [0]+list(range(1,NA))[::-1]: # reverse order # plot reputation comparison
        #for i in [0,1]:
            RepA[i] = np.append(self.IA[:,:i,i,:], self.IA[:,i+1:,i,:], axis=1) #other agents than i
            #breakpoint()
            RepA[i] = np.average(RepA[i], axis=1) # average their reputation over other agents
            RepA[i] = RepA[i].flatten() # make a flat list out of it for binning
            if (i,i) not in ignore and i>0:
                ax.plot([x_est[i]*100, x_est[i]*100], [0,100],   color = Acolor[i],\
                        linewidth = 3, linestyle = '--')

                # reconstruction
                rct_path = get_folder(name)+'/'+get_name(name)+'_rep_rep_nifty_rct/'+get_name(name)+'_rep_rep_nifty_rct_'+str(i)+'.json'
                if os.path.exists(rct_path):
                    densities[str(i)] = load_density(rct_path)
                    print('reloaded previous reconstruction')
                else:
                    densities[str(i)] = estimate_density(np.array(RepA[i])*100, np.array(RepA[0])*100, 101, 101)
                    save_density(densities[str(i)], rct_path)
                
                #ax.scatter(np.array(RepA[i])*100, np.array(RepA[0])*100, s=0.5, color=Acolor[i], alpha = 0.05*Aalpha[i])
        # plot
        for key in densities:
            factor = 1-(NA-1-int(key))*0.3/(NA-2) # transperancy: minimal factor = 1-0.3=0.7 for i=1
            alpha = densities[key]/np.amax(densities[key])
            alpha2 = np.sqrt(alpha)
            my_cmap = mpl.colors.LinearSegmentedColormap.from_list('',[Acolor[int(key)], Acolor[int(key)]])
            ax.imshow(np.zeros(np.shape(alpha)), alpha = factor*alpha2, cmap = my_cmap)
            plt.contour(alpha, levels = [0.005], linewidths=1.5, colors = Acolor[int(key)])
        

        # ticks
        ticks = [0,20,40,60,80,100]
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        dic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        # labels
        plt.ylabel(make_latex_str('red\'s reputation'), fontsize = 30)
        plt.xlabel(make_latex_str('other\'s reputation'), fontsize = 30)
        plt.title(make_latex_str(title), fontsize = 30)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_rep_rep_nifty.png')
        if verbose: print("plot in", name+'_rep_rep_nifty.png')   
        plt.close() 

    
    

    

    def plot_chaos_rep_scatter(self, name, dc): # - FONT:Check
        """averaged reputation vs chaos (=quadraticly added std)"""
        Acolor = self.Acolor 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([0,1])
        ax.set_xlim([0,0.5])
        for i in range(self.NA):
            for j in range(self.NA):
                for r in range(self.Nstat):
                    stds = [[np.std(self.IdataArray[r][I][J]) for J in range(self.NA)] for I in range(self.NA)]
                    stds = np.array(stds).flatten()
                    chaos = np.sqrt(sum((stds)**2))
                    plt.scatter(chaos, np.mean(self.IdataArray[r][i][j]), color = Acolor[i], s=50)
                    plt.scatter(chaos, np.mean(self.IdataArray[r][i][j]), color = Acolor[j], s=10)
        plt.xlabel(make_latex_str('chaos'), fontsize = 20)
        plt.ylabel(make_latex_str('reputation of $a$ with $b$'), fontsize = 20)
        plt.title(make_latex_str(self.mode+' agent among '+str(self.NA-1)+' ordinary agents'), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_chaos_rep.png')
        plt.close() 
        if verbose: print("Plot in", name+'_chaos_rep.png')

    def plot_chaos_rep_nifty(self, name, dc): # - NEW ESTIMATOR: Check - FONT: Check
        """averaged reputation vs chaos"""
        Acolor = self.Acolor 
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_ylim([0,100])
        ax.set_xlim([0,100])
        densities = {}
        Chaos = [] # chaos values for Nstat runs
        for r in range(self.Nstat):
            stds = [[np.std(self.IdataArray[r][I][J]) for J in range(self.NA)] for I in range(self.NA)]
            stds = np.array(stds).flatten()
            chaos = np.sqrt(sum((stds)**2))
            Chaos.append(chaos)
        Reputations = [[[] for i in range(self.NA)] for j in range(self.NA)]
        for i in range(self.NA):
            for j in range(self.NA):
                if i != j:
                    Reputation = []
                    for r in range(self.Nstat):
                        rep = np.mean(self.IdataArray[r][i][j])
                        Reputation.append(rep)
                    Reputations[i][j] = Reputation
        for j in range(self.NA):
        #for j in [0]:
            av_rep = np.zeros((self.Nstat)) # averaged over other agents
            for i in range(self.NA):
                if i != j:
                    av_rep += np.array(Reputations[i][j])
            av_rep = av_rep/(self.NA-1)

            # reconstruction   
            rct_path = get_folder(name)+'/'+get_name(name)+'_chaos_rep_nifty_rct/'+get_name(name)+'_chaos_rep_nifty_rct_'+str(j)+'.json'
            if os.path.exists(rct_path):
                densities[str(j)] = load_density(rct_path)
                print('reloaded previous reconstruction')
            else:
                densities[str(j)] = estimate_density(np.array(Chaos)*100, np.array(av_rep)*100, 101,101)
                save_density(densities[str(j)], rct_path)

            #densities[str(j)] = estimate_density(np.array(Chaos)*100, np.array(av_rep)*100, 101,101)
            #ax.scatter(np.array(Chaos)*100, np.array(av_rep)*100, color = Acolor[j])

        # plot
        for key in densities:
            alpha = densities[key]/np.amax(densities[key])
            cmap = mpl.colors.LinearSegmentedColormap.from_list('',[Acolor[int(key)], Acolor[int(key)]])
            ax.imshow(np.zeros(np.shape(alpha)), alpha = alpha, cmap = cmap)
            plt.contour(alpha, levels = [0.05], linewidths=1.5, colors = Acolor[int(key)])

        # ticks
        ticks = [0,20,40,60,80,100]
        #xticks = [0,10,20,30,40,50]
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        dic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        #labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        """
        #legend
        legend1 = [Ellipse((0,0), 40, 20, ec=None, fc='grey', alpha=0.5, label='$a$')]
        legend2 = [Ellipse((0,0), 40, 20, ec='black', fc= None, fill = False, label='$b$')]
        leg1 = ax.legend(handles=legend1, loc=(0,0.04), frameon=True, framealpha = 0)   
        leg2 = ax.legend(handles=legend2, loc=(0,0), frameon=True, framealpha = 0) 
        ax.add_artist(leg1)
        """
        plt.xlabel(make_latex_str('chaos'), fontsize = 20)
        plt.ylabel(make_latex_str('reputation'), fontsize = 20)
        plt.title(make_latex_str(self.mode+' agent among '+str(self.NA-1)+' ordinary agents'), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_chaos_rep_nifty.png')
        plt.close() 
        if verbose: print("Plot in", name+'_chaos_rep_nifty.png')

    

    def plot_kappa_steps(self, name, dc): # - FONT: Check
        """statistical dymanics of kappa"""
        Acolor = self.Acolor 
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([10**(-4), 10**4])
        ax.set_xlim([0,self.NA*self.NR*2])
        kappa_av = np.average(self.kappaArray, axis = 0)
        kappa_std = np.std(self.kappaArray, axis = 0)
        for i in range(self.NA):
            ax.step(timeline, kappa_av[i][1:], where='post', color = Acolor[i], linewidth = 2)
            #ax.fill_between(timeline, \
            #                            np.array(kappa_av[i][:-1])-np.array(kappa_std[i][:-1]), \
            #                            np.array(kappa_av[i][:-1])+np.array(kappa_std[i][:-1]), \
            #                            color = Acolor[i], alpha = 0.05*Aalpha[i], step = 'post')
            #ax.errorbar(self.NA*self.NR*2*(1.01+i/100), np.exp(np.mean(np.array(self.kappa[i]))), np.std(np.array(self.kappa[i])), \
            #                     color = Acolor[i], capsize = 5, capthick = 1.5, markersize=6)
        plt.yscale('log')
        plt.xlabel(make_latex_str('time $t$'), fontsize = 20)
        plt.ylabel(make_latex_str('kappa'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_kappa.png')
        plt.close() 
        if verbose: print("Plot in", name + '_kappa.png')

    def plot_kappa_comparism_scatter(self, name, dc): # - FONT: Check
        """comparison of different agent's kappas"""
        Acolor = self.Acolor 
        Aalpha = self.Aalpha
        fig, ax = plt.subplots(figsize=(7,7))
        plt.yscale('log')
        plt.xscale('log')
        #ax.set_ylim([10**(-4), 10**4])
        #ax.set_xlim([0,self.NA*self.NR*2])
        for i in range(self.NA-1, 0, -1):
            ax.scatter(np.array([self.kappaArray[R][i] for R in range(self.Nstat)]).flatten(), \
                        np.array([self.kappaArray[R][0] for R in range(self.Nstat)]).flatten(), color = Acolor[i], s=0.1)
        # identity:
        xlimits, ylimits = ax.get_xlim(), ax.get_ylim()
        minimum = max(xlimits[0], ylimits[0])
        maximum = min(xlimits[1], ylimits[1])
        ax.plot([minimum, maximum],[minimum,maximum], color = 'black')

        plt.xlabel(make_latex_str("other's surprise scale"), fontsize = 20)
        plt.ylabel(make_latex_str("red's surprise scale"), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_kappa_comparism.png')
        #plt.savefig('where_is_it.png')
        plt.close() 
        if verbose: print("Plot in", name + '_kappa_comparism.png')

    def plot_surprise_hist(self, name, dc):#Averaged Distributions 
        Acolor = self.Acolor 
        Nstat = len(np.array(self.kappaArray))    
        fig, ax = plt.subplots(figsize = (10,6))
        AllTruths = []
        AllLies = []
        for i in range(self.NA):
            
            #Lists of Surprizes 
            Sj_h = [] #list of surprizes from honest statemets
            Sj_l = [] #list of surprizes from lying statements 
            Honesty = np.array([[self.honestyArray[r][i][t] for t in range(1, self.NA*self.NR*2+1)] for r in range(Nstat)]).flatten() #honesty array of agent i
            K_Values= np.array([[self.KArray[r][i][t] for t in range(1, self.NA*self.NR*2+1)] for r in range(Nstat)]).flatten()
            Kappa_Values = np.array([[self.kappaArray[r][i][t] for t in range(1, self.NA*self.NR*2+1)] for r in range(Nstat)]).flatten()
            for l in range(1, len(Honesty)):
                if Honesty[l] == True:
                    if (l%2)== 0:
                        Sj_h.append(K_Values[l+1]/Kappa_Values[l-1])
                    else: 
                        Sj_h.append(K_Values[l]/Kappa_Values[l-1])
                elif Honesty[l] == False: 
                    if (l%2) == 0:
                        Sj_l.append(K_Values[l+1]/Kappa_Values[l-1])
                    else: 
                        Sj_l.append(K_Values[l]/Kappa_Values[l-1])
            AllTruths += Sj_h
            AllLies += Sj_l 
        N_samples_h = len(Sj_h) 
            
        #Histogram
        bins = np.logspace(np.log10(10**(-4)), np.log10(10**5), 50)  
        bin_height_h, bin_boundary_h = np.histogram(AllTruths, bins) 
        bin_height_l, bin_boundary_l = np.histogram(AllLies, bins)
        #Normalise 
        central_logbins = 0.5*(np.log(bins)[1:]+np.log(bins)[:-1])
        central_bins = np.exp(central_logbins)
        Norm_h = 1/np.dot(central_bins, bin_height_h)
        Norm_l = 1/np.dot(central_bins, bin_height_l)
        bin_height_h = bin_height_h*Norm_h
        bin_height_l = bin_height_l*Norm_l
        
        #Plots
        ax.plot(bin_boundary_h[:-1], bin_height_h, color= 'dodgerblue', alpha = 0.7, lw = 1)
        ax.plot(bin_boundary_h[:-1], bin_height_l, color= 'darkorange', alpha = 0.7, lw = 1)
        ax.bar(bin_boundary_h[:-1], bin_height_h, color= 'dodgerblue', width = 0.7*np.diff(bins), log=True, alpha = 0.3)
        ax.bar(bin_boundary_h[:-1], bin_height_l, color= 'darkorange', width= 0.7*np.diff(bins), log = True,  alpha = 0.2)          
    
        ax.plot([],[], '' , label = 'True Statements', color= 'dodgerblue')
        ax.plot([],[], '' , label = 'Lying Statements', color= 'darkorange')
        ax.legend()

        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlim([central_bins[0], central_bins[-1]*0.8])
        #plt.ylim([1/N_samples_h*0.05,None])

        plt.suptitle('Surprise scale distributions for three ordinary agents' , fontsize = 20)       
        plt.xlabel(make_latex_str('Normalised Surprise Sj'), fontsize = 10)
        plt.ylabel(make_latex_str('P(Sj|state)'), fontsize = 10) #MAKE IT PRETTY
        plt.tight_layout()
        plt.savefig(name+'_surprise_hist.png') 
        plt.close() 
        if verbose: print("Plot in", name + '_surprise_hist.png')
    
    def plot_kappa_rep_nifty(self, name, dc): # - NEW ESTIMATOR: Check - FONT: Check
        """kappa vs reputation"""
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(7,7))
        xmin = -8 # 10**xmin
        xmax = 6 # 10**xmax
        shift = np.abs(xmin) # for nifty reconstruction
        strech = int(np.round(100/(xmax+shift))) # for nifty reconstructions 
        def trafo(data, strech, shift):
            return strech*(np.log10(data)+shift)
        npixels = (xmax+shift)*strech+1
        ax.set_ylim([0,100])
        ax.set_xlim([0, npixels-1])

        densities = {}
        for i in range(self.NA):
            for j in range(self.NA):
                if i != j:
                    kappa_data = np.array([np.mean(self.kappaArray[R][i]) for R in range(self.Nstat)]).flatten()
                    rep_data = np.array([np.mean(self.IdataArray[R][i][j]) for R in range(self.Nstat)]).flatten()

                    # reconstruction  
                    rct_path = get_folder(name)+'/'+get_name(name)+'_kappa_rep_nifty_rct/'+get_name(name)+'_kappa_rep_nifty_rct_'+str(i)+'.json'
                    if os.path.exists(rct_path):
                        densities[str(i)+str(j)] = load_density(rct_path)
                        print('reloaded previous reconstruction')
                    else:
                        densities[str(i)+str(j)] = estimate_density(trafo(kappa_data, strech, shift), rep_data*100, npixels, 101)
                        save_density(densities[str(i)+str(j)], rct_path)

                    #densities[str(i)+str(j)] = estimate_density(trafo(kappa_data, strech, shift), rep_data*100, npixels, 101)
                    #ax.scatter(kappa_data, rep_data*100, color = Acolor[i], s = 50)
                    #ax.scatter(kappa_data, rep_data*100, color = Acolor[j], s = 10)
        
        # plot
        for key in densities:
            alpha = densities[key]/np.amax(densities[key])
            cmap = mpl.colors.LinearSegmentedColormap.from_list('',[Acolor[int(key[1])], Acolor[int(key[1])]])
            ax.imshow(np.zeros(np.shape(alpha)), alpha = alpha, cmap = cmap)
            ax.contour(alpha, levels = [0.01], linewidths=1.5, colors = Acolor[int(key[0])])
        
        # ticks in basis of 10!
        exponents = [i for i in range(-10,10) if trafo(10**xmin, strech, shift) <=  trafo(10**i, strech, shift) <= npixels-1] 
        xticks = [trafo(10**i, strech, shift) for i in exponents]
        yticks = [0,20,40,60,80,100]
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ydic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        xlabels = ['10$^{'+str(i)+'}$' for i in exponents]
        ylabels = [yticks[i] if t not in ydic.keys() else ydic[t] for i,t in enumerate(yticks)]
        ax.set_yticklabels(ylabels)
        ax.set_xticklabels(xlabels)
        
        plt.ylabel(make_latex_str('reputation of $a$ by $b$'), fontsize = 20)
        plt.xlabel(make_latex_str("surprise scale of $b$"), fontsize = 20)
        plt.title(make_latex_str(self.mode+' agent among '+str(self.NA-1)+' ordinary agents'), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_kappa_rep_nifty.png')
        plt.close() 
        if verbose: print("Plot in", name+'_kappa_rep_nifty.png')

    def plot_stat_fr_dyn(self, name, dc):
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.set_ylim([0,1])
        ax.set_xlim([0, self.NA*self.NR*2])           
        for i in np.arange(self.NA):
            for j in np.arange(self.NA):
                if i != j:
                    ax.plot(timeline, np.mean(self.friendships, axis=0)[i][j][1:], color = Acolor[j]) #where='post', 
                    ax.plot(timeline, np.mean(self.friendships, axis=0)[i][j][1:], color = Acolor[i], dashes=[1,4]) #where='post',
                    ax.fill_between(timeline, \
                                    np.array(np.mean(self.friendships, axis=0)[i][j][1:])-np.array(np.std(self.friendships, axis=0)[i][j][1:]), \
                                    np.array(np.mean(self.friendships, axis=0)[i][j][1:])+np.array(np.std(self.friendships, axis=0)[i][j][1:]), \
                                    color = Acolor[j], alpha = 0.05*Aalpha[j]) #step = 'post'
                else:
                    
                    ax.plot(timeline, np.mean(self.friendships, axis=0)[i][i][1:], color = Acolor[i], linewidth = 3)
                    ax.fill_between(timeline, np.array(np.mean(self.friendships, axis=0)[i][i][1:])-np.array(np.std(self.friendships, axis=0)[i][i][1:]), \
                                    np.array(np.mean(self.friendships, axis=0)[i][i][1:])+np.array(np.std(self.friendships, axis=0)[i][i][1:]), \
                                    color = Acolor[i], alpha = 0.15*Aalpha[i])
        plt.xlabel('$\mathrm{time}\ t$', fontsize = 20)
        plt.ylabel(r'$\mathrm{friendship\ strength\ }f_{ab}$', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_stat_fr_dyn.png')
        plt.close() 
        if verbose: print("Plot in", name + '_stat_fr_dyn.png')
    
    def plot_stat_fr(self, name, dc): # - FONT: Check
        """(not average but) final friendship strengths between the agents"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        friendship_sum = [[[0 for i in range(self.NA*self.NR*2)]for i in range(self.NA)] for i in range(self.NA)]
        for r in range(self.Nstat):
            for j in range(self.NA):
                for i in range(self.NA):
                    for t in range(self.NA*self.NR*2):
                        if np.isnan(self.friendships[r][i][j][t]) == False:
                            friendship_sum[i][j][t] += self.friendships[r][i][j][t]
                        else:
                            raise ValueError('friendship value is nan!')
        friendship_average = np.array(friendship_sum)/self.Nstat # normalize
        
        fig, ax = plt.subplots(figsize=(9,9)) 
        # backround : static points
        up = 0.1 # upward shift of all agents (bodies and heads). to be adjusted for >3 agents
        r = 1 # radius on which agents sit
        R = 0.18 # connection point/radius for arrows
        phi = 2*np.pi/self.NA # angle between agents
        eps = np.pi/10 # half the angle between two connection points
        Coord = [[r*np.cos(i*phi), r*np.sin(i*phi)] for i in range(self.NA)] # middle of agents
        # Pentagons
        a = 0.15 # length of each side
        gamma = 3/5*np.pi # angle inside the regular pentagon
        s = 0.03 # downshift compared to the circle (head)
    
        # middle connectionpoints of arrows. coord[i][j] is at i pointing towards j
        coord = [[[Coord[i][0]+R*(Coord[j][0]-Coord[i][0]), Coord[i][1]+R*(Coord[j][1]-Coord[i][1])] for j in range(self.NA)] for i in range(self.NA)] 
        # connection points of arrows. coords[i][j] are two points [[x,y],[x,y]] near i pointing towards j
        coords = [[[rotate(coord[i][j], Coord[i], eps),rotate(coord[i][j], Coord[i], -eps)] for j in range(self.NA)] for i in range(self.NA)] 
        for i in range(self.NA):
            ax.scatter(Coord[i][1], Coord[i][0]+up, color = Acolor[i], s=2000) # agent heads
            # bodies:
            P = (Coord[i][1], Coord[i][0]-s+up) # top point
            A = (Coord[i][1]-a*np.sin(gamma/2),(Coord[i][0]-s)-a*np.cos(gamma/2)+up) # top left
            B = (Coord[i][1]-a/2, (Coord[i][0]-s)-a/2*np.sqrt(5+2*np.sqrt(5))+up) # bottom left
            C = (Coord[i][1]+a/2, (Coord[i][0]-s)-a/2*np.sqrt(5+2*np.sqrt(5))+up) # bottom right
            D = (Coord[i][1]+a*np.sin(gamma/2),(Coord[i][0]-s)-a*np.cos(gamma/2)+up) # top right
            pentagon_patch = PolygonPatch(Polygon([P,A,B,C,D,P]), color = Acolor[i])
            ax.add_patch(pentagon_patch)
        # data and arrows
        ############frstrength = [[np.mean(friendship_average[i][j]) for j in range(self.NA)] for i in range(self.NA)] # average
        frstrength = [[friendship_average[i][j][-1] for j in range(self.NA)] for i in range(self.NA)] # final
        for i in range(self.NA):
            for j in range(self.NA):
                # arrows: from i to j
                color = 'white' if frstrength[i][j] == 0 else Acolor[i]
                width = 0.06*frstrength[i][j]
                head_width = 0.02 if 3*width < 0.02 else 3*width
                plt.arrow(coords[i][j][0][1], coords[i][j][0][0], coords[j][i][1][1]-coords[i][j][0][1], coords[j][i][1][0]-coords[i][j][0][0], \
                            color = color, width = width, head_width = head_width, length_includes_head=True)

        """     
        # time dependence
        for i in range(self.NA):
            for j in range(self.NA):
                if i != j:
                    ax.plot(timeline, self.friendship_average[i][j], color = Acolor[j])
                    ax.plot(timeline, self.friendship_average[i][j], color = Acolor[i], linestyle=':')
        legend = [Line2D([0], [0], color='black', lw=1, label="considers", linestyle=':'),
                Line2D([0], [0], color='red', lw=1, label="a friend")]
        ax.legend(handles=legend, loc='best', frameon=True, ncol=2)
        """
        plt.title(self.mode+' agent', fontsize = 50)
        fig.tight_layout()
        plt.axis('off')
        plt.savefig(name+'_stat_fr.png')
        plt.close() 
        if verbose: print("Plot in", name+'_stat_fr.png')

    def plot_Nfr_rep(self, name, dc): # - FONT: Check
        """number of friends versus reputation"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        IdataArray_Timefr = [[[self.friendships[r][i][j].count(1)/(timeline[-1]+1) for j in range(self.NA)] for i in range(self.NA)] for r in range(self.Nstat)]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([0,1])
        ax.set_xlim([0,self.NA-1])
        ax.set_xticks(np.arange(self.NA))
        for r in range(self.Nstat):
            for j in range(self.NA):
                reps = [np.mean(self.IdataArray[r][i][j]) for i in range(self.NA) if i != j]
                av_rep = np.mean(reps)
                ax.scatter(sum([IdataArray_Timefr[r][_][j] for _ in range(self.NA)])-1, av_rep, color=Acolor[j])
                #ax.scatter(sum([self.IdataArray_Timefr[r][_][j] for _ in range(self.NA)])-1, self.IdataArray_rep[r][j], color=Acolor[j]) 
        plt.xlabel('average number of friends', fontsize=20)
        plt.ylabel('reputation', fontsize=20)
        plt.title(self.mode+' among ordinary agents', fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_Nfr_rep.png')
        plt.close() 
        if verbose: print("Plot in", name+'_Nfr_rep.png')

    def plot_Fr_rep(self, name, dc): # - FONT: Check
        """friendships versus reputation"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        IdataArray_Timefr = [[[self.friendships[r][i][j].count(1)/(timeline[-1]+1) for j in range(self.NA)] for i in range(self.NA)] for r in range(self.Nstat)]
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        #ax.set_xticks(np.arange(self.NA))
        for r in range(self.Nstat):
            for i in range(self.NA):
                for j in range(self.NA):
                    if i != j:
                        plt.scatter(IdataArray_Timefr[r][i][j], np.mean(self.IdataArray[r][i][j]), color= Acolor[i], s=50)
                        plt.scatter(IdataArray_Timefr[r][i][j], np.mean(self.IdataArray[r][i][j]), color= Acolor[j], s=10)
                        #plt.scatter(r.Timefr[i][j], np.mean(r.Idata[i][j]), color= Acolor[i], s=50)
                        #plt.scatter(r.Timefr[i][j], np.mean(r.Idata[i][j]), color= Acolor[j], s=10)
        legend = [Line2D([0], [0], marker='o', color='w', label="'s reputation in the eyes of", markerfacecolor='black', markersize=5),
                Line2D([0], [0], marker='o', color='w', label='considers', markerfacecolor='r', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='r', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='a friend', markerfacecolor='black', markersize=5)]
        plt.legend(handles=legend, loc='best', frameon=True, ncol=2)
        plt.xlabel('average friendship time in %', fontsize=20)
        plt.ylabel('reputation of the inner', fontsize=20)
        plt.title(self.mode+' agent among '+str(self.NA-1)+' ordinary agents', fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_Fr_rep.png')
        plt.close() 
        if verbose: print("Plot in", name+'_Fr_rep.png')

    def plot_Fr_rep_c(self, name, dc): # - FONT: Check
        """contour plot via density(meshgrid)"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        IdataArray_Timefr = [[[self.friendships[r][i][j].count(1)/(timeline[-1]+1) for j in range(self.NA)] for i in range(self.NA)] for r in range(self.Nstat)]
        fig, ax = plt.subplots(figsize=(7,7))
        ax.set_ylim([0,100])
        ax.set_xlim([0,100])
        # gaussian parameters
        A = 0.1 # to be adjusted
        sigma = 0.07 # to be adjusted
        for i in range(self.NA):
            for j in range(self.NA):
                if i != j:
                    # grid
                    x, y = np.meshgrid(np.linspace(0,1,101), np.linspace(0,1,101))
                    z = 0*x+0*y 
                    for r in range(self.Nstat):
                        # add gaussians for each point in Fr_rep on the grid
                        z += A*np.exp(-((x-IdataArray_Timefr[r][i][j])**2+(y-np.mean(self.IdataArray[r][i][j]))**2)/(2*sigma**2))
                    plt.contour(z, levels = [0.5, 1], linewidths=1.5, colors = Acolor[i])
                    plt.contourf(z, levels = [0.5, 100], colors = Acolor[j], alpha = 0.05*Aalpha[j])
                    plt.contourf(z, levels = [1, 100], colors = Acolor[j], alpha = 0.1*Aalpha[j])
        # change ticks and ticklabels, because np.ogrid makes them go from 0 to 100 instead of 0 to 1
        ticks = [0,20,40,60,80,100]
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        dic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        #legend
        legend1 = [Ellipse((0,0), 40, 20, ec=None, fc='grey', alpha=0.5, label='$a$')]
        legend2 = [Ellipse((0,0), 40, 20, ec='black', fc='white', label='$b$')]
        leg1 = ax.legend(handles=legend1, loc=(0,0.04), frameon=True, framealpha = 0)   
        leg2 = ax.legend(handles=legend2, loc=(0,0), frameon=True, framealpha = 0) 
        ax.add_artist(leg1)
        plt.xlabel('fraction of time $a$ is regarded as friend by $b$', fontsize=20)
        plt.ylabel('reputation of $a$ with $b$', fontsize=20)
        plt.title(self.mode+' agent among '+str(self.NA-1)+' ordinary agents', fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_Fr_rep_c.png')
        plt.close() 
        if verbose: print("Plot in", name+'_Fr_rep_c.png')
    
    def plot_Fr_rep_nifty(self, name, dc): # - NEW ESTIMATOR: Check (lower contour) - FONT: Check
        """average reputation vs average friendship strength: contour + density plot"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_ylim([0,100])
        ax.set_xlim([0,100])
        densities = {}
        # calculate data and store in densities
        for i in range(self.NA):
            for j in range(self.NA):
                if i !=j:# and i==0 and j==2:

                    fr_data = [] # friendship mean per run
                    rep_data = [] # reputation mean per run
                    for r in range(self.Nstat):
                        if self.CONTINUOUS_FRIENDSHIP:
                            fr_data.append(np.mean(self.friendships[r][i][j]))
                        else:
                            #fr_data.append(self.friendships[r][i][j].count(1)/(self.friendships[r][i][j].count(1) + self.friendships[r][i][j].count(0)))
                            fr_data.append(self.friendships[r][i][j].count(1)/len(self.friendships[r][i][j]))
                        rep_data.append(np.mean(self.IdataArray[r][i][j]))
                    # reconstruction  
                    rct_path = get_folder(name)+'/'+get_name(name)+'_Fr_rep_nifty_rct/'+get_name(name)+'_Fr_rep_nifty_rct_'+str(i)+str(j)+'.json'
                    if os.path.exists(rct_path):
                        densities[str(i)+str(j)] = load_density(rct_path)
                        print('reloaded previous reconstruction')
                    else:
                        densities[str(i)+str(j)] = estimate_density(np.array(fr_data)*100, np.array(rep_data)*100, 101, 101)
                        save_density(densities[str(i)+str(j)], rct_path)        
        # plot
        for key in densities:
            alpha = densities[key]/np.amax(densities[key])
            cmap = mpl.colors.LinearSegmentedColormap.from_list('',[Acolor[int(key[1])], Acolor[int(key[1])]])
            ax.imshow(np.zeros(np.shape(alpha)), alpha = alpha, cmap = cmap)
            ax.contour(alpha, levels = [0.1], linewidths=1.5, colors = Acolor[int(key[0])])
        for i in range(self.NA):
            ax.plot([0,100], [100*self.x_est[i], 100*self.x_est[i]], color = Acolor[i], linewidth = 3, linestyle = '--', zorder=-1)
        
        # ticks
        ticks = [0,20,40,60,80,100]
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        dic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        #legend
        legend1 = [Ellipse((0,0), 40, 20, ec=None, fc='grey', alpha=0.5, label='$a$')]
        legend2 = [Ellipse((0,0), 40, 20, ec='black', fc= None, fill = False, label='$b$')]
        leg1 = ax.legend(handles=legend1, loc=(0,0.06), frameon=True, framealpha = 0, fontsize=22)   
        leg2 = ax.legend(handles=legend2, loc=(0,0), frameon=True, framealpha = 0, fontsize=22) 
        ax.add_artist(leg1)

        if self.CONTINUOUS_FRIENDSHIP:
            plt.xlabel('$\mathrm{average\ friendship\ strength\ of\ } b\mathrm{\ to\ }a$', fontsize = 30)
        else:
            plt.xlabel('$\mathrm{fraction\ of\ time\ } b\mathrm{\ regards\ } a\mathrm{\ a\ friend}$', fontsize = 30)
        plt.ylabel('$\mathrm{average\ reputation\ of\ } a\mathrm{\ with\ }b$', fontsize = 30)
        plt.title(make_latex_str(self.title), fontsize=30)
        plt.xticks(fontsize=18)#12
        plt.yticks(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_Fr_rep_nifty.png')
        plt.close() 
        if verbose: print("Plot in", name+'_Fr_rep_nifty.png')
    
    def plot_Fr_final_rep_nifty(self, name, dc): # - NEW ESTIMATOR: Check (lower contour) - FONT: Check
        """average reputation vs final friendship strength: contour + density plot"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_ylim([0,100])
        ax.set_xlim([0,100])
        densities = {}
        # calculate data and store in densities
        for i in range(self.NA):
            for j in range(self.NA):
                if i !=j:
                #if i==0 and j==2:
                    fr_data = [] # friendship mean per run
                    rep_data = [] # reputation mean per run
                    for r in range(self.Nstat):
                        fr_data.append(self.friendships[r][i][j][-1])
                        rep_data.append(np.mean(self.IdataArray[r][i][j]))
                    # reconstruction  
                    rct_path = get_folder(name)+'/'+get_name(name)+'_Fr(final)_rep_nifty_rct/'+get_name(name)+'_Fr(final)_rep_nifty_rct_'+str(i)+str(j)+'.json'
                    if os.path.exists(rct_path):
                        densities[str(i)+str(j)] = load_density(rct_path)
                        print('reloaded previous reconstruction')
                    else:
                        densities[str(i)+str(j)] = estimate_density(np.array(fr_data)*100, np.array(rep_data)*100, 101, 101)
                        save_density(densities[str(i)+str(j)], rct_path)

                    #densities[str(i)+str(j)] = estimate_density(np.array(fr_data)*100, np.array(rep_data)*100, 101, 101)
                    #ax.scatter(np.array(fr_data)*100, np.array(rep_data)*100, color = Acolor[i], s=50)
                    #ax.scatter(np.array(fr_data)*100, np.array(rep_data)*100, color = Acolor[j], s=10)
        
        # plot
        for key in densities:
            alpha = densities[key]/np.amax(densities[key])
            cmap = mpl.colors.LinearSegmentedColormap.from_list('',[Acolor[int(key[1])], Acolor[int(key[1])]])
            ax.imshow(np.zeros(np.shape(alpha)), alpha = alpha, cmap = cmap)
            ax.contour(alpha, levels = [0.1], linewidths=1.5, colors = Acolor[int(key[0])])
        for i in range(self.NA):
            ax.plot([0,100], [100*self.x_est[i], 100*self.x_est[i]], color = Acolor[i], linewidth = 3, linestyle = '--', zorder=-1)
        
        # ticks
        ticks = [0,20,40,60,80,100]
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        #dic = {0 : "0.0", 10 : "0.1", 20 : "0.2", 30 : "0.3", 40 : "0.4", 50 : "0.5", 60 : "0.6", 70 : "0.7", 80 : "0.8", 90 : "0.9", 100 : "1.0"}
        dic = {10*i: make_latex_str(f'0.{i}') for i in range(10)}
        dic[100] = make_latex_str('1.0')
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        #legend
        legend1 = [Ellipse((0,0), 40, 20, ec=None, fc='grey', alpha=0.5, label='$a$')]
        legend2 = [Ellipse((0,0), 40, 20, ec='black', fc= None, fill = False, label='$b$')]
        leg1 = ax.legend(handles=legend1, loc=(0,0.04), frameon=True, framealpha = 0, fontsize=15)   
        leg2 = ax.legend(handles=legend2, loc=(0,0), frameon=True, framealpha = 0, fontsize=15) 
        ax.add_artist(leg1)

        
        plt.xlabel('$\mathrm{fraction\ of\ time\ } a\mathrm{\ is\ regarded\ as\ friend\ by\ }b$', fontsize = 24)
        plt.ylabel('$\mathrm{reputation\ of\ } a\mathrm{\ with\ }b$', fontsize = 24)
        #plt.title(make_latex_str(self.mode+' agent among '+str(self.NA-1)+' ordinary agents'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize=24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        fig.tight_layout()
        plt.savefig(name+'_Fr(final)_rep_nifty.png')
        plt.close() 
        if verbose: print("Plot in", name+'_Fr(final)_rep_nifty.png')
    
    def plot_stat_rel(self, name, dc):
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        relationscomb = (Q*np.array(self.relationsc) + np.array(self.relationsm))/(Q+1)
        fig, ax = plt.subplots(figsize=(10, 6)) 
        #ax.set_ylim([0,1])
        ax.set_xlim([0, self.NA*self.NR*2])           
        for i in np.arange(self.NA):
            for j in np.arange(self.NA):
                if i != j:
                    if i>j:
                        # conversations (symmetric)
                        ax.plot(timeline, np.mean(self.relationsc, axis=0)[i][j][1:], color = Acolor[j], linewidth = 2)
                        ax.plot(timeline, np.mean(self.relationsc, axis=0)[i][j][1:], color = Acolor[i], linewidth = 2, dashes = [4,4])
                    # messages
                    ax.plot(timeline, np.mean(self.relationsm, axis=0)[i][j][1:], color = Acolor[j], linewidth = 1)
                    ax.plot(timeline, np.mean(self.relationsm, axis=0)[i][j][1:], color = Acolor[i], linewidth = 1, dashes = [1,4])
                    # combined
                    ax.plot(timeline, np.mean(relationscomb, axis=0)[i][j][1:], color = Acolor[j], alpha = 0.3) #where='post', 
                    ax.plot(timeline, np.mean(relationscomb, axis=0)[i][j][1:], color = Acolor[i], alpha = 0.3, dashes=[1,4]) #where='post',
                    #ax.fill_between(timeline, \
                    #                np.mean(relations, axis=0)[i][j][1:]-np.std(relations, axis=0)[i][j][1:], \
                    #                np.mean(relations, axis=0)[i][j][1:]+np.std(relations, axis=0)[i][j][1:], \
                    #                color = Acolor[j], alpha = 0.05*Aalpha[j]) #step = 'post'
                else:
                    pass
                    #ax.plot(timeline, np.mean(relations, axis=0)[i][i][1:], color = Acolor[i], linewidth = 3)
                    #ax.fill_between(timeline, np.mean(relations, axis=0)[i][i][1:]-np.std(relations, axis=0)[i][i][1:], \
                    #                np.mean(relations, axis=0)[i][i][1:]+np.std(relations, axis=0)[i][i][1:], \
                    #                color = Acolor[i], alpha = 0.15*Aalpha[i])
        plt.xlabel('$\mathrm{time}\ t$', fontsize = 20)
        plt.ylabel(r'$\mathrm{relation\ strength\ }r_{ab}$', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_stat_rel.png')
        plt.close() 
        if verbose: print("Plot in", name + '_stat_rel.png')

    def plot_rel_rep_scatter(self, name, dc):
        """average reputation vs final relation strength"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        relationscomb = (Q*np.array(self.relationsc) + np.array(self.relationsm))/(Q+1)
        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.set_ylim([0,1])
        #ax.set_xlim([0, self.NA*self.NR*2]) 
        for r in range(self.Nstat): 
            for i in range(self.NA):
                for j in range(self.NA):
                    if i != j:
                        plt.scatter(relationscomb[r][i][j][-1], np.mean(self.IdataArray[r][i][j]), color= Acolor[i], s=50)
                        plt.scatter(relationscomb[r][i][j][-1], np.mean(self.IdataArray[r][i][j]), color= Acolor[j], s=10)
        
        plt.xlabel(r'$\mathrm{relation\ strength\ }r_{ab}$', fontsize = 20)
        plt.ylabel(r'$\mathrm{reputation\ }\overline{x}_{ab}$', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_rel_rep.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rel_rep.png')

    def plot_relation_network(self, name, dc): # network: relations
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        relationscomb = (Q*np.array(self.relationsc) + np.array(self.relationsm))/(Q+1)
        fig, ax = plt.subplots(figsize=(10, 6)) 
        G = nx.Graph()
        weights = []
        edge_colors = []       
        for i in range(self.NA):
            G.add_node(i)
            for j in range(self.NA):
                if i != j and i<j: # count every connection once
                    weight_c = np.mean(self.relationsc, axis=0)[i][j][-1] # symmetric
                    weight_m = 0.5*(np.mean(self.relationsm, axis=0)[i][j][-1] + np.mean(self.relationsm, axis=0)[j][i][-1])
                    weight = (Q*weight_c+weight_m)/(Q + 1)
                    weights.append(weight)
                    asymetry = (np.mean(self.relationsm, axis=0)[i][j][-1] - np.mean(self.relationsm, axis=0)[j][i][-1])
                    if asymetry>0: # relation is mostly from i to j
                        edge_colors.append(self.Acolor[i])
                        color=self.Acolor[i]
                    else: # relation mostly from j to i
                        edge_colors.append(self.Acolor[j])
                        color=self.Acolor[j]
                    G.add_edge(i, j, weight=weight, color=color)
        label_dict = {}
        for node in G.nodes:
            label_dict[node]=str(node)
        # average reputation in the eye of all others:
        sizes = []
        for i in range(self.NA):
            reps = []
            for j in range(self.NA):
                if i != j:
                    rep_j_i = np.mean(np.mean(self.IdataArray, axis=0)[j][i])
                    reps.append(rep_j_i)
            sizes.append(1000*np.mean(reps))
        #sizes = [1000*np.mean(self.IdataArray, axis=0)[i] for i in range(self.NA)]
        colors = [self.Acolor[i] for i in range(self.NA)]            
        pos = nx.circular_layout(G)
        nx.drawing.nx_pylab.draw_networkx_nodes(G, pos=pos, node_size=sizes, node_color=colors)
        #nx.draw(G, pos=pos, node_size=sizes, node_color=colors, font_color='white', with_labels=True, font_weight='bold')
        nx.drawing.nx_pylab.draw_networkx_edges(G, pos=pos, width=np.array(weights)/100, edge_color=edge_colors)
        nx.drawing.nx_pylab.draw_networkx_labels(G, pos=pos, labels=label_dict, font_color='white', font_weight='bold')

        plt.axis('off')
        plt.title(self.title, fontsize=20)
        fig.tight_layout()
        plt.savefig(name+'_rel_net.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rel_net.png')

    def plot_rel_fr_scatter(self, name, dc): 
        """average friendship vs final relation strength"""
        Acolor = self.Acolor
        Aalpha = self.Aalpha
        timeline = np.arange(0, self.NA*self.NR*2)
        relationscomb = (Q*np.array(self.relationsc) + np.array(self.relationsm))/(Q+1)
        fig, ax = plt.subplots(figsize=(10, 6)) 
        ax.set_ylim([0,1])
        #ax.set_xlim([0, self.NA*self.NR*2]) 
        for r in range(self.Nstat): 
            for i in range(self.NA):
                for j in range(self.NA):
                    if i != j:
                        plt.scatter(relationscomb[r][i][j][-1], np.mean(self.friendships[r][i][j]), color= Acolor[i], s=50)
                        plt.scatter(relationscomb[r][i][j][-1], np.mean(self.friendships[r][i][j]), color= Acolor[j], s=10)
        
        plt.xlabel(r'$\mathrm{relation\ strength\ }r_{ab}$', fontsize = 20)
        plt.ylabel(r'$\mathrm{friendship\ strength\ }f_{ab}$', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_rel_fr.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rel_fr.png')

    def plot_informedness(self, name, dc): 
        """statistics of informednesses of agents at the end of the game"""
        Acolor = self.Acolor
        timeline = np.arange(0, self.NA*self.NR*2)
        # IdataArray[which rec][who thinks][about whom][time]
        informednesses = np.empty((self.NA, self.Nstat)) # [which agent][rec]
        informednesses.fill(np.nan)
        for rec in range(self.Nstat):
            for i in range(self.NA):
                informedness_i = sum([(self.IdataArray[rec][i][j][-1] - 0.5)*(self.x_est[j] - 0.5) for j in range(self.NA)])
                informednesses[i][rec] = informedness_i
        plt.figure()
        plt.hist(informednesses.T, color=Acolor, bins=20)
        plt.xlim([-0.1, 0.5])
        plt.ylim([0, 30])
        s = ''
        for i in range(self.NA):
            s += f'{self.Acolor[i]}:'.ljust(10) + f'{np.mean(informednesses[i]).round(2)} +-  {np.std(informednesses[i]).round(2)}\n'
        plt.text(-0.07, 23, s)
        plt.xlabel(make_latex_str('informedness'), fontsize = 16)
        plt.ylabel(make_latex_str('frequency'), fontsize = 16)
        plt.title(make_latex_str(self.mode+' among ordinary agents'), fontsize = 16)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(name+'_info.png')
        plt.close() 
        if verbose: print("Plot in", name+'_info.png')

    def plot_judgement_quality(self, name, dc):
        """statistics of the agents' judgement quality
        - measured by the distance of y_true and y_c [qualities1]
        - measured by the surprise when they get to know the real y: -ln(yc) or -ln(1-yc)  [qualities2]"""

        qualities1 = [[] for _ in range(self.NA)]  #[which agent (as a receiver)] -> list with all judgements 
        qualities2 = [[] for _ in range(self.NA)]  #[which agent (as a receiver)] -> list with all judgements 

        #self.ycArray = ycArray # [which rec][who assigned this yc][time when message was sent]
        #self.honestyArray = honestyArray # [which rec][who was honest][at what time]
        N_infinite_surprises = 0
        for rec in range(self.Nstat):
            honesty_rec = self.honestyArray[rec]
            ycs_rec = self.ycArray[rec]
            for t in range(len(self.ycArray[rec][0]))[1:]: # first element is always nan
                found_honesty = False # there must be only 1 speaker at a time
                found_yc = False # there must be only one receiver at a time
                for i in range(self.NA):
                    pass
                    # find agent that has spoken and the corresponding honesty value
                    if not np.isnan(honesty_rec[i][t]):
                        assert found_honesty == False, 'This plot has not been implemented for one-to-many conversations yet.'
                        honesty = honesty_rec[i][t]
                        found_honesty = True
                    # find the receiver and its judgement yc
                    if not np.isnan(ycs_rec[i][t]):
                        assert found_yc == False, 'This plot has not been implemented for one-to-many conversations yet.'
                        yc = ycs_rec[i][t]
                        receiver = i
                        found_yc = True
                
                quality1 = yc - honesty # 0 is good, +1 overrated, -1 underrated
                quality2 = -np.log(yc) if honesty==1 else -np.log(1-yc) # 0 is good, the higher the worse
                qualities1[receiver].append(quality1)
                if np.isfinite(quality2): 
                    N_infinite_surprises  += 1
                    qualities2[receiver].append(quality2) # 

        plt.figure()
        plt.hist(qualities1, color=self.Acolor, bins=20, log=True)
        plt.xlim([-1, 1])
        plt.ylim([100,3*10**4])
        #s = ''
        #for i in range(self.NA):
        #    s += f'{self.Acolor[i]}:'.ljust(10) + f'{np.mean(informednesses[i]).round(2)} +-  {np.std(informednesses[i]).round(2)}\n'
        #plt.text(min(informednesses.flatten()), 23, s)
        plt.xlabel('judgement accuracy\n underrated           correct                overrated', fontsize = 16)
        plt.ylabel(make_latex_str('frequency'), fontsize = 20)
        plt.title(make_latex_str(self.mode+' among ordinary agents'), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(name+'_judgement_quality1.png')
        plt.close() 
        if verbose: print("Plot in", name+'_judgement_quality1.png')

        plt.figure()
        plt.hist(qualities2, color=self.Acolor, bins=20, log=True)
        plt.xlim([0,45])
        plt.ylim([0.5, 100000])
        #s = ''
        #for i in range(self.NA):
        #    s += f'{self.Acolor[i]}:'.ljust(10) + f'{np.mean(informednesses[i]).round(2)} +-  {np.std(informednesses[i]).round(2)}\n'
        #plt.text(min(informednesses.flatten()), 23, s)
        plt.xlabel(make_latex_str('surprise'), fontsize = 24)
        plt.ylabel(make_latex_str('frequency'), fontsize = 24)
        plt.title(make_latex_str('reputation conserving compression'), fontsize = 24)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.savefig(name+'_judgement_quality2.png')
        plt.close() 
        if verbose: print("Plot in", name+'_judgement_quality2.png')

        print(f'{N_infinite_surprises} infinite surprises appeared in {self.Nstat} simulations.')

    def plot_EBEH_stat(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.Acolor
        Aalpha = self.Aalpha 
        ##EBEH_colors = {'bt': 'dimgray', 'bl': 'lightgray'}
        fig, ax = plt.subplots(figsize=(10, 6))
        lower_lim = -0.01
        upper_lim = 0.2
        ax.set_ylim([lower_lim, upper_lim])
        ax.set_xlim([0, self.NA*self.NR*2])
        ax.plot(timeline, [0.1]*len(timeline), color='black', linestyle='solid', lw=1)
        ax.plot(timeline, [0]*len(timeline), color='black', linestyle='solid', lw=1)

        # just for labeling:
        ax.plot(timeline, np.mean(eval(f'self.prob_blArray'), axis=0)[0], color = 'black', lw=3, linestyle='dashed', label='P(blush|dishonest)')
        ax.plot(timeline, np.mean(eval(f'self.prob_btArray'), axis=0)[0], color = 'black', lw=3, linestyle='solid', label='P(blush|honest)')

        for i in np.arange(self.NA):
            for param in ['bt', 'bl']:
                value = eval(f'self.prob_{param}Array')
                mean = np.mean(value, axis=0)
                std = np.std(value, axis=0)
                if param == 'bt':
                    bt = mean[i]
                    ls = 'solid'
                else:
                    bl = mean[i]
                    ls = 'dashed'
                ax.plot(timeline, mean[i], color = Acolor[i], ls=ls, lw=3) #where='post',
                #ax.fill_between(timeline, \
                #                np.array(mean[i])-np.array(std[i]), \
                #                np.array(mean[i])+np.array(std[i]), \
                #                color=Acolor[i], alpha=0.1) #step = 'post'  

            # mark BADE
            upper = bl - 0.1*bl
            lower = bt  + 0.1*bt
            crit_indices = list(np.where((upper-lower)<0)[0])
            lower_limits = [min(bl[_], bt[_])-0.0125 for _ in crit_indices]
            upper_limits = [max(bl[_], bt[_])+0.0125 for _ in crit_indices]
            ax.fill_between(crit_indices, upper_limits, lower_limits, color='yellow', alpha=0.8)           
            
        yticks = [0,0.05, 0.1, 0.15, 0.2]
        ax.set_yticks(yticks)
        Ticks = [_ for _ in timeline if _%2000==0]
        ax.set_xticks(Ticks)
        plt.xlabel(make_latex_str('time'), fontsize = 20)
        plt.ylabel(make_latex_str('perceived blushing frequencies'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(fontsize=18)
        fig.tight_layout()
        plt.savefig(name+'_EBEH.png')
        plt.close() 
        if verbose: print("Plot in", name + '_EBEH.png')

        if False: # write times with BADE to txt file
            for rec in range(self.Nstat):
                for i in range(self.NA):
                    bt = np.array(self.prob_btArray)[rec][i]
                    bl = np.array(self.prob_blArray)[rec][i]
                    upper = bl - 0.1*bl
                    lower = bt + 0.1*bt
                    crit_indices = list(np.where((upper-lower)<0)[0])
                    print(f'searching: agent {i}, run {rec}')
                    if len(crit_indices) > 0:
                        with open(name+'_critical_evidence.txt', 'a+') as crit_ev:
                            text = f'disregarded evidence: run {rec}; agent {i}; times {crit_indices}'
                            crit_ev.write(text + '\n')

    def plot_likelihood_ratio_stat(self, name, dc):
        timeline = np.arange(0, self.NA*self.NR*2+1)
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))
        #ax.set_xlim([0, self.NA*self.NR*2])
        ax.set_xlim([0,self.NA*self.NR*2*(1.01+(self.NA+self.NA)/100)])
        ax.plot(timeline, [1]*len(timeline), color='black', linestyle='dashed', lw=1)

        prob_bt = self.prob_btArray
        prob_bl = self.prob_blArray
        mean_bt = np.array(np.mean(prob_bt, axis=0))
        mean_bl = np.array(np.mean(prob_bl, axis=0))

        for i in np.arange(self.NA):
            R_blush = mean_bl[i]/mean_bt[i]
            R_noblush = (1-mean_bl[i])/(1-mean_bt[i])

            ax.plot(timeline, R_blush, color=Acolor[i], lw=2, ls='dashed') #where='post', 
            ax.plot(timeline, R_noblush, color = Acolor[i], lw=2, ls='dotted') #where='post',
            #ax.fill_between(timeline, \
            #                np.array(mean[i])-np.array(std[i]), \
            #                np.array(mean[i])+np.array(std[i]), \
            #                color=EBEH_colors[param], alpha=0.1) #step = 'post'    

            ax.errorbar(self.NA*self.NR*2*(1.01+i/100), np.mean(R_blush), np.std(R_blush),\
                        marker='_', markersize=9, color = Acolor[i], capsize = 5, capthick = 1.5)
            ax.errorbar(self.NA*self.NR*2*(1.01+(self.NA+i)/100), np.mean(R_noblush), np.std(R_noblush),\
                        marker='.', markersize=9, color = Acolor[i], capsize = 5, capthick = 1.5)  
            
        plt.xlabel(make_latex_str('time'), fontsize = 20)
        plt.ylabel(make_latex_str('likelihood ratio'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        fig.tight_layout()
        plt.savefig(name+'_LR.png')
        plt.close() 
        if verbose: print("Plot in", name + '_LR.png')

    def plot_likelihood_ratio_hist(self, name, dc):
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim([0.3, 150])
        ax.set_ylim([1, 5e6])

        LR_dict = {}

        for i in np.arange(self.NA):
            data_bt = [self.prob_btArray[rec][i] for rec in range(self.Nstat)]
            data_bt = np.array(data_bt).flatten()
            data_bl = [self.prob_blArray[rec][i] for rec in range(self.Nstat)]
            data_bl = np.array(data_bl).flatten()

            R_blush = data_bl/data_bt
            R_noblush = (1-data_bl)/(1-data_bt)

            if Acolor[i] == 'black':
                label_b = 'R(blush)'
                label_nob = 'R(no blush)'
            else:
                label_b = None
                label_nob = None

            logbins = np.geomspace(0.3, 150, 200)
            plt.hist(R_blush, bins=logbins, histtype='step', log=True, color = Acolor[i], linewidth = 2, ls='solid', label=label_b)  
            plt.hist(R_noblush, bins=logbins, histtype='step', log=True, color = Acolor[i], linewidth = 2, ls='dotted', label=label_nob)    
        
            LR_dict[i] = R_blush

        # save means for combined plot
        dc.collect_from_strategy(name, 'likelihoodRatio', LR_dict)  
        
        plt.xscale('log')
        plt.xlabel(make_latex_str('likelihood ratio R'), fontsize = 20)
        plt.ylabel(make_latex_str('frequency'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=16)
        fig.tight_layout()
        plt.savefig(name+'_LRhist.png')
        plt.close() 
        if verbose: print("Plot in", name + '_LRhist.png')

    def plot_BADE_aligned(self, name, dc):
        Acolor = self.Acolor
        time_before = 500
        time_after = 500 + 1 
        BADE_alignment_time = np.linspace(-time_before, time_after - 1, time_before + time_after) # 500 timesteps before BADE, 500 after
        right_half = np.linspace(0, time_after - 1, time_after)

        data_kappa = [[] for _ in range(self.NA)]
        data_ToM = [[] for _ in range(self.NA)]
        data_rel_red = [[] for _ in range(self.NA)]
        data_fr_red = [[] for _ in range(self.NA)]
        data_info = [[] for _ in range(self.NA)]
        data_rep0 = [[] for _ in range(self.NA)]

        # save recs, agents and times of BADE for comparison with non-BADE
        self.BADE_history = []

        for rec in range(self.Nstat):
            for i in np.arange(self.NA): 
                # find BADE
                bt = np.array(self.prob_btArray[rec][i])
                bl = np.array(self.prob_blArray[rec][i])
                upper = bl - 0.1*bl
                lower = bt  + 0.1*bt
                crit_indices = list(np.where((upper-lower)<0)[0])
                starts = [_ for _ in crit_indices if not _-1 in crit_indices]

                if len(starts) > 0:
                    self.BADE_history.append((rec, i, starts))

                # prepare data around BADE
                for start in starts:
                    obs_start = start - time_before
                    obs_end = start + time_after
                    buffer_start = []
                    buffer_end = []
                    if obs_start < 0:
                        buffer_start = [np.nan]*(0 - obs_start)
                        obs_start = 0
                    if obs_end > self.NA*self.NR*2 + 1:
                        buffer_end = [np.nan]*(obs_end - (self.NA*self.NR*2 + 1))
                        obs_end = self.NA*self.NR*2 + 1
                        
                    # kappa
                    kappa = buffer_start + list(self.kappaArray[rec][i][obs_start:obs_end]) + buffer_end
                    data_kappa[i].append(kappa)

                    # rep0                           
                    rep0 = buffer_start + list(self.IdataArray[rec][i][0][obs_start:obs_end]) + buffer_end
                    data_rep0[i].append(rep0)

                    # info
                    info_fulltime = np.nanmean([(np.array(self.IdataArray[rec][i][j]) - 0.5)*(self.x_est[j] - 0.5) for j in range(self.NA)], axis=0)
                    info = buffer_start + list(info_fulltime[obs_start:obs_end]) + buffer_end
                    data_info[i].append(info)

                    # ToM correctness
                    sum_ToM_correctness = np.zeros((time_before + time_after))
                    for j in range(self.NA):
                        if j != i:
                            for k in range(self.NA):
                                ToM_correctness = np.array(self.IothersArray[rec][i][j][k][obs_start:obs_end]) - np.array(self.IdataArray[rec][j][k][obs_start:obs_end])
                                sum_ToM_correctness += np.array(buffer_start + list(ToM_correctness) + buffer_end)
                    av_ToM_correctness = sum_ToM_correctness/(self.NA*(self.NA - 1))
                    data_ToM[i].append(list(av_ToM_correctness))

                    if i != 0:
                        # friendship to red
                        fr_red = buffer_start + list(self.friendships[rec][i][0][obs_start:obs_end]) + buffer_end
                        data_fr_red[i].append(fr_red)

                        # relation to red
                        rel_red = np.array(self.relationsc[rec][i][0][obs_start:obs_end]) / np.array([np.array(self.relationsc[rec][i][j][obs_start:obs_end]) for j in range(self.NA)]).sum(axis=0)
                        rel_red = buffer_start + list(rel_red) + buffer_end
                        data_rel_red[i].append(rel_red)

        

        # average over all runs
        plot_kappa = [np.nanmean(data_kappa[i], axis=0) for i in range(self.NA)]
        plot_ToM = [np.nanmean(data_ToM[i], axis=0) for i in range(self.NA)]
        plot_rel_red = [np.nanmean(data_rel_red[i], axis=0) for i in range(self.NA)]
        plot_fr_red = [np.nanmean(data_fr_red[i], axis=0) for i in range(self.NA)]
        plot_info = [np.nanmean(data_info[i], axis=0) for i in range(self.NA)]
        plot_rep0 = [np.nanmean(data_rep0[i], axis=0) for i in range(self.NA)]


        # fill red data with nan where there is no data for red
        plot_rel_red[0] = [np.nan]*(time_before + time_after)
        plot_fr_red[0] = [np.nan]*(time_before + time_after)

        # organize all plots
        all_plots = {'info': {'data': plot_info, 'loc': [0,0], 'y_label': 'informedness', 'limits': [-0.25, 0.25], 'center': 0, 'logy': False},
                    'rep0': {'data': plot_rep0, 'loc': [0,1], 'y_label': 'reputation of agent red', 'limits': [-0.05, 1.05], 'center': 0.5, 'logy': False},
                    'kappa': {'data': plot_kappa, 'loc': [1,0], 'y_label': 'surprise', 'limits': [0.01, 60000], 'center': 1, 'logy': True},
                    'ToM': {'data': plot_ToM, 'loc': [1,1], 'y_label': 'ToM accuracy', 'limits': [-0.35, 0.25], 'center': 0, 'logy': False},
                    'rel_red': {'data': plot_rel_red, 'loc': [2,0], 'y_label': 'conversations to agent red', 'limits': [0.35, 0.75], 'center': 0.5, 'logy': False},
                    'fr_red': {'data': plot_fr_red, 'loc': [2,1], 'y_label': 'friendship to agent red', 'limits': [-0.05, 1.05], 'center': 0.5, 'logy': False}
                    }

        plot_style = 'single' # single, combined

        if plot_style == 'combined':
            fig, axs = plt.subplots(3, 2, figsize=(10, 9))
            for plot, params in all_plots.items():
                for i in range(self.NA):
                    axs[params['loc'][0], params['loc'][1]].plot(BADE_alignment_time, params['data'][i], color=Acolor[i])
                axs[params['loc'][0], params['loc'][1]].set(xlabel='time', ylabel=params['y_label'])
                axs[params['loc'][0], params['loc'][1]].set_xlim([-time_before, time_after-1])
                axs[params['loc'][0], params['loc'][1]].set_ylim(params['limits'])
                if params['logy']:
                    axs[params['loc'][0], params['loc'][1]].set_yscale('log')
                # orientation
                axs[params['loc'][0], params['loc'][1]].plot(BADE_alignment_time, [params['center']]*len(BADE_alignment_time), color='lightgray', ls='dashed')
                axs[params['loc'][0], params['loc'][1]].fill_between(right_half, [params['limits'][0]]*len(right_half), [params['limits'][1]]*len(right_half), color='whitesmoke')

            fig.suptitle(make_latex_str(self.title), fontsize = 20)
            fig.tight_layout()
            plt.savefig(name+'_BADE_aligned.png', dpi=300)
            plt.close() 
            if verbose: print("Plot in", name + '_BADE_aligned.png')
        
        elif plot_style == 'single':       
            for plot, params in all_plots.items():
                print(plot, params['data'])
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(self.NA):
                    ax.plot(BADE_alignment_time, params['data'][i], color=Acolor[i])
                ax.set_xlim([-time_before, time_after-1])
                ax.set_ylim(params['limits'])
                if params['logy']:
                    ax.set_yscale('log')
                # orientation
                ax.plot(BADE_alignment_time, [params['center']]*len(BADE_alignment_time), color='lightgray', ls='dashed')
                ax.fill_between(right_half, [params['limits'][0]]*len(right_half), [params['limits'][1]]*len(right_half), color='whitesmoke')

                plt.title(make_latex_str(self.title), fontsize = 20)
                plt.xlabel(make_latex_str('time'), fontsize = 20)
                plt.ylabel(make_latex_str(params['y_label']), fontsize = 20)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                fig.tight_layout()
                plt.savefig(name+f'_BADE_aligned_{plot}.png', dpi=200)
                plt.close() 
                if verbose: print("Plot in", name + f'_BADE_aligned_{plot}.png')            
 

    def plot_notBADE_aligned(self, name, dc):
        Acolor = self.Acolor
        fig, axs = plt.subplots(3, 2, figsize=(10, 9))
        time_before = 500
        time_after = 500 + 1
        BADE_alignment_time = np.linspace(-time_before, time_after - 1, time_before + time_after ) # 500 timesteps before BADE, 500 after
        right_half = np.linspace(0, time_after - 1, time_after)

        data_kappa = [[] for _ in range(self.NA)]
        data_ToM = [[] for _ in range(self.NA)]
        data_rel_red = [[] for _ in range(self.NA)]
        data_fr_red = [[] for _ in range(self.NA)]
        data_info = [[] for _ in range(self.NA)]
        data_rep0 = [[] for _ in range(self.NA)]

        # get recs, agents and times of BADE for comparison with non-BADE
        assert hasattr(self, 'BADE_history'), 'BADE_history is not prepared. Please run plot_BADE_aligned BEFORE plot_notBADE_aligned!'

        # for each rec with BADE: find rec without BADE to compare with:
        recs_with_BADE = [_[0] for _ in self.BADE_history]
        recs_without_BADE = [_ for _ in range(self.Nstat) if not _ in recs_with_BADE]

        for idx in range(len(self.BADE_history)):
            rec = recs_without_BADE[idx]
            i = self.BADE_history[idx][1]
            starts = self.BADE_history[idx][2]

            # prepare data around BADE
            for start in starts:
                obs_start = start - time_before
                obs_end = start + time_after
                buffer_start = []
                buffer_end = []
                if obs_start < 0:
                    buffer_start = [np.nan]*(0 - obs_start)
                    obs_start = 0
                if obs_end > self.NA*self.NR*2 + 1:
                    buffer_end = [np.nan]*(obs_end - (self.NA*self.NR*2 + 1))
                    obs_end = self.NA*self.NR*2 + 1
                    
                # kappa
                kappa = buffer_start + list(self.kappaArray[rec][i][obs_start:obs_end]) + buffer_end
                data_kappa[i].append(list(kappa))

                # rep0                           
                rep0 = buffer_start + list(self.IdataArray[rec][i][0][obs_start:obs_end]) + buffer_end
                data_rep0[i].append(rep0)

                # info
                info_fulltime = np.nanmean([(np.array(self.IdataArray[rec][i][j]) - 0.5)*(self.x_est[j] - 0.5) for j in range(self.NA)], axis=0)
                info = buffer_start + list(info_fulltime[obs_start:obs_end]) + buffer_end
                data_info[i].append(info)

                # ToM correctness
                sum_ToM_correctness = np.zeros((obs_end - obs_start))
                for j in range(self.NA):
                    if j != i:
                        for k in range(self.NA):
                            ToM_correctness = np.array(self.IothersArray[rec][i][j][k][obs_start:obs_end]) - np.array(self.IdataArray[rec][j][k][obs_start:obs_end])
                            sum_ToM_correctness += ToM_correctness
                av_ToM_correctness = buffer_start + list(sum_ToM_correctness/(self.NA*(self.NA - 1))) + buffer_end
                data_ToM[i].append(list(av_ToM_correctness))

                if i != 0:
                    # friendship to red
                    fr_red = buffer_start + self.friendships[rec][i][0][obs_start:obs_end] + buffer_end
                    data_fr_red[i].append(list(fr_red))

                    # relation to red
                    rel_red = np.array(self.relationsc[rec][i][0][obs_start:obs_end]) / np.array([np.array(self.relationsc[rec][i][j][obs_start:obs_end]) for j in range(self.NA)]).sum(axis=0)
                    rel_red = buffer_start + list(rel_red) + buffer_end
                    data_rel_red[i].append(list(rel_red))

        # average over all runs
        # data[which agent][what time]
        plot_kappa = [np.nanmean(data_kappa[i], axis=0) for i in range(self.NA)]
        plot_ToM = [np.nanmean(data_ToM[i], axis=0) for i in range(self.NA)]
        plot_rel_red = [np.nanmean(data_rel_red[i], axis=0) for i in range(self.NA)]
        plot_fr_red = [np.nanmean(data_fr_red[i], axis=0) for i in range(self.NA)]
        plot_info = [np.nanmean(data_info[i], axis=0) for i in range(self.NA)]
        plot_rep0 = [np.nanmean(data_rep0[i], axis=0) for i in range(self.NA)]

        # fill red data with nan where there is no data for red
        plot_rel_red[0] = [np.nan]*(time_before + time_after)
        plot_fr_red[0] = [np.nan]*(time_before + time_after)

        # organize all plots
        all_plots = {'info': {'data': plot_info, 'loc': [0,0], 'y_label': 'informedness', 'limits': [-0.25, 0.25], 'center': 0, 'logy': False},
                    'rep0': {'data': plot_rep0, 'loc': [0,1], 'y_label': 'reputation of agent red', 'limits': [-0.05, 1.05], 'center': 0.5, 'logy': False},
                    'kappa': {'data': plot_kappa, 'loc': [1,0], 'y_label': 'surprise', 'limits': [0.01, 60000], 'center': 1, 'logy': True},
                    'ToM': {'data': plot_ToM, 'loc': [1,1], 'y_label': 'ToM accuracy', 'limits': [-0.35, 0.25], 'center': 0, 'logy': False},
                    'rel_red': {'data': plot_rel_red, 'loc': [2,0], 'y_label': 'conversations to agent red', 'limits': [0.35, 0.75], 'center': 0.5, 'logy': False},
                    'fr_red': {'data': plot_fr_red, 'loc': [2,1], 'y_label': 'friendship to agent red', 'limits': [-0.05, 1.05], 'center': 0.5, 'logy': False}
                    }

        combined_plots = False

        if combined_plots:
            fig, axs = plt.subplots(3, 2, figsize=(10, 9))
            for plot, params in all_plots.items():
                for i in range(self.NA):
                    axs[params['loc'][0], params['loc'][1]].plot(BADE_alignment_time, params['data'][i], color=Acolor[i])
                axs[params['loc'][0], params['loc'][1]].set(xlabel='time', ylabel=params['y_label'])
                axs[params['loc'][0], params['loc'][1]].set_xlim([-time_before, time_after-1])
                axs[params['loc'][0], params['loc'][1]].set_ylim(params['limits'])
                if params['logy']:
                    axs[params['loc'][0], params['loc'][1]].set_yscale('log')
                # orientation
                axs[params['loc'][0], params['loc'][1]].plot(BADE_alignment_time, [params['center']]*len(BADE_alignment_time), color='lightgray', ls='dashed')
                axs[params['loc'][0], params['loc'][1]].fill_between(right_half, [params['limits'][0]]*len(right_half), [params['limits'][1]]*len(right_half), color='whitesmoke')

            fig.suptitle(make_latex_str(self.title), fontsize = 20)
            fig.tight_layout()
            plt.savefig(name+'_notBADE_aligned.png', dpi=300)
            plt.close() 
            if verbose: print("Plot in", name + '_notBADE_aligned.png')
        
        else:       
            for plot, params in all_plots.items():
                fig, ax = plt.subplots(figsize=(10, 6))
                for i in range(self.NA):
                    ax.plot(BADE_alignment_time, params['data'][i], color=Acolor[i])
                ax.set_xlim([-time_before, time_after-1])
                ax.set_ylim(params['limits'])
                if params['logy']:
                    ax.set_yscale('log')
                # orientation
                ax.plot(BADE_alignment_time, [params['center']]*len(BADE_alignment_time), color='lightgray', ls='dashed')
                ax.fill_between(right_half, [params['limits'][0]]*len(right_half), [params['limits'][1]]*len(right_half), color='whitesmoke')

                plt.title(make_latex_str(self.title), fontsize = 20)
                plt.xlabel(make_latex_str('time'), fontsize = 20)
                plt.ylabel(make_latex_str(params['y_label']), fontsize = 20)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                fig.tight_layout()
                plt.savefig(name+f'_notBADE_aligned_{plot}.png', dpi=200)
                plt.close() 
                if verbose: print("Plot in", name + f'_notBADE_aligned_{plot}.png')

    def plot_BADEnotBADE_hist(self, name, dc):
        Acolor = self.Acolor

        def prepare_data(rec, i, start: int, end: int, NA, kappaArray, IdataArray, x_est, IothersArray, friendships, relationsc, ordinary_info_only=False):
            # include times before BADE to smoothen effects for short BADEs
            observation_time_before_BADE = 500
            start = max(0,start - observation_time_before_BADE)

            # kappa
            kappa = [np.nanmean(kappaArray[rec][i][start:end])]
            # info
            if ordinary_info_only:
                info_fulltime = np.nanmean([(np.array(IdataArray[rec][i][j]) - 0.5)*(x_est[j] - 0.5) for j in range(NA) if j != 0], axis=0)
            else:
                info_fulltime = np.nanmean([(np.array(IdataArray[rec][i][j]) - 0.5)*(x_est[j] - 0.5) for j in range(NA)], axis=0)
            

            info = [np.nanmean(info_fulltime[start:end])]
            # ToM correctness
            all_ToM = []
            for j in range(NA):
                if j != i:
                    for k in range(NA):
                        ToM_correctness = list(np.array(IothersArray[rec][i][j][k][start:end]) - np.array(IdataArray[rec][j][k][start:end]))
                        all_ToM += ToM_correctness
            ToM = [np.nanmean(all_ToM)]
            if i != 0:
                # rep0                           
                rep0 = [np.nanmean(IdataArray[rec][i][0][start:end])]
                # friendship to red
                fr_red = [np.nanmean(friendships[rec][i][0][start:end])]
                # relation to red
                rel_red = np.array(relationsc[rec][i][0][start:end]) / np.array([np.array(relationsc[rec][i][j][start:end]) for j in range(NA)]).sum(axis=0)
                rel_red = [np.nanmean(rel_red)]
            else: 
                fr_red = []
                rel_red = []
                rep0 = []

            return {'info':info, 'rep0':rep0, 'kappa':kappa, 'ToM':ToM, 'fr_red':fr_red, 'rel_red':rel_red}

        # save recs, agents and times of BADE for comparison with non-BADE
        local_BADE_history = []

        BADE_kappa = [[] for _ in range(self.NA)]
        BADE_ToM = [[] for _ in range(self.NA)]
        BADE_rel_red = [[] for _ in range(self.NA)]
        BADE_fr_red = [[] for _ in range(self.NA)]
        BADE_info = [[] for _ in range(self.NA)]
        BADE_info_ordinary_only = [[] for _ in range(self.NA)] # what ordinaries think about other ordinaries
        BADE_rep0 = [[] for _ in range(self.NA)]
        notBADE_kappa = [[] for _ in range(self.NA)]
        notBADE_ToM = [[] for _ in range(self.NA)]
        notBADE_rel_red = [[] for _ in range(self.NA)]
        notBADE_fr_red = [[] for _ in range(self.NA)]
        notBADE_info = [[] for _ in range(self.NA)]
        notBADE_info_ordinary_only = [[] for _ in range(self.NA)]
        notBADE_rep0 = [[] for _ in range(self.NA)]

        # BADE
        for rec in range(self.Nstat):
            for i in np.arange(self.NA):
                # find BADE
                bt = np.array(self.prob_btArray[rec][i])
                bl = np.array(self.prob_blArray[rec][i])
                upper = bl - 0.1*bl
                lower = bt  + 0.1*bt
                crit_indices = list(np.where((upper-lower)<0)[0])
                starts = [_ for _ in crit_indices if not _-1 in crit_indices]
                ends = [_ for _ in crit_indices if not _+1 in crit_indices]

                if len(crit_indices) > 0:
                    local_BADE_history.append((rec, i, starts, ends))

                for idx in range(len(starts)):
                    if i != 0:
                        BADE_data_ordinary_only = prepare_data(rec, i, starts[idx], ends[idx], self.NA, self.kappaArray, self.IdataArray, self.x_est, self.IothersArray, self.friendships, self.relationsc, ordinary_info_only=True)
                        BADE_info_ordinary_only[i] += BADE_data_ordinary_only['info']

                    BADE_data = prepare_data(rec, i, starts[idx], ends[idx], self.NA, self.kappaArray, self.IdataArray, self.x_est, self.IothersArray, self.friendships, self.relationsc)
                    BADE_kappa[i] += BADE_data['kappa']
                    BADE_ToM[i] += BADE_data['ToM']
                    BADE_rel_red[i] += BADE_data['rel_red']
                    BADE_fr_red[i] += BADE_data['fr_red']
                    BADE_info[i] += BADE_data['info']
                    BADE_rep0[i] += BADE_data['rep0']

        # notBADE
        recs_with_BADE = [_[0] for _ in local_BADE_history]
        recs_without_BADE = [_ for _ in range(self.Nstat) if not _ in recs_with_BADE]

        for idx in range(len(recs_without_BADE)):
            idx_rescaled = idx % len(recs_with_BADE)
            rec = recs_without_BADE[idx]
            i = local_BADE_history[idx_rescaled][1]
            starts = local_BADE_history[idx_rescaled][2]
            ends = local_BADE_history[idx_rescaled][3]

            for _ in range(len(starts)):
                notBADE_data = prepare_data(rec, i, starts[_], ends[_], self.NA, self.kappaArray, self.IdataArray, self.x_est, self.IothersArray, self.friendships, self.relationsc)
                notBADE_kappa[i] += notBADE_data['kappa']
                notBADE_ToM[i] += notBADE_data['ToM']
                notBADE_rel_red[i] += notBADE_data['rel_red']
                notBADE_fr_red[i] += notBADE_data['fr_red']
                notBADE_info[i] += notBADE_data['info']
                notBADE_rep0[i] += notBADE_data['rep0']     
                if i != 0:
                    notBADE_data_ordinary_only = prepare_data(rec, i, starts[_], ends[_], self.NA, self.kappaArray, self.IdataArray, self.x_est, self.IothersArray, self.friendships, self.relationsc, ordinary_info_only=True)
                    notBADE_info_ordinary_only[i] += notBADE_data_ordinary_only['info']

        # plotting
        Nbins = 20
        #rgb_colors = [(1,0,0,0.5), (0,1,1,0.5), (0,0,0,0.5)]
        rgb_colors = [(169/255,169/255,169/255), (255/255,165/255,0)]

        # organize all plots
        all_plots = {'info': {'data': [BADE_info, notBADE_info], 'label': 'informedness', 'xlimits': [-0.2, 0.2], 'ylimits': [0.8, 1500], 'center': 0, 'log': False},
                    'info_ordinary_only': {'data': [BADE_info_ordinary_only, notBADE_info_ordinary_only], 'label': 'informedness', 'xlimits': [-0.2, 0.2], 'ylimits': [0.8, 1500], 'center': 0, 'log': False},
                    'rep0': {'data': [BADE_rep0, notBADE_rep0], 'label': 'reputation of agent red', 'xlimits': [0, 1], 'ylimits': [0.8, 1e3], 'center': 0.5, 'log': False}, # ylimits [0.01, 20]
                    'kappa': {'data': [BADE_kappa, notBADE_kappa], 'label': 'surprise', 'xlimits': [1e-6, 1e7], 'ylimits': [0.8, 300], 'center': 1, 'log': True},
                    'ToM': {'data': [BADE_ToM, notBADE_ToM], 'label': 'ToM accuracy', 'xlimits': [-0.5, 0.6], 'ylimits': [0.8, 300], 'center': 0, 'log': False},
                    'rel_red': {'data': [BADE_rel_red, notBADE_rel_red], 'label': 'conversations to agent red', 'xlimits': [0.25, 0.75], 'ylimits': [0.8, 200], 'center': 0.5, 'log': False},
                    'fr_red': {'data': [BADE_fr_red, notBADE_fr_red], 'label': 'friendship to agent red', 'xlimits': [0, 1], 'ylimits': [0.8, 500], 'center': 0.5, 'log': False}
                    }
        
        # save means for combined plot
        means_dict = {}
        for k,v in all_plots.items():
            BADE = [np.nanmean(v['data'][0][_]) for _ in range(self.NA)]
            notBADE = [np.nanmean(v['data'][1][_]) for _ in range(self.NA)]
            means_dict[k] = [BADE, notBADE]
        dc.collect_from_strategy(name, 'BADEnotBADE_means', means_dict)
        
        
        for plot, params in all_plots.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim(params['xlimits'])
            ax.set_ylim(params['ylimits'])

            if params['log']:
                bins = 10 ** np.linspace(np.log10(params['xlimits'][0]), np.log10(params['xlimits'][1]), Nbins)
            else:
                bins = np.linspace(params['xlimits'][0], params['xlimits'][1], Nbins)

            # separate between special and ordinary agents
            special_agents = [0]
            ordinary_agents = [i for i in range(self.NA) if i not in special_agents]

            # no BADE
            ordinary_noBADE_combined = [list(params['data'][1][i]) for i in ordinary_agents]
            ordinary_noBADE_combined = [item for sublist in ordinary_noBADE_combined for item in sublist]
            plt.hist(ordinary_noBADE_combined, bins=bins, log=True, density=False, fc=rgb_colors[0], zorder=1, alpha = 0.4)
            plt.hist(ordinary_noBADE_combined, bins=bins, log=True, density=False, histtype='step', color = rgb_colors[0], lw=3, zorder=2)
            # BADE
            ordinary_BADE_combined = [list(params['data'][0][i]) for i in ordinary_agents]
            ordinary_BADE_combined = [item for sublist in ordinary_BADE_combined for item in sublist]
            plt.hist(ordinary_BADE_combined, bins=bins, log=True, density=False, fc=rgb_colors[1], zorder=10, alpha = 0.4)
            plt.hist(ordinary_BADE_combined, bins=bins, log=True, density=False, histtype='step', color = rgb_colors[1], lw=3, zorder=11)

            #for i in ordinary_agents:
            #    # BADE
            #    plt.hist(params['data'][0][i], bins=bins, log=True, density=True, fc=rgb_colors[i])
            #    plt.hist(params['data'][0][i], bins=bins, log=True, density=True, histtype='step', color = Acolor[i], lw=3)
            #    # not BADE
            #    plt.hist(params['data'][1][i], bins=bins, log=True, density=True, histtype='step', color = Acolor[i], lw=3, ls='dashed')

            if params['log']:
                ax.set_xscale('log')
            
            plt.title(make_latex_str(self.title), fontsize = 24)
            plt.xlabel(make_latex_str(params['label']), fontsize = 20)
            plt.ylabel(make_latex_str('frequency'), fontsize = 20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            fig.tight_layout()
            plt.savefig(name+f'_BADEnotBADE_hist_{plot}.png', dpi=200)
            plt.close() 
            if verbose: print("Plot in", name + f'_BADEnotBADE_hist_{plot}.png')    


    def plot_LR_rep0(self, name, dc):
        # corr: for every timestep in every run
        # R_red/black/cyan vs rep_red
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim([0.3, 150])
        ax.set_ylim([0, 1])

        # red's reputation:
        sum_opinion = np.zeros((self.Nstat*(self.NA*self.NR*2+1)))
        for i in range(1, self.NA):
            opinion = np.array([self.IdataArray[rec][i][0] for rec in range(self.Nstat)]).flatten()
            sum_opinion += opinion              
        rep0 = sum_opinion/(self.NA -1)

        corrs = []

        for i in np.arange(self.NA-1, -1, -1):
            data_bt = [self.prob_btArray[rec][i] for rec in range(self.Nstat)]
            data_bt = np.array(data_bt).flatten()
            data_bl = [self.prob_blArray[rec][i] for rec in range(self.Nstat)]
            data_bl = np.array(data_bl).flatten()

            R_blush = data_bl/data_bt
            R_noblush = (1-data_bl)/(1-data_bt)

            if Acolor[i] == 'black':
                label_b = 'R(blush)'
                label_nob = 'R(no blush)'
            else:
                label_b = None
                label_nob = None

            corr = pd.DataFrame({'R': R_blush, 'rep0': rep0}).corr().iloc[0][1].round(3)
            corrs.append(corr)

            ax.scatter(R_blush, rep0, color=Acolor[i], alpha=0.05, s=0.1, marker='.', label=label_b)
            #ax.scatter(R_noblush, rep0, color=Acolor[i], marker='+', label=label_nob)
            
        ax.set_xscale('log')  
        text = f'correlations: \nred: {corrs[2]} \ncyan: {corrs[1]} \nblack: {corrs[0]}' 
        plt.text(0.02, 0.04, text, va='bottom', ha='left', transform=ax.transAxes, fontsize=16)
        plt.xlabel(make_latex_str('likelihood ratio R(blush)'), fontsize = 20)
        plt.ylabel(make_latex_str("agent red's reputation"), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_LR_rep0.png', dpi=200)
        plt.close() 
        if verbose: print("Plot in", name + '_LR_rep0.png')

    def plot_LR_rep0_individual(self, name, dc):
        # corr: for every timestep in every run
        # R_red/black/cyan vs rep_red
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))
        #ax.set_xlim([0.3, 150])
        #ax.set_ylim([0, 1])

        corrs = []

        for i in np.arange(self.NA-1, -1, -1):
            data_bt = [self.prob_btArray[rec][i] for rec in range(self.Nstat)]
            data_bt = np.array(data_bt).flatten()
            data_bl = [self.prob_blArray[rec][i] for rec in range(self.Nstat)]
            data_bl = np.array(data_bl).flatten()

            rep0 = [self.IdataArray[rec][i][0] for rec in range(self.Nstat)]
            rep0 = np.array(rep0).flatten()

            R_blush = data_bl/data_bt
            R_noblush = (1-data_bl)/(1-data_bt)

            if Acolor[i] == 'black':
                label_b = 'R(blush)'
                label_nob = 'R(no blush)'
            else:
                label_b = None
                label_nob = None

            corr = pd.DataFrame({'R': R_blush, 'rep0': rep0}).corr().iloc[0][1].round(3)
            corrs.append(corr)

            ax.scatter(R_blush, rep0, color=Acolor[i], alpha=0.05, s=0.1, marker='.', label=label_b)
            #ax.scatter(R_noblush, rep0, color=Acolor[i], marker='+', label=label_nob)
            
        ax.set_xscale('log')   
        text = f'correlations: \nred: {corrs[2]} \ncyan: {corrs[1]} \nblack: {corrs[0]}' 
        plt.text(0.02, 0.04, text, va='bottom', ha='left', transform=ax.transAxes, fontsize=16) 
        plt.xlabel(make_latex_str('likelihood ratio R(blush)'), fontsize = 20)
        plt.ylabel(make_latex_str("agent red's reputation"), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_LR_rep0_individual.png', dpi=200)
        plt.close() 
        if verbose: print("Plot in", name + '_LR_rep0_individual.png')

    def plot_LR_info(self, name, dc):
        # corr: for every timestep in every run
        # R vs informedness for every agent
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim([0.3, 150])
        ax.set_ylim([-0.25, 0.25])

        corrs = []

        for i in np.arange(self.NA-1, -1, -1):
            # informedness
            info = [np.mean([(np.array(self.IdataArray[rec][i][j]) - 0.5)*(self.x_est[j] - 0.5) for j in range(self.NA)], axis=0) for rec in range(self.Nstat)]
            info = np.array(info).flatten()

            # LR
            data_bt = [self.prob_btArray[rec][i] for rec in range(self.Nstat)]
            data_bt = np.array(data_bt).flatten()
            data_bl = [self.prob_blArray[rec][i] for rec in range(self.Nstat)]
            data_bl = np.array(data_bl).flatten()

            R_blush = data_bl/data_bt
            R_noblush = (1-data_bl)/(1-data_bt)

            if Acolor[i] == 'black':
                label_b = 'R(blush)'
                label_nob = 'R(no blush)'
            else:
                label_b = None
                label_nob = None

            corr = pd.DataFrame({'R': R_blush, 'info': info}).corr().iloc[0][1].round(3)
            corrs.append(corr)

            # save data for causality detection
            if i == 1 and 'destructive' in name:
                with open('causality_data_cyan.txt', 'w+') as f:
                    R_blush_no_nan = np.array([0.25 if np.isnan(_) else _ for _ in R_blush])
                    info_no_nan = np.array([0 if np.isnan(_) else _ for _ in info])
                    np.savetxt('R_blush.txt', R_blush_no_nan, delimiter=',')
                    np.savetxt('info.txt', info_no_nan, delimiter=',')
                    #f.write(str(list(R_blush_no_nan)))
                    #f.write('\n')
                    #f.write(str(list(info_no_nan)))

            ax.scatter(R_blush, info, color=Acolor[i], alpha=0.05, s=0.1, marker='.', label=label_b)
            #ax.scatter(R_noblush, info, color=Acolor[i], marker='+', label=label_nob)

        ax.set_xscale('log')   
        text = f'correlations: \nred: {corrs[2]} \ncyan: {corrs[1]} \nblack: {corrs[0]}' 
        plt.text(0.02, 0.04, text, va='bottom', ha='left', transform=ax.transAxes, fontsize=16)  
        plt.xlabel(make_latex_str('likelihood ratio R(blush)'), fontsize = 20)
        plt.ylabel(make_latex_str("informedness"), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_LR_info.png', dpi=200)
        plt.close() 
        if verbose: print("Plot in", name + '_LR_info.png')

    def plot_EBEH_hist(self, name, dc):
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim([0, 0.27])
        ax.set_ylim([1, 6e5])

        # labeling
        data_bt = [self.prob_btArray[rec][0] for rec in range(self.Nstat)]
        data_bt = np.array(data_bt).flatten()
        data_bl = [self.prob_blArray[rec][0] for rec in range(self.Nstat)]
        data_bl = np.array(data_bl).flatten()
        plt.hist(data_bl, bins=100, histtype='step', log=True, color = 'lightgray', linewidth = 2, ls='solid', label='P(blush|dishonest)') 
        plt.hist(data_bt, bins=100, histtype='step', log=True, color = 'dimgray', linewidth = 2, ls='solid', label='P(blush|honest)')
        

        for i in np.arange(self.NA):
            data_bt = [self.prob_btArray[rec][i] for rec in range(self.Nstat)]
            data_bt = np.array(data_bt).flatten()
            data_bl = [self.prob_blArray[rec][i] for rec in range(self.Nstat)]
            data_bl = np.array(data_bl).flatten()


            #logbins = np.geomspace(0.3, 150, 200)
            plt.hist(data_bl, bins=100, histtype='step', log=True, color = 'lightgray', linewidth = 2, ls='solid') 
            plt.hist(data_bl, bins=100, histtype='step', log=True, color = Acolor[i], linewidth = 2, ls=(0,(1,10)))    
            plt.hist(data_bt, bins=100, histtype='step', log=True, color = 'dimgray', linewidth = 2, ls='solid')  
            plt.hist(data_bt, bins=100, histtype='step', log=True, color = Acolor[i], linewidth = 2, ls=(0,(1,10)))
            plt.vlines(0.1, 1, 6e5, colors='lightgray', linestyles='dashed', lw=3)

        #plt.xscale('log')
        plt.xlabel(make_latex_str('perceived blushing frequency'), fontsize = 20)
        plt.ylabel(make_latex_str('frequency'), fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=16)
        fig.tight_layout()
        plt.savefig(name+'_EBEHhist.png')
        plt.close() 
        if verbose: print("Plot in", name + '_EBEHhist.png')

    def plot_info_ToM(self, name, dc):
        """the agents' informedness vs the correctness of their ToM
        - in general
        - per agent"""

        # general
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6)) 

        for i in np.arange(self.NA)[::-1]:
            # informedness (general info  in every simulation of agent i)
            info = [np.nanmean(np.mean([(np.array(self.IdataArray[rec][i][j]) - 0.5)*(self.x_est[j] - 0.5) for j in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
            info = np.array(info).flatten() 
            # ToM (general correctess of ToM in every simulation of agent i -> same formula but comparing x_abc with x_bc)
            ToM = [np.nanmean(np.mean([np.mean([(np.array(self.IothersArray[rec][i][j][k]) - 0.5)*(np.array(self.IdataArray[rec][j][k]) - 0.5) for j in range(self.NA) if i!=j], axis=0) for k in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
            ToM = np.array(ToM).flatten()

            corr = round(np.corrcoef(info, ToM)[0][1], 2)
            label = f'correlation: {corr}'
            if 'clever' in name:
                info_high_info = [_ for _ in info if _>0.07]
                ToM_high_info = [ToM[_] for _ in range(len(ToM)) if info[_]>0.07]
                corr_high_info = round(np.corrcoef(info_high_info, ToM_high_info)[0][1], 2)
                label = f'correlation: {corr} ({corr_high_info})'

            plt.scatter(info, ToM, color=Acolor[i], s=30, alpha=0.3, label=label)

        plt.xlim([-0.16, 0.2])
        plt.ylim([-0.05, 0.25])
        plt.xlabel('informedness', fontsize = 20)
        plt.ylabel('ToM correctness', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=16)
        fig.tight_layout()
        plt.savefig(name+'_info_ToM.png')
        plt.close() 
        if verbose: print("Plot in", name + '_info_ToM.png')

        # per agent
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6)) 

        for i in np.arange(self.NA):
            for j in np.arange(self.NA):
                if i != j:

                    # informedness (info of agent i about agent j in every simulation )
                    info = [np.nanmean((np.array(self.IdataArray[rec][i][j]) - 0.5)*(self.x_est[j] - 0.5)) for rec in range(self.Nstat)]
                    info = np.array(info).flatten() 
                    # ToM (general correctess of ToM  in every simulation of agent i -> same formula but comparing x_abc with x_bc)
                    ToM = [np.nanmean(np.mean([(np.array(self.IothersArray[rec][i][j][k]) - 0.5)*(np.array(self.IdataArray[rec][j][k]) - 0.5) for k in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
                    ToM = np.array(ToM).flatten()

                    plt.scatter(info, ToM, color=Acolor[i], s=30, alpha=0.3)
                    plt.scatter(info, ToM, color=Acolor[j], s=5, alpha=0.3)
        
        
        plt.xlim([-0.27, 0.27])
        plt.ylim([-0.11, 0.25])
        plt.xlabel('informedness', fontsize = 20)
        plt.ylabel('ToM correctness', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_info_ToM_perAgent.png')
        plt.close() 
        if verbose: print("Plot in", name + '_info_ToM_perAgent.png')

    def plot_ToM_rep(self, name, dc):
        """correctness of the agents' ToM vs their reputation
        - general
        - per agent (ToM of i about j vs x_ji )"""

        # general
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6)) 

        for i in np.arange(self.NA)[::-1] :
            # ToM (general correctess of ToM  in every simulation of agent i -> same formula but comparing x_abc with x_bc)
            ToM = [np.nanmean(np.mean([np.mean([(np.array(self.IothersArray[rec][i][j][k]) - 0.5)*(np.array(self.IdataArray[rec][j][k]) - 0.5) for j in range(self.NA) if i!=j], axis=0) for k in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
            ToM = np.array(ToM).flatten()
            # reputation of i
            rep = [np.nanmean(np.mean([self.IdataArray[rec][j][i] for j in range(self.NA) if j!=i], axis=0)) for rec in range(self.Nstat)]
            rep = np.array(rep).flatten()

            corr = round(np.corrcoef(ToM, rep)[0][1], 2)

            # correlation for values where rep>0.5
            rep_high = [_  for _ in rep if _>=0.5 ]
            ToM_high = [ToM[_] for _ in range(len(ToM)) if rep[_]>=0.5]
            corr_high = round(np.corrcoef(ToM_high, rep_high)[0][1], 2)

            plt.scatter(ToM, rep, color=Acolor[i], s=30, alpha=0.3, label=f'correlation: {corr} ({corr_high})')
        
        plt.xlim([-0.05, 0.25])
        plt.ylim([0,1])
        plt.xlabel('ToM correctness', fontsize = 20)
        plt.ylabel('reputation', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=16)
        fig.tight_layout()
        plt.savefig(name+'_ToM_rep.png')
        plt.close() 
        if verbose: print("Plot in", name + '_ToM_rep.png')

        # general swapped
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6)) 

        for i in np.arange(self.NA)[::-1] :
            # ToM (general correctess of ToM  in every simulation of agent i -> same formula but comparing x_abc with x_bc)
            ToM = [np.nanmean(np.mean([np.mean([(np.array(self.IothersArray[rec][i][j][k]) - 0.5)*(np.array(self.IdataArray[rec][j][k]) - 0.5) for j in range(self.NA) if i!=j], axis=0) for k in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
            ToM = np.array(ToM).flatten()
            # reputation of i
            rep = [np.nanmean(np.mean([self.IdataArray[rec][j][i] for j in range(self.NA) if j!=i], axis=0)) for rec in range(self.Nstat)]
            rep = np.array(rep).flatten()

            corr = round(np.corrcoef(ToM, rep)[0][1], 2)

            plt.scatter(rep, ToM, color=Acolor[i], s=30, alpha=0.3, label=f'correlation: {corr}')
        
        plt.ylim([-0.05, 0.25])
        plt.xlim([0,1])
        plt.ylabel('ToM correctness', fontsize = 20)
        plt.xlabel('reputation', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_rep_ToM.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rep_ToM.png')

        # per agent
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6)) 

        for i in np.arange(self.NA)[::-1] :
            for j in np.arange(self.NA)[::-1] :
                if i != j:
                    # ToM ( correctess of ToM  in every simulation of agent i -> same formula but comparing x_abc with x_bc)
                    ToM = [np.nanmean(np.mean([(np.array(self.IothersArray[rec][i][j][k]) - 0.5)*(np.array(self.IdataArray[rec][j][k]) - 0.5) for k in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
                    ToM = np.array(ToM).flatten()
                    # reputation of i as seen from j
                    rep = [np.nanmean(self.IdataArray[rec][j][i]) for rec in range(self.Nstat)]
                    rep = np.array(rep).flatten()

                    plt.scatter(ToM, rep, color=Acolor[i], s=30, alpha=0.3)
                    plt.scatter(ToM, rep, color=Acolor[j], s=5, alpha=0.3)
        
        plt.xlim([-0.10, 0.25])
        plt.ylim([0,1])
        plt.xlabel('ToM correctness', fontsize = 20)
        plt.ylabel('reputation', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_ToM_rep_perAgent.png')
        plt.close() 
        if verbose: print("Plot in", name + '_ToM_rep_perAgent.png')

        # per agent swapped
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6)) 

        for i in np.arange(self.NA)[::-1] :
            for j in np.arange(self.NA)[::-1] :
                if i != j:
                    # ToM ( correctess of ToM  in every simulation of agent i -> same formula but comparing x_abc with x_bc)
                    ToM = [np.nanmean(np.mean([(np.array(self.IothersArray[rec][i][j][k]) - 0.5)*(np.array(self.IdataArray[rec][j][k]) - 0.5) for k in range(self.NA)], axis=0)) for rec in range(self.Nstat)]
                    ToM = np.array(ToM).flatten()
                    # reputation of i as seen from j
                    rep = [np.nanmean(self.IdataArray[rec][j][i]) for rec in range(self.Nstat)]
                    rep = np.array(rep).flatten()

                    plt.scatter(rep, ToM, color=Acolor[i], s=30, alpha=0.3)
                    plt.scatter(rep, ToM, color=Acolor[j], s=5, alpha=0.3)
        
        plt.ylim([-0.10, 0.25])
        plt.xlim([0,1])
        plt.ylabel('ToM correctness', fontsize = 20)
        plt.xlabel('reputation', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+'_rep_ToM_perAgent.png')
        plt.close() 
        if verbose: print("Plot in", name + '_rep_ToM_perAgent.png')

    def plot_rep_LieAcceptance(self, name, dc):
        """average reputation of an agent vs that agent's average acceptance-rate, i.e. yc"""

        # general
        Acolor = self.Acolor
        fig, ax = plt.subplots(figsize=(10, 6))   

        complete_yc = [np.nanmean(self.ycArray[rec], axis=0) for rec in range(self.Nstat)] # [rec][which yc is assigned by anyone] [at time t]

        # only use last p percent of the simulation (as there the final reputation should be reached more or less)
        p = 1
        last_timesteps = round(len(complete_yc[0])*p)

        for i in np.arange(self.NA)[::-1]:
            # reputation of i
            rep = [np.nanmean(np.mean([self.IdataArray[rec][j][i][-last_timesteps:] for j in range(self.NA) if j!=i])) for rec in range(self.Nstat)]
            rep = np.array(rep).flatten()
            # lie acceptance rate
            lie_acceptance_rate = [np.nansum((1-np.array(self.honestyArray[rec][i][-last_timesteps:]))*np.array(complete_yc[rec][-last_timesteps:]))/np.nansum(1-np.array(self.honestyArray[rec][i][-last_timesteps:])) for rec in range(self.Nstat)] # average over all non-zero elements
            lie_acceptance_rate = np.array(lie_acceptance_rate).flatten()

            corr = round(np.corrcoef(rep, lie_acceptance_rate)[0][1], 2)

            plt.scatter(rep, lie_acceptance_rate, color=Acolor[i], s=30, alpha=0.3, label=f'correlation: {corr}')
        
        #plt.xlim([0, 1])
        #plt.ylim([0, 0.7])
        plt.xlabel('reputation', fontsize = 20)
        plt.ylabel('undetected lies', fontsize = 20)
        plt.title(make_latex_str(self.title), fontsize = 20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.savefig(name+f'_rep_LieAcceptance_{p}.png')
        plt.close() 
        if verbose: print("Plot in", name + f'_rep_LieAcceptance_{p}.png')

class CombinedStrategies:
    def __init__(self):
        pass

    def plot_hist0(self, NA, repA0, modes, folder, Nbins=20): # special_agent0_NA_hX - FONT: Check
        """histograms for all special agents"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_ylim([0.02,20])
        ax.set_xlim([0,1])
        ax.set_yscale('log')
        #Acolor = ['black', 'cyan', 'red', 'yellow', 'green'] # for 5 special strategies (no deceptive)
        Acolor = ['black', 'blue', 'cyan', 'red', 'yellow', 'green'] # for 6 special strategies
        bin_edges = np.linspace(0,1,Nbins+1)
        if True: # separate lines for each strategy
            for i in range(len(repA0)): # plot reputation
                rep = repA0[i]
                ax.hist(bin_edges[:-1], bin_edges, weights=rep, histtype='step',\
                        color = Acolor[i], linewidth = 2, label=modes[i])  

                np.savetxt(f'{modes[i]}_new.out', rep, delimiter=',')

            ax.plot([0,1],[1,1], color='grey', linewidth = 0.5)
            ax.legend(fontsize = 20, loc='lower left') #14 # upper right
            plt.xlabel(make_latex_str('perceived honesty'), fontsize = 20) #20
            plt.ylabel(make_latex_str('frequency density'), fontsize = 20) 
            plt.xticks(fontsize = 18) #12
            plt.yticks(fontsize = 18)
            plt.title(make_latex_str('special agent among '+str(NA-1)+' ordinary agents'), fontsize = 24) 
            fig.tight_layout()
            plt.savefig(folder+"/special_agent0_"+str(NA)+"Ah"+str(i)+".png")
            plt.close() 
            if verbose: print("Combined histogram plot in",folder+"/special_agent0_"+str(NA)+"Ah"+str(i)+".png") 
    
        elif len(modes) >= 3: # compine all special strategies
            all_special = []
            ordinary = None
            for i in range(len(repA0)): # plot reputation
                rep = repA0[i]
                if modes[i] == 'ordinary':
                    ordinary = rep
                elif modes[i] != 'clever':
                    all_special.append(rep)
            all_special = np.sum(all_special, axis=0)

            # save for comparism wiht old data
            np.savetxt('ordinary_old.out', ordinary, delimiter=',')
            np.savetxt('all_special_old.out', all_special, delimiter=',')

            ax.hist(bin_edges[:-1], bin_edges, weights=ordinary, histtype='step',\
                    color = 'black', linewidth = 2, density=True, label='ordinary') 
            ax.hist(bin_edges[:-1], bin_edges, weights=all_special, histtype='step',\
                    color = 'red', linewidth = 2, density=True, label='special strategies') 
            
            ax.plot([0,1],[1,1], color='grey', linewidth = 0.5)
            ax.legend(fontsize = 20, loc='lower left') #14 # upper right
            plt.xlabel(make_latex_str('perceived honesty'), fontsize = 20) #20
            plt.ylabel(make_latex_str('frequency density'), fontsize = 20) 
            plt.xticks(fontsize = 18) #12
            plt.yticks(fontsize = 18)
            plt.title(make_latex_str('special agent among '+str(NA-1)+' ordinary agents'), fontsize = 24) 
            fig.tight_layout()
            plt.savefig(folder+"/all_special_agent0_NA_"+str(i)+".png", dpi=300)
            plt.close() 
            if verbose: print("Combined histogram plot in", folder+"/all_special_agent0_NA_"+str(i)+".png")   

    def plot_BADEnotBADE_means(self, NA, dc, folder):
        data = dc.BADEnotBADE_means['data']
        strategies = list(data.keys())
        # no clever
        if 'clever' in strategies:
            strategies.remove('clever')
        assert len(strategies) > 0, 'Data for combined plot is empty. Please check where the data should come from and activate the corresponding stat_plot.'
        x_data = [_ for _ in range(len(strategies))]

        all_plots = {'info': {'y_label': 'informedness of ordinary agents', 'limits': [-0.25, 0.25], 'center': 0, 'logy': False},
                    'rep0': {'y_label': 'reputation of agent red', 'limits': [-0.05, 1.05], 'center': 0.5, 'logy': False},
                    'kappa': {'y_label': 'surprise', 'limits': [0.01, 60000], 'center': 1, 'logy': True},
                    'ToM': {'y_label': 'ToM accuracy', 'limits': [-0.35, 0.25], 'center': 0, 'logy': False},
                    'rel_red': {'y_label': 'conversations to agent red', 'limits': [0.35, 0.75], 'center': 0.5, 'logy': False},
                    'fr_red': {'y_label': 'friendship to agent red', 'limits': [-0.05, 1.05], 'center': 0.5, 'logy': False}
                    }

        for plot, params in all_plots.items():
            assert NA==3
            fig, ax = plt.subplots(figsize=(10, 6))
            # average over ordinary agents 1 and 2
            BADE = [np.nanmean(np.array([data[s][plot][0][1], data[s][plot][0][2]])) for s in strategies]
            not_BADE = [np.nanmean(np.array([data[s][plot][1][1], data[s][plot][1][2]])) for s in strategies]
            if plot == 'info':
                BADE_ordinary_only = [np.nanmean(np.array([data[s][plot+'_ordinary_only'][0][1], data[s][plot+'_ordinary_only'][0][2]])) for s in strategies]
                not_BADE_ordinary_only = [np.nanmean(np.array([data[s][plot+'_ordinary_only'][1][1], data[s][plot+'_ordinary_only'][1][2]])) for s in strategies]

                        
            # Set position of bar on X axis 
            x_center = np.arange(len(strategies)) 
            if plot == 'info':
                barWidth = 0.25
                br1 = [x-0.5*barWidth for x in x_center]
                br2 = [x+0.5*barWidth for x in x_center]
                plt.bar(br1, BADE, color ='orange', width = barWidth, label ='suffering from LICS') 
                plt.bar(br2, not_BADE, color ='lightgray', width = barWidth, label ='healthy') 
                plt.bar(br1, BADE_ordinary_only, edgecolor ='orange', linestyle='-', linewidth=2, color='none', width = barWidth, label ='') 
                plt.bar(br2, not_BADE_ordinary_only, edgecolor ='lightgray', linestyle='-', linewidth=2, color='none', width = barWidth, label ='w.r.t. ordinary agents') 
            else:
                barWidth = 0.25
                br1 = [x-barWidth/2 for x in x_center]
                br2 = [x+barWidth/2 for x in x_center]
                plt.bar(br1, BADE, color ='orange', width = barWidth, label ='suffering from LICS') 
                plt.bar(br2, not_BADE, color ='lightgray', width = barWidth, label ='healthy') 

            if params['logy']:
                plt.yscale('log')
        
            plt.legend(fontsize=18)#, loc = "upper left")
            plt.xticks(x_data, [s+'\nagent red' for s in strategies], rotation=30)
            plt.ylabel(make_latex_str(params['y_label']), fontsize = 20) 
            plt.xticks(fontsize = 16) #12
            plt.yticks(fontsize = 16)
            fig.tight_layout()
            plt.savefig(folder+f"/BADEnotBADE_combined_{plot}.png", dpi=200)
            plt.close() 
            if verbose: print("Combined histogram plot in",folder+f"/BADEnotBADE_combined_{plot}.png")

    def plot_comb_LR_hist(self, NA, dc, folder):
        data = dc.likelihoodRatio['data']
        strategies = data.keys()
        # ignore clever:
        strategies = [s for s in strategies if s != 'clever']
        colors = {0:'red', 1:'cyan', 2:'black'}
        rgb_colors = {0:'red', 1:'cyan', 2:'black'}
        for s in strategies:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlim([0.4, 130])
            ax.set_ylim([10, 3e5])
            logbins = np.geomspace(0.3, 150, 200)

            for i in np.arange(NA):
                print('plotting ', i)
                R_blush = data[s][i]
                R_blush_ordinary = data['ordinary'][i]

                ax.hist(R_blush_ordinary, bins=logbins, histtype='step', log=True, color = colors[i], linewidth = 1, ls='solid', alpha=0.4)  
                ns , bins , _ = ax.hist(R_blush, bins=logbins, histtype='step', log=True, color = colors[i], linewidth = 3, ls='solid')
                # mark BADE region
                BADE_start = 11/9
                BADE_bins = [b for b in bins if b <= BADE_start]
                BADE_ns = ns[:len(BADE_bins)]
                ax.fill_between(BADE_bins, BADE_ns, alpha=0.4, step='post', color=colors[i])

                #for n, l, r in zip(ns, bins, bins[1:]):
                #    if l > start:
                #        if r < stop:
                #            # these bins fall completely within the range
                #            ax.fill_between([l, r], 0, [n, n], alpha=0.4, color=colors[i])
                #        elif l < stop < r:
                #            ax.fill_between([l, stop], 0, [n, n], alpha=0.4, color=colors[i])  # partial fill
                #    elif l < start < r:
                #        ax.fill_between([start, r], 0, [n, n], alpha=0.4, color=colors[i])  # partial fill
            if s == 'ordinary':
                title = 'ordinary agents'
            else:
                title = f'{s} among ordinary agents'

            plt.xscale('log')
            plt.xlabel(make_latex_str('blushing likelihood ratio'), fontsize = 20)
            plt.ylabel(make_latex_str('frequency'), fontsize = 20)
            plt.title(make_latex_str(title), fontsize = 24)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            fig.tight_layout()
            plt.savefig(folder+f"/{s}_LRhist.png", dpi=300)
            plt.close() 
            print('Plot in ', folder+f"/{s}_LRhist.png")

            

def reconstruct(filename, index=None, Nfiles=None, prop = 0):
    """Takes a simulation file and reconstructs a simulation Record from it.

    Args:
        filename (str): the location of the simulation file (in .json format)
        prop (int, optional): timescale. Supported values: 0, 1. Defaults to 0.

    Returns:
        _type_: _description_
    """
    with open(filename, 'r') as f:
        # read the first line
        first_line = json.loads(f.readline()) # dict
        NA = first_line['NA']
        NR = first_line['NR']
        mode = first_line['mode']
        title = first_line['title']
        outfile = first_line['outfile']
        RSeed = first_line['RSeed']
        colors = first_line['colors']
        alphas = first_line['alphas']
        x_est = first_line['x_est']
        fr_affinities = first_line['fr_affinities']
        shynesses = first_line['shynesses']
        CONTINUOUS_FRIENDSHIP = first_line['switches']['CONTINUOUS_FRIENDSHIP']

        if not index is None and not Nfiles is None:
            print(f'reconstructing file {index+1} of {Nfiles}.', end='\r')

        # create Reconstruction instance with basic infos from 1st line
        rec = Rec(NA, NR, mode, title, RSeed, colors, alphas, x_est, fr_affinities, shynesses, CONTINUOUS_FRIENDSHIP)
        if prop == -1: # other timescale
            rec.Idata = [[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)]
            rec.Idata_rms = [[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)]
            rec.kappa = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.K = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.yc = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.honesties = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.friendships = [[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)]
            rec.friendships_rms = [[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)]
            rec.relationsc = [[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)]
            rec.relationsm = [[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)]
            rec.Iothers = [[[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
            rec.Iothers_rms = [[[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
            rec.Cothers = [[[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
            rec.Cothers_rms = [[[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
            rec.Jothers = [[[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
            rec.Jothers_rms = [[[[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)] for i in range(NA)] for i in range(NA)]
            rec.prob_bt = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_bt_rms = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_bl = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_bl_rms = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_ct = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_ct_rms = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_cl = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
            rec.prob_cl_rms = [[np.nan for i in range((NA-1)*NR*3+1)]for i in range(NA)]
        
        # read all the other lines and fill rec
        line = f.readline()

        # ignore initial status
        while line and json.loads(line)['event_type'] != 'initial_status':
            line = f.readline()

        last_speaker = None
        last_comm_time = None
        while line and json.loads(line)['event_type'] != 'final_status':
            line = json.loads(line)
            if line['event_type'] == 'communication':
                last_speaker = line['a']
                time = line['time'] 
                last_comm_time = time
                rec.honesty[line['a']][line['time']] = int(line['honest'])
                rec.comm.append({'time':line['time'], 'a':line['a'], 'b_set':line['b_set'], 'c':line['c'], \
                            'J_mean':Info(line['J'][0], line['J'][1]).mean, 'honest':line['honest']})
            elif line['event_type'] == 'self_update':
                id = line['id']
                time = line['time'] 
                EventNr = line['EventNr']
                for key in line:
                    # information
                    if 'I_' in key: 
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec.Idata[id][id][time] = data.mean
                        rec.Idata_rms[id][id][time] = data.rms
                    # theory of mind
                    elif 'Jothers' in key:
                            data = line[key] # list [mu, la]
                            data = Info(data[0], data[1]) # info-object
                            rec.Jothers[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.mean
                            rec.Jothers_rms[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.rms
                    # Nt/Nl
                    elif key == 'Nt':
                        rec.Nt[id] = line['Nt']
                    elif key == 'Nl':
                        rec.Nl[id] = line['Nl'] 
            elif line['event_type'] == 'update':
                id = line['id']
                time = line['time']
                EventNr = line['EventNr']
                for key in line:
                    # information
                    if 'I_' in key: 
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec.Idata[id][convert_name(key)[0]][time] = data.mean
                        rec.Idata_rms[id][convert_name(key)[0]][time] = data.rms
                    # theory of mind
                    elif 'Iothers' in key:
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec.Iothers[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.mean
                        rec.Iothers_rms[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.rms
                    elif 'Cothers' in key:
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec.Cothers[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.mean
                        rec.Cothers_rms[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.rms
                    elif 'Jothers' in key:
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec.Jothers[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.mean
                        rec.Jothers_rms[id][convert_name(key)[0]][convert_name(key)[1]][time] = data.rms
                    # friendships
                    elif 'friendship' in key:
                        numbers = get_numbers(key)
                        assert len(numbers) == 1
                        addressee = int(numbers[0]) # id of addressee
                        if CONTINUOUS_FRIENDSHIP:
                            rec.friendships[id][addressee][time] = Info(line[key][0], line[key][1]).mean
                            rec.friendships_rms[id][addressee][time] = Info(line[key][0], line[key][1]).rms
                        else:
                            if 'friendship+' in key: # new friend
                                assert line[key][0]==1 and  line[key][1]==0
                                rec.friendships[id][addressee][time] = 1
                            if 'friendship-' in key: # new enemy
                                assert line[key][0]==0 and  line[key][1]==1
                                rec.friendships[id][addressee][time] = 0
                            # rms is not tracked in the discrete case
                    # relations
                    elif 'relation' in key:
                        addressee = int(key[11:])
                        if 'relationsc' in key: # talked to somebody
                            rec.relationsc[id][addressee][time] = line[key] 
                        elif 'relationsm' in key: # heard about somebody
                            rec.relationsm[id][addressee][time] = line[key]
                    # kappa
                    elif key == 'kappa':
                        rec.kappa[id][time] = line[key]
                    # K
                    elif key == 'new_K':
                        rec.K[id][time] = line[key]
                    # yc
                    elif key == 'yc':
                        if prop == 0:
                            message_time = last_comm_time
                        else:
                            message_time = time # in propaganda: save yc at the time where it is determined (not where message was sent)
                        rec.yc[id][int(message_time)] = line[key]
                    # EBEH parameters
                    elif 'prob_' in key:
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        if key == 'prob_bt':
                            rec.prob_bt[id][time] = data.mean
                            rec.prob_bt_rms[id][time] = data.rms
                        if key == 'prob_bl':
                            rec.prob_bl[id][time] = data.mean
                            rec.prob_bl_rms[id][time] = data.rms
                        if key == 'prob_ct':
                            rec.prob_ct[id][time] = data.mean
                            rec.prob_ct_rms[id][time] = data.rms
                        if key == 'prob_cl':
                            rec.prob_cl[id][time] = data.mean
                            rec.prob_cl_rms[id][time] = data.rms



            elif line['event_type'] == 'final_status':
                id = int(line['id'])
                time = 0
                data = line['I_0'] # list [mu, la]
                data = Info(data[0], data[1]) # info-object
                rec.Idata[id][0][time] = data.mean
                rec.Idata_rms[id][0][time] = data.rms
            line = f.readline()
            
    #for i in range(NA): # add self-friedship
    #    rec.friendships[i][i][1] = 1
    #    for j in range(NA): # delete t=0
    #        rec.friendships[i][j] = rec.friendships[i][j]
    rec.complete(cont_fr = CONTINUOUS_FRIENDSHIP)
    #print('relationsc_4,5: ', rec.relationsc[4], rec.relationsc[5])
    if compatibility:
        rec.calculate_awa_rep_half()
    else:
        rec.calculate_awa_rep()
    return rec, title
        
def reload_data_stat(filename):
    with open(filename, 'r') as f:
        d = json.loads(f.readline()) # first line: dict
        d = json.loads(f.readline()) # second line: dict
        if False: # new
            data_stat = Data_Stat(d['mode'], d['Nstat'], d['NA'], d['NR'], d['Acolor'], d['Aalpha'], d['x_est'], d['title'], d['CONTINUOUS_FRIENDSHIP'], d['IdataArray'], \
                                    d['IdataArray_rms'], d['friendships'], d['friendships_rms'], d['relationsc'], d['relationsm'], d['kappaArray'], d['KArray'], d['ycArray'], d['honestyArray'], d['IothersArray'], \
                                    d['prob_btArray'],  d['prob_blArray'],  d['prob_ctArray'],  d['prob_clArray'], \
                                    d['prob_btArray_rms'],  d['prob_blArray_rms'],  d['prob_ctArray_rms'],  d['prob_clArray_rms'])
        elif False: # old
            data_stat = Data_Stat(d['mode'], d['Nstat'], d['NA'], d['NR'], d['Acolor'], d['Aalpha'], d['x_est'], d['title'], d['IdataArray'], \
                                d['IdataArray_rms'], d['friendships'], d['kappaArray'], \
                                #d['prob_btArray'],  d['prob_blArray'],  d['prob_ctArray'],  d['prob_clArray'], \
                                #d['prob_btArray_rms'],  d['prob_blArray_rms'],  d['prob_ctArray_rms'],  d['prob_clArray_rms']
                                )
        else:
            data_stat = Data_Stat(d)

    return data_stat

def save_density(arr, filename):
    data = arr.tolist()
    # check if directory exists
    if os.path.exists(get_folder(filename)):
        pass
    else:
        os.mkdir(get_folder(filename))
    with open(filename, 'w+') as f:
        json.dump(data, f)
        f.write('\n') 
    
def load_density(filename):
    with open(filename, 'r') as f:
        data = json.loads(f.readline())
    return np.array(data)

def evaluate(file: str, dc = None, index=None, Nfiles=None, plot_selection=None, default=False):
    """Takes the file location of the simiulation to be evaluated and plots it

    Args:
        file (str): the path to the simiulation results
        plot_mode (str): the plot mode. Default value: 'dynamics'

    Returns:
        Rec: the instance of the simulation record
    """

    if not dc:
        dc = DataCollector()

    if not default and plot_selection is None:
        plot_selection = {}

    rec, _ = reconstruct(file, index=index, Nfiles=Nfiles)
    name = file[:-5]
    
    if default: # only default plot
        rec.plot_dynamics(name, dc)
    else: # plot according to plot_selection
        plots = [plots for plots, istrue in plot_selection.items() if istrue==True]
        for plot in plots: 
            getattr(rec, plot)(name, dc) # make the plot
    return rec

def evaluate_stat(folder: str, plot_selection_stat=None, plot_selection_comb_stat=None, plot_selection_single=None, default_single=False, default_stat=False, new_evaluation=False):
    dc = DataCollector()
    combined_strategies = CombinedStrategies()
    
    if plot_selection_stat is None:
        assert default_single or default_stat
    
    Nstat, modes, NAs, NR, seeds = overview(folder)
    for NA in NAs:
        repA0 = []
        for mode in modes:
            if 'all' in mode.keys() and mode['all']=='ordinary' and len(mode)>1: mode.pop('all', None)
            mode_name = '__'.join([f"{names[strategy]}_{'_'.join([k for k,v in mode.items() if v == strategy])}" for strategy in set(mode.values())])
            name = folder+'/stat_'+mode_name+'_NA'+str(NA)
            # check if data_stat file already exists
            if os.path.exists(name+'_data_stat.json') and not new_evaluation: 
                print('reloaded previous evaluation')
                data_stat = reload_data_stat(name+'_data_stat.json')
                Ireph0 = data_stat.plot_statistics(name, dc)
                repA0.append(Ireph0)
                plots = [plot for plot, istrue in plot_selection_stat.items() if istrue==True]
                for plot in plots: 
                    getattr(data_stat, plot)(name, dc) # make the plot

                del data_stat

            else: # evaluate for the first time
                print('1st evaluation')
                files = glob.glob(folder + f"/*{mode_name}_NA{NA}_RS*.json", recursive = False)
                # bring files in correct order
                name_without_seed = [f.rsplit('RS', 1)[0] for f in files]
                assert all(name == name_without_seed[0] for name in name_without_seed)
                name_without_seed = name_without_seed[0]
                seeds.sort()
                ordered_files = [name_without_seed + f'RS{seed}.json' for seed in seeds]
                # evaluate file by file
                recs = []
                for index,f in enumerate(ordered_files):
                    rec = evaluate(f, dc = dc, index=index, plot_selection=plot_selection_single, Nfiles=len(ordered_files)) # all single plots are created here, default: no single plots in stat_evaluation
                    recs.append(rec)
                data_stat = create_data_stat(mode, NA, NR, Nstat, recs)

                Ireph0 = data_stat.plot_statistics(name, dc)
                repA0.append(Ireph0)

                # save data_stat in json file
                data_stat.save(name)

                # make plots
                if default_stat:
                    plots = ['plot_histrogram']
                elif plot_selection_stat:
                    plots = [plot for plot, istrue in plot_selection_stat.items() if istrue==True]
                else:
                    plots = []
                
                for plot in plots: 
                    getattr(data_stat, plot)(name, dc) # make the plot

                del recs
                del data_stat

            # clean RAM (especially data_stat and recs)
            gc.collect()

        if plot_selection_comb_stat:
            plots = [plot for plot, istrue in plot_selection_comb_stat.items() if istrue==True]
            for plot in plots: 
                if plot == 'plot_hist0':
                    for i in range(1,len(repA0)+1): # how many special agents should be plotted together
                        combined_strategies.plot_hist0(NA, repA0[:i], modes[:i], folder)
                else:
                    getattr(combined_strategies, plot)(NA, dc, folder) # make the plot



def evaluate_prop(file, prop):
    dc = DataCollector()
    folder = get_folder(file)
    rec, title = reconstruct(file, prop = prop)
    name = file[:-5]
    # prepare ignore list
    if prop == 1:
        ignore = {(i,j) for i in range(rec.NA) for j in range(rec.NA)}
        for i in range(1,rec.NA): 
            ignore.discard((i,0))
            ignore.add(i)
        ignore.add(0)
    if prop == -1:
        ignore = {(i,i) for i in range(rec.NA)}
        for i in range(rec.NA):
            ignore.add(i)
            ignore.add((0,i))
    rec.plot_dynamics(name, dc, ignore = ignore, prop=prop)
