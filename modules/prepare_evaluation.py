# prepare evaluation:
# take .json file 
# return .hdf5 file containing sparse matrices

import os
import gc
import glob
import json
import pandas as pd
import numpy as np

from modules.globals import init_LUT, names
init_LUT(load=False)
from modules.informationtheory import Info
from modules.helperfunc import convert_name, overview

class Rec:
    def __init__(self, NA, NR, mode, title, RSeed, colors, alphas, x_est, fr_affinities, shynesses):
        # always needed:
        self.NA = NA
        self.NR = NR
        self.mode = mode
        self.title = title
        self.RSeed = RSeed
        self.colors = colors
        self.alphas = alphas
        self.x_est = x_est
        self.fr_affinities = fr_affinities
        self.shynesses = shynesses

        ## SPARSE DATA

        # Idata: who thinks, about whom, when, what
        self.Idata = [[],[],[],[]]
        self.Idata_rms = [[],[],[],[]]        

        #friendship: who considers, whom a friend, when, how much
        self.friendships = [[],[],[],[]]
        self.friendships_rms = [[],[],[],[]]

        #relationsc: who has talked, with whom, when, how often
        self.relationsc = [[], [], [], []]
        #relationsm[who has heard][about whom][time] how often
        self.relationsm = [[], [], [], []]

        # theory of mind
        # Iothers: who's mind assumes, that who thinks, about whom, when, what
        self.Iothers = [[],[],[],[],[]]
        self.Iothers_rms = [[],[],[],[],[]]
        # Cothers: whose mind assumes, that who wants me/self to believe, about whom, when, what
        self.Cothers = [[],[],[],[], []]
        self.Cothers_rms = [[],[],[],[], []]

        # kappa: who's kappa, when, what
        self.kappa = [[],[],[]]

        # yc: who assigned, when (when message was sent), which yc
        self.yc = [[],[],[]]

        # honesty: who was honest, when (when message was sent), (0,1)
        self.honesty = [[],[],[]]

        # Number of honest / dishonest statements
        self.Nt = [0 for i in range(NA)] # keep only the final values here
        self.Nl = [0 for i in range(NA)] # keep only the final values here

        # EBEH
        self.prob_bt = [[],[],[]] # who's assumed blushing frequency, when what
        self.prob_bl = [[],[],[]] # who's assumed blushing frequency, when what
        self.prob_bt_rms = [[],[],[]] # who's assumed blushing frequency, when what
        self.prob_bl_rms = [[],[],[]] # who's assumed blushing frequency, when what

    def _append_sparse_data(self, obj, values):
        dim = len(obj)
        assert len(values) == dim, f'got wrong dimensions for {obj}: {len(values)} instead of {dim}.'
        for i in range(dim):
            obj[i].append(values[i]) 

class Rec_Stat:
    """contains all recs belonging to one stat evaluation: only 1 mode and NA."""
    def __init__(self, mode, Nstat, NA, NR, Acolor, Aalpha, x_est, fr_affinities_stat, shynesses_stat, title, IdataArray, IdataArray_rms, friendships, friendships_rms, relationsc, relationsm, kappaArray, ycArray, honestyArray, \
                    Iothers_stat, Iothers_rms_stat, Cothers_stat, Cothers_rms_stat, prob_btArray, prob_blArray, prob_bt_rmsArray, prob_bl_rmsArray, Nt, Nl):
        self.mode = mode
        self.Nstat = Nstat
        self.NA = NA
        self.NR = NR
        self.Acolor = Acolor
        self.Aalpha = Aalpha
        self.x_est = x_est
        self.fr_affinities_stat = fr_affinities_stat
        self.shynesses_stat = shynesses_stat
        self.title = title
        self.IdataArray = IdataArray # IdataArray[which rec][who thinks][about whom][time]
        self.IdataArray_rms = IdataArray_rms # IdataArray_rms[which rec][who thinks][about whom][time]
        self.friendships = friendships
        self.friendships_rms = friendships_rms
        self.relationsc = relationsc
        self.relationsm = relationsm
        self.kappaArray = kappaArray # [which rec][who's kappa][time]
        self.ycArray = ycArray # [which rec][who assigned this][time of the message]
        self.honestyArray = honestyArray # [which rec][who's honesty][time of the message]
        self.Iothers_stat = Iothers_stat
        self.Iothers_rms_stat = Iothers_rms_stat
        self.Cothers_stat = Cothers_stat
        self.Cothers_rms_stat = Cothers_rms_stat
        self.prob_btArray = prob_btArray
        self.prob_blArray = prob_blArray
        self.prob_bt_rmsArray = prob_bt_rmsArray
        self.prob_bl_rmsArray = prob_bl_rmsArray
        self.Nt = Nt
        self.Nl = Nl

    def save(self, name):
        gc.collect()
        # save values to hdf5 file
        filename = name+'_data.hdf5'

        d = {'mode': str(self.mode), 'Nstat':self.Nstat, 'NA':self.NA, 'NR':self.NR, 'title': self.title, \
                'x_est':self.x_est, 'Acolor': self.Acolor, 'Aalpha':self.Aalpha}
        df = pd.DataFrame.from_dict(d)
        df.to_hdf(filename, 'general information')
        gc.collect()
        for key, value in self.__dict__.items():
            if key not in  ['mode', 'Nstat', 'NA', 'NR', 'title', 'x_est', 'Acolor', 'Aalpha']:
                # sort by dimension of dataframe:
                if key in ['IdataArray', 'IdataArray_rms', 'friendships', 'friendships_rms', 'relationsc', 'relationsm']: # rec, who, about whom, when, what
                    # make dataframe
                    data = np.array(value)
                    rec_numbers = np.array([[i]*np.shape(data)[-1] for i in range(self.Nstat)]).flatten()
                    runs_flattened = np.concatenate(data, axis=-1) # make the different runs be one huge list
                    df = pd.DataFrame(runs_flattened).transpose()
                    df.columns = ['who', 'about whom', 'when', 'what']
                    df.insert(0, 'rec', rec_numbers)
                    df.to_hdf(filename, str(key))
                elif key in ['kappaArray', 'ycArray', 'honestyArray', 'prob_btArray', 'prob_blArray', 'prob_bt_rmsArray', 'prob_bl_rmsArray']: # rec, who, when, what
                    # make dataframe
                    data = np.array(value)
                    rec_numbers = np.array([[i]*np.shape(data)[-1] for i in range(self.Nstat)]).flatten()
                    runs_flattened = np.concatenate(data, axis=-1) # make the different runs be one huge list
                    df = pd.DataFrame(runs_flattened).transpose()
                    df.columns = ['who', 'when', 'what']
                    df.insert(0, 'rec', rec_numbers)
                    df.to_hdf(filename, str(key))
                elif key in ['Iothers_stat', 'Iothers_rms_stat', 'Cothers_stat', 'Cothers_rms_stat']: # rec, who, who thinks/wants, about whom, when, what 
                    # make dataframe
                    data = np.array(value)
                    rec_numbers = np.array([[i]*np.shape(data)[-1] for i in range(self.Nstat)]).flatten()
                    runs_flattened = np.concatenate(data, axis=-1) # make the different runs be one huge list
                    df = pd.DataFrame(runs_flattened).transpose()
                    df.columns = ['who', 'who thinks/wants', 'about whom', 'when', 'what']
                    df.insert(0, 'rec', rec_numbers)
                    df.to_hdf(filename, str(key))
                elif key in ['Nt', 'Nl', 'fr_affinities_stat', 'shynesses_stat']: # rec, who, what
                    data = np.array(value)
                    rec_numbers = np.array([[i]*np.shape(data)[-1] for i in range(self.Nstat)]).flatten()
                    agents = list(np.arange(self.NA))*self.Nstat
                    runs_flattened = np.concatenate(data, axis=-1) # make the different runs be one huge list
                    df = pd.DataFrame([rec_numbers, agents, runs_flattened]).transpose()
                    df.columns = ['rec', 'who', 'what']
                    df.to_hdf(filename, str(key))
                else:
                    raise KeyError(f'key {key} not found!')
                gc.collect()
        print()
        print(f'Evaluation prepared. \nInformation in {name}_data.hdf5')  
    
        return name+'_data.hdf5'

def reconstruct_sparse(filename, index, Nfiles):
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
        # create Reconstruction instance with basic infos from 1st line
        rec = Rec(NA, NR, mode, title, RSeed, colors, alphas, x_est, fr_affinities, shynesses)
        gc.collect()
        # read all the other lines and fill rec
        line = f.readline()
        last_speaker = None
        last_comm_time = None

        # skip initial status
        while line and json.loads(line)['event_type'] == 'initial_status':
            line = f.readline()

        while line and json.loads(line)['event_type'] != 'final_status':
            line = json.loads(line)
            
            if line['time']%1000 == 0:
                gc.collect()

            if line['event_type'] == 'communication':
                last_speaker = line['a']
                last_receiver_set = line['b_set']
                time = line['time'] 
                last_comm_time = time
                rec._append_sparse_data(rec.honesty, [last_speaker, time, line['honest']]) 
            if line['event_type'] == 'self_update':
                id = line['id']
                time = line['time'] 
                print(f'reconstructing file {index+1} of {Nfiles}.', end='\r')
                for key in line:
                    # information
                    if 'I_' in key: 
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec._append_sparse_data(rec.Idata, [id, id, time, data.mean])
                        rec._append_sparse_data(rec.Idata_rms, [id, id, time, data.rms])
                    # Nt/Nl
                    elif key == 'Nt':
                        rec.Nt[id] = line['Nt']
                    elif key == 'Nl':
                        rec.Nl[id] = line['Nl'] 
            elif line['event_type'] == 'update':
                id = line['id']
                time = line['time']
                for key in line:
                    # information
                    if 'I_' in key: 
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec._append_sparse_data(rec.Idata, [id, convert_name(key)[0], time, data.mean])
                        rec._append_sparse_data(rec.Idata_rms, [id, convert_name(key)[0], time, data.rms])

                    # theory of mind
                    elif 'Iothers' in key:
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec._append_sparse_data(rec.Iothers, [id, convert_name(key)[0], convert_name(key)[1], time, data.mean])
                        rec._append_sparse_data(rec.Iothers_rms, [id, convert_name(key)[0], convert_name(key)[1], time, data.rms])
                    elif 'Cothers' in key:
                        data = line[key] # list [mu, la]
                        data = Info(data[0], data[1]) # info-object
                        rec._append_sparse_data(rec.Cothers, [id, convert_name(key)[0], convert_name(key)[1], time, data.mean])
                        rec._append_sparse_data(rec.Cothers_rms, [id, convert_name(key)[0], convert_name(key)[1], time, data.rms])

                    # friendships
                    elif 'friendship' in key:
                        addressee = int(key[15:]) # id of addressee
                        data = Info(line[key][0], line[key][1])
                        rec._append_sparse_data(rec.friendships, [id, addressee, time, data.mean])
                        rec._append_sparse_data(rec.friendships_rms, [id, addressee, time, data.rms])
                    # relations
                    elif 'relation' in key:
                        try:
                            addressee = int(key[11:])
                        except ValueError:
                            if key[11:][0] == '_':
                                addressee = int(key[12:])
                        if key[9] =='c' or (key[9]=='_' and key[10] == 'c'): # talked to somebody
                            rec._append_sparse_data(rec.relationsc, [id, addressee, time, line[key]])

                        elif key[9] =='m' or (key[9]=='_' and key[10] == 'm'): # heard about somebody
                            rec._append_sparse_data(rec.relationsm, [id, addressee, time, line[key]])
                        else:
                            raise ValueError(f'Can not identify relation type {key}.')
                    # kappa
                    elif key == 'kappa':
                        rec._append_sparse_data(rec.kappa, [id, time, line[key]])
                    # yc
                    elif key == 'yc':
                        if len(last_receiver_set) > 1: # one-to-many
                            message_time = last_comm_time
                        elif id in last_receiver_set:
                            message_time = last_comm_time
                        else:
                            message_time = last_comm_time - 1
                        rec._append_sparse_data(rec.yc, [id, message_time, line[key]])

                    # EBEH
                    elif 'prob_' in key:
                        data = line[key] # list [mu, la]
                        info_obj = Info(data[0], data[1])
                        mean, std = info_obj.mean, info_obj.rms
                        if key == 'prob_bt':
                            rec._append_sparse_data(rec.prob_bt, [id, time, mean])
                            rec._append_sparse_data(rec.prob_bt_rms, [id, time, std])
                        elif key == 'prob_bl':
                            rec._append_sparse_data(rec.prob_bl, [id, time, mean])
                            rec._append_sparse_data(rec.prob_bl_rms, [id, time, std])



                    else:
                        if (key not in ['event_type', 'time', 'EventNr', 'id', 'new_K']) and ('Jothers' not in key): # currently unused information
                            raise ValueError('Invalid Input. Can not identify the attribute '+key+'.')

            elif line['event_type'] == 'initial_status':
                gc.collect()
                id = int(line['id'])
                time = 0
                data = line['I_0'] # list [mu, la]
                data = Info(data[0], data[1]) # info-object
                rec._append_sparse_data(rec.Idata, [id, 0, time, data.mean])
                rec._append_sparse_data(rec.Idata_rms, [id, 0, time, data.rms])
            line = f.readline()

        return rec

def create_rec_stat(mode, Nstat, NA, NR, recs):
    # same for all recs
    Acolor = recs[0].colors
    Aalpha = recs[0].alphas
    x_est = recs[0].x_est
    title = recs[0].title
    
    # padding to make all lists that get stacked the same length
    def pad(some_list, target_len):
        assert len(some_list) <= target_len
        return some_list + [np.nan]*(target_len - len(some_list))

    # stack lists
    fr_affinities_stat = [rec.fr_affinities for rec in recs]
    shynesses_stat = [rec.shynesses for rec in recs]

    Idata_maxlength = max([len(recs[i].Idata[0]) for i in range(len(recs))])
    IdataArray = [[pad(rec.Idata[i], Idata_maxlength) for i in range(len(rec.Idata))] for rec in recs]
    IdataArray_rms = [[pad(rec.Idata_rms[i], Idata_maxlength) for i in range(len(rec.Idata_rms))] for rec in recs]

    friendships_maxlength = max([len(recs[i].friendships[0]) for i in range(len(recs))])
    friendships = [[pad(rec.friendships[i], friendships_maxlength) for i in range(len(rec.friendships))] for rec in recs]
    friendships_rms = [[pad(rec.friendships_rms[i], friendships_maxlength) for i in range(len(rec.friendships_rms))] for rec in recs]

    relationsc_maxlength = max([len(recs[i].relationsc[0]) for i in range(len(recs))])
    relationsc = [[pad(rec.relationsc[i], relationsc_maxlength) for i in range(len(rec.relationsc))] for rec in recs]

    relationsm_maxlength = max([len(recs[i].relationsm[0]) for i in range(len(recs))])
    relationsm = [[pad(rec.relationsm[i], relationsm_maxlength) for i in range(len(rec.relationsm))] for rec in recs]

    kappa_maxlength = max([len(recs[i].kappa[0]) for i in range(len(recs))])
    kappaArray = [[pad(rec.kappa[i], kappa_maxlength) for i in range(len(rec.kappa))] for rec in recs]

    yc_maxlength = max([len(recs[i].yc[0]) for i in range(len(recs))])
    ycArray = [[pad(rec.yc[i], yc_maxlength) for i in range(len(rec.yc))] for rec in recs]

    honesty_maxlength = max([len(recs[i].honesty[0]) for i in range(len(recs))])
    honestyArray = [[pad(rec.honesty[i], honesty_maxlength) for i in range(len(rec.honesty))] for rec in recs]

    Iothers_maxlength = max([len(recs[i].Iothers[0]) for i in range(len(recs))])
    Iothers_stat = [[pad(rec.Iothers[i], Iothers_maxlength) for i in range(len(rec.Iothers))] for rec in recs]
    Iothers_rms_stat = [[pad(rec.Iothers_rms[i], Iothers_maxlength) for i in range(len(rec.Iothers_rms))] for rec in recs]

    Cothers_maxlength = max([len(recs[i].Cothers[0]) for i in range(len(recs))])
    Cothers_stat = [[pad(rec.Cothers[i], Cothers_maxlength) for i in range(len(rec.Cothers))] for rec in recs]
    Cothers_rms_stat = [[pad(rec.Cothers_rms[i], Cothers_maxlength) for i in range(len(rec.Cothers_rms))] for rec in recs]

    prob_bt_maxlength = max([len(recs[i].prob_bt[0]) for i in range(len(recs))])
    prob_btArray = [[pad(rec.prob_bt[i], prob_bt_maxlength) for i in range(len(rec.prob_bt))] for rec in recs]
    prob_bl_maxlength = max([len(recs[i].prob_bl[0]) for i in range(len(recs))])
    prob_blArray = [[pad(rec.prob_bl[i], prob_bl_maxlength) for i in range(len(rec.prob_bl))] for rec in recs]
    prob_bt_rms_maxlength = max([len(recs[i].prob_bt_rms[0]) for i in range(len(recs))])
    prob_bt_rmsArray = [[pad(rec.prob_bt_rms[i], prob_bt_rms_maxlength) for i in range(len(rec.prob_bt_rms))] for rec in recs]
    prob_bl_rms_maxlength = max([len(recs[i].prob_bl_rms[0]) for i in range(len(recs))])
    prob_bl_rmsArray = [[pad(rec.prob_bl_rms[i], prob_bl_rms_maxlength) for i in range(len(rec.prob_bl_rms))] for rec in recs]


    Nt = [rec.Nt for rec in recs]
    Nl = [rec.Nl for rec in recs]

    rec_stat = Rec_Stat(mode, Nstat, NA, NR, Acolor, Aalpha, x_est, fr_affinities_stat, shynesses_stat, title, IdataArray, IdataArray_rms, \
                    friendships, friendships_rms, relationsc, relationsm, kappaArray, ycArray, honestyArray, \
                    Iothers_stat, Iothers_rms_stat, Cothers_stat, Cothers_rms_stat, prob_btArray, prob_blArray, prob_bt_rmsArray, prob_bl_rmsArray, Nt, Nl) 

    return rec_stat

def prepare_evaluation(folder):
    Nstat, modes, NAs, NR, seeds = overview(folder)
    for NA in NAs:
        repA0 = []
        for mode in modes:
            if 'all' in mode.keys() and mode['all']=='ordinary' and len(mode)>1: mode.pop('all', None)
            mode_name_splits = [f"{names[strategy]}_{'_'.join([k for k,v in mode.items() if v == strategy])}" for strategy in set(mode.values())]
            mode_name_splits.sort()
            mode_name = '__'.join(mode_name_splits)
            name = folder+'/stat_'+mode_name+'_NA'+str(NA)
            # check if _data already exists
            if os.path.exists(name+'_data.hdf5'): 
                print('This folder has already been prepared for evaluation.')
                path = name+'_data.hdf5'
            else: # evaluate for the first time
                print('1st evaluation')
                all_files = glob.glob(folder + "/*.json", recursive = False)
                mode_files = []
                for f in all_files:
                    mode_specifying_name = f.split('_NA')[0].split('/')[-1]
                    mode_specifying_name_splits = set(mode_specifying_name.split('__'))
                    if mode_specifying_name_splits==set(mode_name_splits) and not 'data' in f: # exclude evaluation_stat files in case there are some
                        mode_files.append(f)
                recs = []
                gc.collect()
                for index,f in enumerate(mode_files):
                    rec = reconstruct_sparse(f, index, len(mode_files)) # all single plots are created here
                    recs.append(rec)
                    gc.collect()
                data_stat = create_rec_stat(mode, Nstat, NA, NR, recs)
                gc.collect()
                path = data_stat.save(name)
    return path
   


if __name__ == '__main__':
    prepare_evaluation(os.path.join('ytracking_50A_x_newTime','ordinary_small'))
