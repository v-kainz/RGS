# some small helper functions

import os
import numpy as np
import pandas as pd
from numpy import exp, log
from scipy.special import erf
import math
import glob
import json
import yaml
import re
from collections import OrderedDict
import matplotlib.colors as mcolors


import jax.numpy as jnp

from modules.globals import event_buffer, names

with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)
tiny = float(config['constants']['tiny'])
minCount = config['constants']['minCount']

class Counter:
    """counts time, communication- and update events, as well as np.random calls"""
    def __init__(self, n):
        self.n = n
    def __str__(self):
        return str(self.n)
    def inc(self):
        self.n += 1
    def dec(self):
        self.n -= 1 

class Message:
    """all details of a message are stored here"""
    def __init__(self, c, a, b_set, J, honest, blush, time, event_nr):
        self.time = time
        self.event_nr = event_nr
        self.c = c # topic of communication
        self.a = a # communicator
        self.b_set = b_set # addressat of communication
        self.J = J # message
        self.honest = honest
        self.blush = blush
    def __str__(self):
         return "Agent " + str(self.a) + " to " + \
            str(self.b_set) + " about "+str(self.c) + ": " +\
            str(self.J)#.rms)
    def val(self):
        return [self.c, self.a, self.b_set, self.J.mu, self.J.la]
    def message_dict(self):
        return {'time':int(self.time.n), 'EventNr':int(self.event_nr.n), 'a':int(self.a), 'b_set':list(self.b_set), 'c':int(self.c), 'J':self.J.to_list(), 'honest':self.honest, 'blush':self.blush}


class Comm_event: 
    """Each communication is stored as an event"""
    def __init__(self):
        # initialize them. values will be added later
        self.event_type = 'communication'
        self.comm = '' # dict
        
    def __str__(self):
        if self.comm['honest']:
            verb = ' tells   '
        else:
            if self.comm['blush']:
                verb = ' LIES to '
            else:
                verb = ' lies to '
        string = 'EventNr: '+str(self.comm['EventNr'])+', t = '+str(self.comm['time'])+': '+str(self.comm['a'])+verb+str(self.comm['b_set'])+' about '+str(self.comm['c'])+':  '+ \
                    str(self.comm['J'])
        return string

    def save(self, filenames):
        event_data = {**{'event_type': self.event_type}, **self.comm}
        event_buffer.append(event_data)

class Update_event:
    """Each update is stored as an event"""
    def __init__(self, id, time, EventNr):
        # initialize them. values will be added later
        self.event_type = 'update'
        self.update = {'time':time, 'EventNr':EventNr, 'id':id}

    def save(self, filenames):
        event_data = {**{'event_type': self.event_type}, **self.update}
        event_buffer.append(event_data)

def non_match_elements(list_a, list_b): # list_a is larger
    """searches for elements in list_a which are not in list_b"""
    non_match = []
    for i in list_a:
        if i not in list_b:
            non_match.append(i)
    return non_match

def convert_name(name_str):
    """takes a string in format 'Jothers_12_5' and returns [12, 5]"""
    strings = name_str.split('_')[1:] # ['12', '5']
    numbers = [int(strings[i]) for i in range(len(strings))]
    return numbers

def clean(inlist):
    """cleans list from nans: [nan, 3, 4, nan, nan, 1] -> [3,4,1]"""
    outlist = []
    for element in inlist:
        if np.isnan(element):
            pass
        else:
            outlist.append(element)
    return outlist

def get_folder(filename):
    """takes filename like "trash_folder/Exp1/Game0_flattering_NA3.json" and 
    returns its directory's name: "trash_folder/Exp1" """
    return filename.rpartition('/')[0] 
def get_name(filename):
    """takes filename like "trash_folder/Exp1/Game0_flattering_NA3.json" and 
    returns its name without the directory: "Game0_flattering_NA3.json" """
    return filename.rpartition('/')[2] 

def extract_mode(filename):
    # Extract the relevant part of the filename
    relevant_part = filename.split('/')[-1].split('_data_processed.hdf5')[0]
    
    # Split the relevant part by double underscores
    strategy_parts = relevant_part.split('__')
    
    mode = {}
    
    for part in strategy_parts:
        # Split each part by single underscores
        elements = part.split('_')
        
        # Extract the strategy name and agent numbers
        strategy = None
        for element in elements[1:]: # ignore strategy number
            if element.isdigit():
                agent_number = element
                mode[agent_number] = strategy
            elif element == 'all':
                mode['all'] = strategy
            else:
                strategy = element
    
    return mode

def overview(folder):
    """takes folder and analyzes the contained .json files. 
       returns Nstat, different modes and different NAs. Used for statistic runs.
       All files in the folder should have same Nstat"""

    # see if hdf5 files are already present
    hdf5_files = glob.glob(folder + "/*.hdf5", recursive = False)
    hdf5_files = [f for f in hdf5_files if 'data_processed' in f]
    if len(hdf5_files) > 0: # take them for overview
        # get modes
        modes = []
        for f in hdf5_files:
            mode = extract_mode(f)
            modes.append(mode)
        for mode in modes: # get Nstat, NA, NR
            mode_name_splits = [f"{names[strategy]}_{'_'.join([k for k,v in mode.items() if v == strategy])}" for strategy in set(mode.values())]
            mode_name = '__'.join(mode_name_splits)

            equal_filenames = []
            for f in hdf5_files:
                mode_specifying_name = f.split('_NA')[0].split('/')[-1]
                mode_specifying_name_splits = set(mode_specifying_name.split('__'))
                if mode_specifying_name_splits==set(mode_name_splits):
                    equal_filenames.append(f)

            filename = equal_filenames[0]
            df = pd.read_hdf(filename, 'general information')
            Nstat = int(df['Nstat'].head(1))
            NAs = [int(df['NA'].head(1))] # list of len 1
            NR = int(df['NR'].head(1))
            seeds = None


    else: # take raw data files
        all_files = glob.glob(folder + f"/*_RS*.json", recursive = False)
        assert len(all_files) != 0, f'No files were found. Make sure you have specified the right path and spelled it correctly. Folder: {folder}'
        modes = []
        NAs = []
        NR = 0
        seeds = []
        for F in all_files:
            with open(F, 'r') as f:
                first_line = json.loads(f.readline())
                modes.append(first_line['mode'])
                NAs.append(first_line['NA'])
                NR = first_line['NR']
                seeds.append(int(first_line['RSeed']))
        # assumption: all NAs/seeds/Nstat are equal for all modes
        modes = list(OrderedDict((tuple(d.items()), d) for d in modes).values())
        NAs = list(dict.fromkeys(NAs)) # remove duplicates, keeping the right order
        seeds = list(dict.fromkeys(seeds)) # remove duplicates, keeping the right order
        Nstat = int(len(all_files)/(len(modes)*len(NAs)))
    return Nstat, modes, NAs, NR, seeds

def prep_hlines(list):
    """preparation for histogram plots"""
    y = [list[0]] # hights of the lines
    xmin = [0] # beginnings of the lines
    xmax = [] # end of the lines
    for index in range(1, len(list)):
        if np.isnan(list[index]) and np.isnan(list[index-1]):
            pass # values are 'equal'
        elif list[index]!=list[index-1]:
            xmin.append(index)
            xmax.append(index-1)
            y.append(list[index])
    xmax.append(len(list)-2)
    return y, xmin, xmax

def rotate(point, origin, angle):
    """Rotate a point Counterclockwise by a given angle around a given origin.
    The angle should be given in radians."""
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]

def make_latex_str(string):
    """takes a string like 'any title $t$' and 
        returns a string in latex format like '$\mathrm{any\ title\ } t$'."""
    many_strings = string.split(' ')
    spaced_string = ''
    for s in many_strings:
        if '$' not in s:
            spaced_string += s + '\ '
    if '$' in string:
        for s in many_strings:
            if '$' in s:
                symbol = s
        latex_str = '$\mathrm{'+spaced_string+'} '+symbol[1:-1]+'$'
    else:
        spaced_string = spaced_string[:-2]
        latex_str = '$\mathrm{'+spaced_string+'}$'
    return latex_str

def map_to_01interval(x, mean, sigma):
    """maps x in [0,inf) to [0,1] using a sigmoid function f.
    parameters of the sigmoid function are defined by mu and sigma: f(mu)=0.5, f(mu+sigma)=0.84, f(mu-sigma)=0.16
    -> there are 68% of the points between f(mu+sigma) and f(mu-sigma)"""
    SIGMA = 0.5*(1-erf(1/np.sqrt(2))) # 0.16
    b = 1/sigma*np.log(1/SIGMA - 1)
    x_rescaled = 1/(1+np.exp(-b*(x-mean)))
    if x_rescaled<0 or x_rescaled>1:
        raise ValueError(f'rescaled number should be in the interval [0,1] but is {x_rescaled}!')
    if np.isnan(x_rescaled) and sigma==0 and x == mean: # only one point to determine shape of sigmoid
        return 0.5 # turning point of sigmoid
    else:
        return x_rescaled 

def average_sparse(list_of_df, start):
    """takes several dataframes and averaged over the 'what' column, i.e. still time-resolved.
    returns a sparse list [[times],[average]]"""
    times_lists = [list(list_of_df[i]['when']) for i in range(len(list_of_df))]
    times = [0]
    for l in times_lists:
        times += l
    times = list(set(times))
    times.sort()
    average = list()
    numbers_to_average = [start]*len(list_of_df)
    for t in times:
        for i in range(len(list_of_df)):
            df = list_of_df[i]
            try: 
                numbers_to_average[i] = np.float(df[df['when']==t]['what'])
            except:
                TypeError
        average.append(np.mean(numbers_to_average))
    return times, average

def make_float(x, default):
    try:
        x = float(x)
    except:
        TypeError
        if isinstance(x,pd.Series) and x.empty:
            return default
        else:
            raise TypeError(f"{x} can't be converted into float.")
    return x

def make_int(x, default):
    try:
        x = int(x)
    except:
        TypeError
        if isinstance(x,pd.Series) and x.empty:
            return default
        else:
            raise TypeError(f"{x} can't be converted into float.")
    return x

def abs_nan(x):
        if x > 0:
                return x
        elif x == 0:
            return np.nan
        else:
                return -x

def get_numbers(str):
    array = re.findall(r'[0-9]+', str)
    return array

def make_binned_averages(x_values, y_values, Min, Max, Nbins):
    """returns x and y values for Nbins bins.
    x_averages are the bin centers
    y_averages are the means of all datapoints within this bin"""

    assert len(x_values) == len(y_values)

    bin_edges = np.linspace(Min, Max, Nbins+1)
    bin_centers = np.array([(bin_edges[i] + bin_edges[i+1])/2 for i in range(Nbins)])

    av_y = [np.mean([y_values[i] for i in range(len(x_values)) if x_values[i]>=bin_edges[b] and x_values[i]<bin_edges[b+1]]) for b in range(Nbins)]

    # remove nans in empty bins -> plot will go on
    nan_indices = np.array([np.isnan(_) for _ in av_y])

    y_averages = np.array(av_y)[~nan_indices]
    x_averages = bin_centers[~nan_indices]

    return x_averages, y_averages

def makefolder(folder, subfolder):
    """makes a new folder/subfolder directory in case it doesn't already exist. Otherwise The existing subfolder will 
    renamed into subfolder_ and an empty (new) subfolder is generated."""
    try: # create folder
        os.mkdir(folder)  
    except OSError as error:  
        pass 
    else:
        print("Created folder "+folder)    

    try: # create new subfolder
        os.mkdir(os.path.join(folder,subfolder))  
    except OSError as error: # first rename the old and then create an new one
        n = 0
        while os.path.exists(os.path.join(folder,subfolder+'_'*n)):
            n = n+1
        os.rename(os.path.join(folder,subfolder), os.path.join(folder,subfolder+'_'*n))
        os.mkdir(os.path.join(folder,subfolder))
        print("Renamed old subfolder and created "+folder+'/'+subfolder)
    else:
        print("Created subfolder "+folder+'/'+subfolder)

def get_tbc_files(continue_folder, path_reload):
    # check that if continue_folder==False -> single file path must be specified
    # return 
    # either list of all files in this folder
    # or list with one file
    if not continue_folder:
        assert path_reload[-5:] == '.json', f'You wanted to continue a single file, but the given path does not seem to specify a json file: {path_reload}.'
        return [path_reload]
    else:
        # get all files in this folder
        all_files = glob.glob(path_reload + "/*.json", recursive = False)
        assert len(all_files)>0, f'There are no files in the folder {path_reload}. Make sure you specified the correct folder.'
        assert all([f[-5:]=='.json' for f in all_files])
        all_files = [f for f in all_files if not 'propaganda' in f]
        all_files = [f for f in all_files if not 'stat' in f.split('/')[-1]]
        return all_files

def read_and_get_parameters(old_file, new_NR):
    content = []
    with open(old_file, 'r') as old_f:
        # read the first line
        first_line = json.loads(old_f.readline()) # dict

        remaining_rounds = new_NR - first_line['NR']
        new_first_line = first_line
        new_first_line['NR'] = new_NR
        
        content.append(new_first_line)

        last_time = 0
        last_EventNr = 0

        line = old_f.readline()
        while line and json.loads(line)['event_type'] != 'final_status':
            line = json.loads(line)
            content.append(line)
            if line['event_type'] != 'initial_status':
                last_time = line['time']
                last_EventNr = line['EventNr']
            line = old_f.readline()
        
        # final status
        final_stati = []
        while line and json.loads(line)['event_type'] == 'final_status':
                line = json.loads(line)
                final_stati.append(line)
                line = old_f.readline()

        params = new_first_line
        params['last_time'] = last_time
        params['last_EventNr'] = last_EventNr

    return params, final_stati, content, remaining_rounds





def fc(x): # function to move singularity away
    return exp(x)+minCount -tiny # or with out -tiny

def invfc(fcount):
    return log(fcount-minCount+tiny)

def jfc(x): # function to move singularity away
    return jnp.exp(x)+minCount -tiny # or without -tiny

def jinvfc(fcount):
    return jnp.log(fcount-minCount+tiny)

def value_to_color(value):
    if np.isfinite(value):
        if value < 0.5:
            return mcolors.to_hex((1, 2 * value, 0))  # Red to Yellow
        else:
            return mcolors.to_hex((2 * (1 - value), 1, 0))  # Yellow to Green
    else:
        return mcolors.to_hex((1,1,1)) # white