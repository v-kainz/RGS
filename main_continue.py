#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
import os
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from modules.helperfunc import get_tbc_files, read_and_get_parameters, makefolder


with open('config/config_continue.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

path_reload = os.path.join(*config['game']['path_reload'])

folder = config['game']['folder']
subfolder = config['game']['subfolder'] 
makefolder(folder, subfolder)

continue_folder = config['game']['continue_folder']
files_tbc = get_tbc_files(continue_folder, path_reload)

NR = config['game']['number_of_rounds']
Nstat_cont = config['game']['Nstat_cont']
RSeed_offset = config['randomseed']['RSeed_offset']
MULTIPROCESSING = config['switches']['MULTIPROCESSING']

honesties = config['agent_specification']['honesties']
mindI = config['agent_specification']['mindI']
Ks = config['agent_specification']['Ks']
prob_bts = config['agent_specification']['prob_bts']
prob_bls = config['agent_specification']['prob_bls']

# choose COMPRESSION_METHOD according to continued files
with open(files_tbc[0], 'r') as f:
    # read the first line
    first_line = json.loads(f.readline()) # dict
    COMPRESSION_METHOD = first_line['switches']['COMPRESSION_METHOD']
if COMPRESSION_METHOD == 'LUT':
    from modules.globals import init_LUT
    init_LUT(load=True)
from modules.game import continue_game
from modules.setup import simulation_config_from_parameters
from modules.evaluation_sg import evaluate_stat as evaluate_sg_stat

files_tbc_seed = []
for seed in range(Nstat_cont):
    for f in files_tbc:
        files_tbc_seed.append((f, seed+RSeed_offset))

if __name__ == "__main__":
    if MULTIPROCESSING:
        # get the amount of CPUs for multiprocessing
        N_cores = multiprocessing.cpu_count()
        # separate the simulation jobs into batches, which are multiples of the core amount
        multiple = int(round(len(files_tbc_seed)/(N_cores), 0))
        # rest jobs
        if multiple * N_cores < len(files_tbc_seed):
            rest_jobs = len(files_tbc_seed) - N_cores * multiple
        else:
            rest_jobs = 0
        # multiprocessing executor
        executor = ProcessPoolExecutor(N_cores)
        for i in range(multiple):
            file_seed_tuples_multi = files_tbc_seed[i*N_cores: (1+i)*N_cores]
            sim_list = []
            for t in file_seed_tuples_multi:
                params, final_stati, content, remaining_rounds = read_and_get_parameters(t[0], NR)
                sim = simulation_config_from_parameters(NR, folder, subfolder, params, t[1])
                args = [sim, remaining_rounds, params, honesties, mindI, Ks, prob_bts, prob_bls, final_stati, content]
                sim_list.append(args)
            
            results = executor.map(continue_game, sim_list) # play game
            try: # Catch any errors inside the multiprocessing worker. Otherwise it stays silent.
                for result in results:
                    pass
            except Exception as e:
                raise e
        # do the rest jobs which were the remainder of the division by the amount of cores
        file_seed_tuples_single = files_tbc_seed[multiple*N_cores: multiple*N_cores + rest_jobs]
        print('single: ', file_seed_tuples_single)
        sim_list = []
        for t in file_seed_tuples_single:
            params, final_stati, content, remaining_rounds = read_and_get_parameters(t[0], NR)
            sim = simulation_config_from_parameters(NR, folder, subfolder, params, t[1])
            args = [sim, remaining_rounds, params, honesties, mindI, Ks, prob_bts, prob_bls, final_stati, content]
            sim_list.append(args)
        results = executor.map(continue_game, sim_list)
        try:
            for result in results:
                pass
        except Exception as e:
                raise e
    else: # if multiprocessing is switsched off
        for t in files_tbc_seed:
            params, final_stati, content, remaining_rounds = read_and_get_parameters(t[0], NR)
            sim = simulation_config_from_parameters(NR, folder, subfolder, params, t[1])
            continue_game([sim, remaining_rounds, params, honesties, mindI, Ks, prob_bts, prob_bls, final_stati, content]) # play game

    default_single = config['evaluation']['evaluate_single_runs']
    default_stat = config['evaluation']['evaluate_statistics']
    if default_single or default_stat:
        evaluate_sg_stat(os.path.join(folder, subfolder), default_single=default_single, default_stat=default_stat)