#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml
import os
from timeit import default_timer as timer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

with open('config/config.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

if config['switches']['COMPRESSION_METHOD'] == 'LUT':
    from modules.globals import init_LUT
    init_LUT(load=True)

from modules.setup import simulation_config
from modules.informationtheory import Info
from modules.helperfunc import makefolder

def run_main():  
    from modules.game import play_game
    from modules.evaluation_sg import evaluate_stat as evaluate_sg_stat 

    start_time = timer()
    Nstat = config['game']['statistics_size'] 
    NA = config['game']['number_of_agents']
    NR = config['game']['number_of_rounds']
    RSeed_offset = config['randomseed']['RSeed_offset']
    modes = config['game']['modes']
    honesties = config['agent_specification']['honesties']
    mindI = config['agent_specification']['mindI']
    Ks = config['agent_specification']['Ks']
    prob_bts = config['agent_specification']['prob_bts']
    prob_bls = config['agent_specification']['prob_bls']

    folder = config['game']['folder']
    subfolder = config['game']['subfolder'] 
    makefolder(folder, subfolder)

    if config['switches']['MULTIPROCESSING']:    
        # get the amount of CPUs for multiprocessing
        N_cores = multiprocessing.cpu_count()
        # separate the simulation jobs into batches, which are multiples of the core amount
        multiple = int(round(Nstat/(2*N_cores), 0))
        # rest jobs
        if multiple * N_cores < Nstat:
            rest_jobs = Nstat - N_cores * multiple
        else:
            rest_jobs = 0
        # multiprocessing executor
        executor = ProcessPoolExecutor(N_cores*2)
        for run in range(multiple):
            for mode in modes:
                simulation_l = [simulation_config(NA, NR, mode, honesties, mindI, Ks, prob_bts, prob_bls, RSeed + RSeed_offset, folder, subfolder, sort_by_honesty=config['switches']['sort_by_honesty'])
                                                    for RSeed in range(run*N_cores, (1+run)*N_cores)]
                results = executor.map(play_game, simulation_l)
                try: # Catch any errors inside the multiprocessing worker. Otherwise it stays silent.
                    for result in results:
                        pass
                except Exception as e:
                    raise e
        # do the rest jobs which were the remainder of the division by the amount of cores
        for mode in modes:
            simulation_l = [simulation_config(NA, NR, mode, honesties, mindI, Ks, prob_bts, prob_bls, RSeed + RSeed_offset, folder, subfolder, sort_by_honesty=config['switches']['sort_by_honesty'])
                            for RSeed in range(multiple*N_cores, multiple*N_cores + rest_jobs)]
            results = executor.map(play_game, simulation_l)
            try:
                for result in results:
                    pass
            except Exception as e:
                raise e
    else: # if multiprocessing is switsched off
        for mode in modes:
            for RSeed in range(Nstat):
                simulation = simulation_config(NA, NR, mode, honesties, mindI, Ks, prob_bts, prob_bls, int(RSeed + RSeed_offset), folder = folder, subfolder = subfolder, sort_by_honesty=config['switches']['sort_by_honesty']) # set up game
                play_game(simulation)


    # Evaluation
    default_single = config['evaluation']['evaluate_single_runs']
    default_stat = config['evaluation']['evaluate_statistics']
    if default_single or default_stat:
        evaluate_sg_stat(os.path.join(folder, subfolder), default_single=default_single, default_stat=default_stat)

    end_time = timer()
    print(f'time elapsed: {round(end_time - start_time)}s')

def run_propaganda(prop):
    from modules.game import play_propaganda, play_antipropagangda
    from modules.evaluation_sg import evaluate_prop

    NR = config['game']['number_of_rounds']
    modes = config['game']['modes']
    RSeed_offset = config['randomseed']['RSeed_offset']
    honesties = config['agent_specification']['honesties']

    folder = config['game']['folder']
    subfolder = config['game']['subfolder'] 
    makefolder(folder, subfolder)

    Ip = Info(1000,0) # propaganda information

    for mode in modes:
        NA = 4
        sim = simulation_config(NA, NR, mode, honesties, RSeed=0 + RSeed_offset, folder=folder, subfolder=subfolder, prop=prop)
        if prop == 1: 
            play_propaganda(sim, Ip)
        elif prop == -1:
            play_antipropagangda(sim, Ip)
        else:
            raise ValueError(f'Propaganda {prop} not found.')
        evaluate_prop(sim.filenames['outfile'], prop)

if __name__ == "__main__":
    if config['switches']['propaganda'] == 0:
        run_main()
    elif config['switches']['propaganda'] == 1:
        run_propaganda(1)
    elif config['switches']['propaganda'] == -1:
        run_propaganda(-1)
    else:
        raise ValueError(f"Unknown propaganda specification: {config['switches']['propaganda']}.")