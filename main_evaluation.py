import os
import yaml
from modules.prepare_evaluation import prepare_evaluation
from modules.evaluation_sg import evaluate as evaluate_sg
from modules.evaluation_sg import evaluate_stat as evaluate_sg_stat

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

with open('config/config_evaluation.yml', 'r') as config_file:
    config = yaml.safe_load(config_file)

plot_selection_sg = config['plot_selection_sg']
new_evaluation = config['new_evaluation']
MULTIPROCESSING = config['MULTIPROCESSING']


def evaluate_path(path_to_evaluate, config, plot_selection_sg, new_evaluation):
    if len(config['files_to_evaluate']) > 0:
        for f in config['files_to_evaluate']:
            path = os.path.join(*config['path_to_evaluate'], f)
            evaluate_sg(path, plot_selection=plot_selection_sg['single_plots'])
    else:
        evaluate_sg_stat(path_to_evaluate, 
                            plot_selection_stat=plot_selection_sg['stat_plots'], 
                            plot_selection_comb_stat=plot_selection_sg['comb_stat_plots'], 
                            plot_selection_single=plot_selection_sg['single_plots'], 
                            default_single=True, 
                            new_evaluation=new_evaluation)

def main(paths_to_evaluate, config, plot_selection_sg, new_evaluation):
    if MULTIPROCESSING:
        # Get the amount of CPUs for multiprocessing
        N_cores = multiprocessing.cpu_count()
        # Separate the evaluation jobs into batches, which are multiples of the core amount
        multiple = int(round(len(paths_to_evaluate) / (N_cores), 0))
        # Rest jobs
        if multiple * N_cores < len(paths_to_evaluate):
            rest_jobs = len(paths_to_evaluate) - N_cores * multiple
        else:
            rest_jobs = 0

        # Multiprocessing executor
        executor = ProcessPoolExecutor(N_cores)
        
        # Process the main batches
        for run in range(multiple):
            batch_paths = paths_to_evaluate[run * N_cores : (run + 1) * N_cores]
            results = executor.map(evaluate_path, batch_paths, [config]*len(batch_paths), [plot_selection_sg]*len(batch_paths), [new_evaluation]*len(batch_paths))
            try:
                for result in results:
                    pass
            except Exception as e:
                raise e

        # Process the rest jobs which were the remainder of the division by the amount of cores
        if rest_jobs > 0:
            rest_paths = paths_to_evaluate[multiple * N_cores : multiple * N_cores + rest_jobs]
            results = executor.map(evaluate_path, rest_paths, [config]*len(rest_paths), [plot_selection_sg]*len(rest_paths), [new_evaluation]*len(rest_paths))
            try:
                for result in results:
                    pass
            except Exception as e:
                raise e
    else:
        for path_to_evaluate in paths_to_evaluate:
            if len(config['files_to_evaluate']) > 0:
                path_to_evaluate = [config['path_to_evaluate'][0], config['path_to_evaluate'][1][0]]
                for f in config['files_to_evaluate']:
                    path = os.path.join(*path_to_evaluate, f)
                    evaluate_sg(path, plot_selection = plot_selection_sg['single_plots'])
            else:
                evaluate_sg_stat(path_to_evaluate, plot_selection_stat = plot_selection_sg['stat_plots'], plot_selection_comb_stat = plot_selection_sg['comb_stat_plots'], plot_selection_single=plot_selection_sg['single_plots'], default_single=True, new_evaluation=new_evaluation)


if __name__ == "__main__":
    #evaluate_main()
    folder = config['path_to_evaluate'][0]
    subfolders = config['path_to_evaluate'][1]
    paths_to_evaluate = [os.path.join(folder, subfolder) for subfolder in subfolders]

    main(paths_to_evaluate, config, plot_selection_sg, new_evaluation)

    
