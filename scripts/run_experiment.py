import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import subprocess
import itertools
import time
from utils import log
from config import Config

config = Config.get_config_dict()

RUN_IDENTIFIERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEEDS = [17, 31, 62, 79, 94, 117, 131, 147, 169, 178, 199]
STRATEGIES = [
    "ideal",    
    "randomized",
    "continual",
    "continual_online",
    "finetuning",
    ]
RANDOMIZATIONS = ["T", "L", "N"]

TRAINING_TIMESTEPS = 2_000_000 if config['experiment_type'] == 'grasper' else 1_000_000
STARTING_TIMESTEP_PRETRAINED_MODEL = 4_000_000 if config['experiment_type'] == 'grasper' else 1_000_000
SPECIFIC_SEQUENCES = [0, 5]

# set SPECIFIC_RUN to None if you want to run all seeds
SPECIFIC_RUN = 1

for CURRENT_RUN in RUN_IDENTIFIERS:
        for CURRENT_STRATEGY in STRATEGIES:
            if SPECIFIC_RUN is None or SPECIFIC_RUN == CURRENT_RUN:
                run_folder = f"{config['root_directory']}/experiments/{config['experiment_type']}/{CURRENT_STRATEGY}/run{CURRENT_RUN}"
                print("creating experiment run folder")
                os.makedirs(run_folder, exist_ok=True)
                if CURRENT_STRATEGY == "ideal" or CURRENT_STRATEGY == "randomized":
                    for part in range(0,4):
                        if (part == 0):
                            previous_model = ""
                        else:
                            rt, rn, rl = (0, 0, 0) if CURRENT_STRATEGY == "ideal" else (1, 1, 1) 
                            previous_model = f"{config['root_directory']}/experiments/{config['experiment_type']}/{CURRENT_STRATEGY}/run{CURRENT_RUN}" + \
                            f"/Model/model_NJ{config['number_joints']}_STa_RT{rt}_RN{rn}_RL{rl}_S{SEEDS[CURRENT_RUN]}" + \
                            f"_PM{1 if part > 1 else 0}--00{0 if part < 3 else 1}/rl_model_{STARTING_TIMESTEP_PRETRAINED_MODEL if part == 1 else TRAINING_TIMESTEPS}_steps"
                        command = ["python3",
                                   "scripts/main.py",
                                   "-m", "1",
                                   "-st", CURRENT_STRATEGY,
                                   "-rl", str(0 if CURRENT_STRATEGY == "ideal" else 1),
                                   "-rt",  str(0 if CURRENT_STRATEGY == "ideal" else 1),
                                   "-rn", str(0 if CURRENT_STRATEGY == "ideal" else 1),
                                   "-n_envs", str(config["n_envs"]),
                                   "-nt", str(STARTING_TIMESTEP_PRETRAINED_MODEL if part == 0 else TRAINING_TIMESTEPS),
                                   "-mn", previous_model,
                                   "-l", run_folder,
                                   "-s", str(SEEDS[CURRENT_RUN]),
                                   "-pn", str(9090 + CURRENT_RUN + (0 if CURRENT_STRATEGY == "ideal" else 6)),
                                   ]
                        log(str(command), global_log=True)
                        with open(run_folder + "/stdout" + str(part) + ".txt", "wb") as out, open(run_folder + "/stderr" + str(part) + ".txt", "wb") as err:
                            print("starting run in folder %s"%(run_folder))
                            subprocess.call(command, stdout=out, stderr=err)
                        time.sleep(20)
                    
                if CURRENT_STRATEGY in ["finetuning", "continual", "continual_online"]:
                    for sequence_number, sequence in enumerate(itertools.permutations(RANDOMIZATIONS)):
                        previous_model = f"{config['root_directory']}/experiments/{config['experiment_type']}/ideal/run{CURRENT_RUN}" + \
                            f"/Model/model_NJ{config['number_joints']}_STa_RT0_RN0_RL0_S{SEEDS[CURRENT_RUN]}_PM0--000" + \
                            f"/rl_model_{STARTING_TIMESTEP_PRETRAINED_MODEL}_steps"
                        if SPECIFIC_SEQUENCES is None or sequence_number in SPECIFIC_SEQUENCES:
                                for index, randomization_parameter in enumerate(sequence):
                                    rl = 1 if randomization_parameter == "L" else 0
                                    rt = 1 if randomization_parameter == "T" else 0
                                    rn = 1 if randomization_parameter == "N" else 0
                                    command = ["python3",
                                        "scripts/main.py",
                                        "-m", "1",
                                        "-stg", CURRENT_STRATEGY,
                                        "-ind", str(index + 1), # the first task (0) is the ideal simulation
                                        "-rl", str(rl),
                                        "-rt", str(rt),
                                        "-rn", str(rn),
                                        "-el", str(config["ewc_lambda"] if CURRENT_STRATEGY != "finetuning" else 0),
                                        "-n_envs", str(config["n_envs"]),
                                        "-mn", previous_model,
                                        "-nt", str(TRAINING_TIMESTEPS),
                                        "-l", run_folder + "/" + "".join(sequence),
                                        "-s", str(SEEDS[CURRENT_RUN]),
                                        "-pn", str(9000 + CURRENT_RUN * 100 + sequence_number * 10 + index + (
                                        0 if CURRENT_STRATEGY == "finetuning" else 4 if CURRENT_STRATEGY == "continual" else 8)),
                                        ]
                                    log(str(command), global_log=True)
                                    with open(run_folder + "/" + "".join(sequence) + "_stdout" + str(index+1) + ".txt", "wb") as out, open(run_folder + "/" +  "".join(sequence) + "_stderr" + str(index+1) + ".txt", "wb") as err:
                                        print("starting run in folder %s for sequence %s"%(run_folder, sequence))
                                        subprocess.call(command, stdout=out, stderr=err)
                                    time.sleep(20)
                                    previous_model = f"{run_folder}/{''.join(sequence)}/Model/model_NJ{config['number_joints']}_STa" + \
                                        f"_RT{rt}_RN{rn}_RL{rl}_S{SEEDS[CURRENT_RUN]}_PM{0 if previous_model == '' else 1}--000/rl_model_{TRAINING_TIMESTEPS}_steps"
        
print("exiting")
