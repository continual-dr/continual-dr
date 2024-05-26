# Continual Domain Randomization
This repository contains trained models and code to reproduce the results of the paper [Continual Domain Randomization](https://arxiv.org/pdf/2403.12193)

<p align="center">
  <img src="assets/cdr_sim2real_for_gif-opt.gif" width="620" alt="animated_overview" />
</p>

## Instructions
Prerequisites: Docker (and optionally [Nvidia-Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) if you have NVIDIA GPU). The default description below assumes that you have a Linux machine with a display. It is also possible to run the code under Windows or in headless mode with some restrictions; for details, see the Troubleshooting section.

The repository contains submodules, to clone everything at once use:

```git clone --recurse-submodules git@github.com:continual-dr/continual-dr.git```

All dependencies are provided through a docker image. To build the image, simply run the following command from the root folder of the repo:

```docker build -t continual-dr-img -f docker/Dockerfile .```

After building the image, you can create and start a container for running the code with:
```
./docker/run_local_gpu.sh
```


For running a specific experiment or evaluating a trained model, you should edit the following values in scripts/config.py:
```python
###################################### GENERAL PARAMETERS ##################################################
# the experiment type: reacher or grasper
'experiment_type': 'reacher',

# environment-specific parameter, e.g, number of controllable joints (should be 2 for reacher and 7 for grasper)
'number_joints': 2,
```

Then, to run the experiment from inside the container, run:

```python3 scripts/run_experiment.py```

This will start the experiment and start creating log data in the _experiments_/[reacher/grasper] folder. If you rerun the experiment, first delete the files/folders from a previous run.

## Model Evaluation

To check the performance of a single model, you can use the _evaluate_model.py_ script from the repository root folder and provide the model path through the -mp parameter, for example: 

```
python3 scripts/evaluate_model.py -mp experiments/reacher/experiment/continual_online/run1/NLT/Model/model_NJ2_STa_RT1_RN0_RL0_S31_PM1--000/rl_model_1000000_steps.zip
```

## Troubleshooting

#### The simulator only shows a gray screen

Upon simulator start, the default camera does not show any environment in order to avoid unnecessary rendering and computations. You can use the "Switch view" button to change to another camera and get a view of the environments. Additional cameras/views can be defined in the configuration.xml file (in experiments/configurations/ folder) by specifying x,y,z position in meters and x,y,z rotation in degrees. If you are starting the simulator manually, you should copy the correct configuration.xml file for either the grasper or the reacher in the same folder as the ManipulatorEnvironment executable. 

#### The simulator is not starting, and you get a failed connection error
If you are starting the simulator from inside the docker container (either manually, or through subprocess from scripts/simulator_vec_env.py), make sure that you allow connection to the xhost of the local machine from inside docker, by running the command ```xhost + local:``` before creating the container. For details, check the docker/commands file in this repo 

#### PermissionError: [Errno 13] Permission denied .../ManipulatorEnvironment.x86_64 
You need to give execution permissions to the file with ```chmod +x ManipulatorEnvironment.x86_64```

####  No module named 'stable_baselines3'
You might have cloned the repository without the --recursive-submodules flag, in which case the external/stable_baelines3 folder is empty, you can use the command below to additionally get the submodules:

```
git submodule update --init --recursive
```

#### Running on Windows

You can use Docker Desktop to run the code on Windows with some modifications:
- It is tricky and very slow to make the simulator open from inside the container in windowed mode (in order to be displayed on your monitor it should work inside a Linux virtual environment that uses WSL/HyperV). Still, you can use Docker and [WSL](https://docs.docker.com/desktop/wsl/) to enable GPU acceleration for pytorch inside the container, while the simulator should be started manually on your host machine and it will communicate with the container through a specified port. These are the steps:
- Assuming that you have installed Docker Desktop (and optionally WSL), after cloning the repository and the submodules and building the image as explained above, you can start a container with:
```
docker run  --rm -it --name continual-dr-container --gpus all -v ${PWD}:/home/continual-dr:rw --network host continual-dr-img:latest bash -c "cd  /home/continual-dr/ && bash"
``` 
- Next, copy the configuration.xml file for the experiment you want from the experiments/configurations/ folder and paste it (replace the existing one) in the
  _simulator\VTPRL\environment\simulator\v0.92cdr\Windows\ManipulatorEnvironment_ folder.
- Next, double-click on ManipulatorEnvironment.exe to open the simulator manually (Note: the simulator always reads the configuration from the configuration.xml file only upon start. Once it is started, it will open a GRPC server on the port specified in the configuration.xml file and wait for connections)
- Edit the scripts/config.py file by changing the ip address:
```python
# the ip address of the simulator, if you run the python code and the simulator inside docker it should be:
# 'localhost" for Linux
# 'host.docker.internal' for Windows
# an actual ip address of another machine if the simulator is running on a separate machine
'ip_address': 'host.docker.internal',
```
- Comment the automated opening of the simulator in _scripts/simulator_vec_env.py_:
```
# comment the sub-process creation, opening and sleep below if the simulator is manually started outside of the docker container
#self.env_process = subprocess.Popen(
#    [config['root_directory'] + "/simulator/v0.92/Linux/ManipulatorEnvironment/ManipulatorEnvironment.x86_64",
#     "-pn", str(config["port_number"]),
#     "-s", str(config["seed"]),
#     "-rt", str(0 if config["random_torque"] is False else 1),
#    "-logFile", config['working_dir'] + "/simulation.log",
#    # uncomment the line below for headless running of the simulator, note that there might be discrepency of the results between headless and windowed mode
#    #"-batchmode", "-nographics"
#    ],
#    stdout=PIPE, stderr=PIPE, stdin=PIPE,
#    cwd=config['root_directory'] + ("/experiments/configurations/reaching" if config['experiment_type'] == 'reacher' else
#    "/experiments/configurations/grasping"),
#    shell=False)
#atexit.register(kill_proc, self.env_process)
#time.sleep(10)
```
- After these steps, you can test the communication with the simulator by running a model evaluation from the container, e.g., after running the line below it should instantiate the environments in the simulator and run the evaluation without errors:
```
python3 scripts/evaluate_model.py -mp experiments/reacher/experiment/continual/run3/NLT/Model/model_NJ2_STa_RT1_RN0_RL0_S79_PM1--000/rl_model_1000000_steps.zip
```
- The logic of scripts/run_experiment.py won't work as it assumes that it can automatically open the simulators on the designated ports, for running experiments on Windows you would need to implement a manual logic (for example, from the randomizations only the torque is randomized on the simulator side, while the latency and the noise are randomized on the python side, so you can open manually two simulators by specifying different ports for them and passing the -tr command line parameter for one to have randomized torques. Then, you can use the different port numbers on the python side to connect to the simulators based on whether you are randomizing the torque or not). 

#### Running in headless mode

This mode works both on Linux and Windows with the automated start of the simulator from the scripts/simulator_vec_env.py script inside the container, so it is possible to run everything without the GUI of the simulator. However, there might be discrepancies in the results since the simulator behaves differently in headless mode. Therefore, we recommend always using a windowed mode of the simulator. In case you want to run the code on a server, ideally you should provide fake display/Xserver setup to enable windowed mode running. If you want to use headless mode, follow the default setup and just uncomment the _-batchmode and -nographics_ parameters in _scripts/simulator_vec_env.py_ script.

## Citations

If you use this work in your research, please cite our paper:
```
@article{josifovski_auddy2024continual,
  title={Continual Domain Randomization},
  author={Josifovski, Josip and Auddy, Sayantan and Malmir, Mohammadhossein and Piater, Justus and Knoll, Alois and Navarro-Guerrero, Nicol{\'a}s},
  journal={arXiv preprint arXiv:2403.12193},
  year={2024}
}
```
We are using the [VTPRL](https://github.com/tum-i6/VTPRL) simulator, in case you are using it in your work, please provide a reference to it.
