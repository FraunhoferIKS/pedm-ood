# PEDM OOD

Repository for the paper: [Out-of-Distribution Detection for Reinforcement Learning Agents with Probabilistic Ensemble Dynamics Model](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p851.pdf)" presented at [AAMAS 2023](https://aamas2023.soton.ac.uk/).
 
## Structure of this repo

        └──this repo 
             ├── data
             │   ├── checkpoints        -> checkpoints for models (TD3) 
             │   └── supplementary      -> supplementary material for the paper 
             ├── mujoco_envs            -> implementation of some mujoco envs with disturbances
             ├── ood_baselines          -> ood detection baselines
             ├── pedm                   -> everything for the Probabilistic Ensemble Dynamics Model(PEDM), e.g. neural-network models, dynamics models, ablated versions
             └── utils                  -> callbacks, wrappers, statistics, gpu_utils, ...


## Requirements

install python packages: 

        pip install -r requirements.txt 

- Recommended python version: 3.8.10
- Recommended OS: Ubuntu 20.04.4 
- Recommended Mujoco Version: 2.1 
   - You need mujuco already installed. Download it [here](https://github.com/deepmind/mujoco/releases). For versions prior to 2.1 get your key [here](https://roboti.us/license.html). See the [Dockerfile](./Dockerfile) for more details on how to install it. 


## Running Experiments

- oodd_runner.py: main script to train ood_detector on nominal (non-disturbed) environments and test it on disturbed environments.
  - Example usage:

        python oodd_runner.py --env_id MJPusher-v0 --detector_name PEDM_Detector --test_episodes 100 --mods "['act_factor_severe']" --experiment_tag "test_pusher"

        
  - arguments:

        -h, --help                      -> show this help message and exit
        --env_id                        -> which env to run on; choose from {MJCartpole-v0,MJHalfCheetah-v0,MJReacher-v0,MJPusher-v0}
        --n_train_episodes              -> number of training episodes to use for training the detector (if applicable)
        --test_episodes                 -> number of evaluation episodes to test the detectr
        --policy_name                   -> name/type of the policy that interacts with the env
        --policy_path                   -> where to find this policy
        --mods                          -> which mods to evaluate on, e.g. type: << --mods "['act_factor_severe']" >>; if not provided, will run all mods
        --data_path                     -> path to the databuffer if existing, if None will look at default location
        --data_tag                      -> tag for identifying the databuffer
        --detector_name                 -> class/type of the detector to use
        --detector_kwargs               -> kwargs for the constructor of the detector
        --detector_tag                  -> tag for identifying the detector
        --detector_path                 -> path to the model of the detector (if applicable)
        --detector_fit_kwargs           -> kwargs for the training loop of the detector
        --results_save_dir              -> where to save all results
        --experiment_tag                -> tag for identifying the experiment
        --device                        -> which device to use, cuda recommended!

- utils/train_mf.py: script to train a model-free policy that interacts with an env. Example usage:

        python train_policy.py --env_id MJHalfCheetah-v0 --policy_name TD3 --train_steps 10_000_000 --exp_name train_td3_hc


## Disclaimer
This software was solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## Acknowledgement
The base-environments used in this project are taken from: https://github.com/kchua/handful-of-trials/tree/master/dmbrl/env
The MIT License from this repo applies only to these environemnts. 

## Citing the project:

If you find our work useful in your research, please consider citing:

        @inproceedings{haider2023out,
                       title={Out-of-Distribution Detection for Reinforcement Learning Agents with Probabilistic Dynamics Models},
                       author={Haider, Tom and Roscher, Karsten and Schmoeller da Roza, Felippe and G{\"u}nnemann, Stephan},
                       booktitle={Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems},
                       pages={851--859},
                       year={2023}
        }
