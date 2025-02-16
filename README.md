# Learning-based Collaborative Vehicle Routing
The directory contains the dataset, code, and results for Huang, X (2023) *Learning to Distribute for Centralised Collaborative Vehicle Routing*. MEng Final Year Project. Imperial College London. 

The study proposes a learning-based framework to tackle the collaborative vehicle routing problem in the form of multi-depot vehicle routing problem (MDVRP). Genetic algorithm and nearest neighbourhood insertion together
with the learned model are hybridised to solve the problem in two stages. We also compare the proposed framework to benchmark methods of hybrid genetic algorithm and nearest insertion heuristic.

## Overview
Our learning-based framework runs the following scripts in the directory `exps/learning_cvrp` to generate data, train the model, and obtain numerical results
1. `generate_init.py`: Generate training and validation data consist of subproblems with allocation/cost pairs. Each subproblem is solved by the nearest insertion heuristic
2. `process_assignment.py`: Process and concatenate generated results of different instances into a single file to facilitate the training process
3. `train.py`: Train subproblem data to and get the best model with the lowest cost in the validation dataset
4. `main.py`: Get results from the proposed and two benchmark methods

The following files are used in hybrid with our learned model:
1. `exps/learning_cvrp/ga.py`: A genetic algorithm that employs the learned model as the fitness function to explore a wider search range of subproblems
2. `vrpkit/vrp/insertion.py`: A nearest insertion heuristic to obtain labels for the subproblems used for training

The following files in the directory `vrpkit/vrp` are used to generate solutions for two benchmark methods in solving MDVRP:
1. `hga.py`: A hybrid genetic algorithm proposed by [Zhang *et al.* (2022)](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1473.pdf)
2. `insertion.py`: A nearest insertion heuristic

## Setup
We implement our framework using Pytorch in Python 3.9. The environment can be set up through the following command:
`conda env create environment.yml`

## Instruction
The following sections describe the command to run in terminal to execute the files.
### `generate_init.py`
The total time required for label generation depend on the sample size `N`. It takes approximately 0.1 second to generate a label for an allocation. 
```commandline
export INSTANCE_NAME=benchmark_2D #options:[benchmark_2D, benchmark_4D]
export N=20000
export DATASET_DIR=data/instances/$INSTANCE_NAME
export SAVE_DIR=data/generations/$INSTANCE_NAME/sample$N
python -m expt.learning_cvrp.generate_init $DATASET_DIR $SAVE_DIR --n_sample $N --instance_name $INSTANCE_NAME
```
### `process_assignment.py`
We run the following command three times to process the generated results for training, testing, and validation datasets.
```commandline
export PARTITION=train #options: [train, test, val]
export python -m expt.learning_cvrp.process_assignment $SAVE_DIR --partition test 
```
### `train.py`
We present the hyperparameters used in our study as arguments.
```commandline
export TRAIN_DIR=exps/$INSTANCE_NAME/sample$N 
python -m expt.learning_cvrp.train $SAVE_DIR $TRAIN_DIR --fit_subproblem --augment_rotate  --augment_flip --lr 0.001 --n_layers 3 --transformer_heads 4 --n_steps 20000 --d_hidden 64 --loss MAE --n_batch 2048
```
### `main.py`
We solve each instance using both proposed and benchmark methods for 50 times
```commandline
python -m expt.learning_cvrp.main $DATASET_DIR $INSTANCE_NAME $TRAIN_DIR --lga_nni --hga --nih --n_gen 50
```
