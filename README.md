# IR-Benchmark

This repository contains the code for the submission to the KDD 2025.

## Installation

We provide a `environment.yml` file to create a conda environment with all the dependencies. To create the environment, run the following command:

```bash
conda env create -f environment.yml
```

This will create a conda environment named `itemrec`. To activate the environment, run:

```bash
conda activate itemrec
```

## Code Structure

The code is organized as follows:

```
IR-Benchmark-SLatK
│   README.md                           # This file
│   environment.yml                     # Conda environment file
|   run_nni.py                          # NNI hyperparameter tuning script
│   data                                # IID datasets
│   │   gowalla                         # Gowalla IID dataset
|   |   ... (other datasets)            # Other IID datasets
│   itemrec                             # Main package
|   |   __main__.py                     # Main script to run 
|   |   cli.py                          # CLI
|   |   hyper.py                        # NNI hyperparameter tuning
|   |   args.py                         # Argument parsing
|   |   ... (other modules)             # Other modules
```

## CLI

IR-Benchmark provides a CLI to run the experiments. To see the available commands, run:

```bash
python -m itemrec --help
```

In general, the CLI follows the following structure:

```bash
python -u -m itemrec [-h] [-v] --log LOG --save_dir SAVE_DIR --seed SEED 
model [--model_args ...] dataset [--dataset_args ...] optim [--optim_args ...]
```

where `model`, `dataset`, and `optim` are the subcommands to specify the model, dataset, and optimization algorithm, respectively. Each subcommand has its own set of arguments. Please see the help message or `itemrec/args.py` for more information.


## NNI Hyperparameter Tuning

A more easy way to run the code is to use our hyperparameter tuning script, i.e., `./run_nni.py`. This script uses the NNI framework to run hyperparameter tuning experiments. You only need to modify the following paths in the script:

```python
# main function -----------------------------------------------------
def main():
    args = parse_args()
    # TODO: /path/to/your/ must be replaced with the actual paths
    save_dir = f"/path/to/your/logs/{args.dataset}/{args.model}/{args.optim}"
    if not args.ood:
        dataset_path = f"/path/to/your/data/{args.dataset}/proc"
    else:
        dataset_path = f"/path/to/your/data_ood/{args.dataset}/proc"
    ...
    # NNI experiment
    ...
    # TODO: /path/to/your/code must be replaced with the actual path
    experiment.config.trial_code_directory = '/path/to/your/code'
    # TODO: specify the port and GPU
    experiment.config.training_service.platform = 'local'
    experiment.config.training_service.use_active_gpu = True
    experiment.config.training_service.max_trial_number_per_gpu = 2
    experiment.config.training_service.gpu_indices = [0, 1, 2, 3]
    ...
    experiment.run(args.port)
```

`run_nni.py` also provides a CLI to run the code, which is more user-friendly than the main CLI, since most the arguments are automatically set by the script. Please see the arguments of the script for more information.
