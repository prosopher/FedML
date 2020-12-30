#!/bin/bash

set -ex

# code checking
# pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login 83340ef4e39f0875b49b76cf40ca51d5cc7a6b7e
#wandb off

cd ./fedml_experiments/standalone/hierarchical_fl
# FedAvg-r10-e50
# sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 1 10 1 50

# sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 1 10 50
# sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 5 2 50
# sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 10 1 50
# sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 10 5 10
sh run_standalone_pytorch.sh 0 1000 100 10 mnist ./../../../data/mnist lr hetero 0.03 sgd random 10 10 50 1
cd ./../../../