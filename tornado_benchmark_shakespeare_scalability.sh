#!/bin/bash

set -ex

# code checking
# pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login 83340ef4e39f0875b49b76cf40ca51d5cc7a6b7e
#wandb off

cd ./fedml_experiments/standalone/tornado

# STAR (FedAvg)
if [ "$1" = "FedAvg" ]; then
sh run_standalone_pytorch.sh $2 1000 $4 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR stars random 1 1 10 1 100

# Tornado
elif [ "$1" = "Tornado" ]; then
sh run_standalone_pytorch.sh $2 1000 $4 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 RING stars iid 2 2 10 10 10

# Tornadoes
elif [ "$1" = "Tornadoes" ]; then
sh run_standalone_pytorch.sh $2 1000 $4 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR rings cluster 5 5 10 10 10

else
echo "$1"
fi

cd ./../../../
