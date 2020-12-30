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

# CL
if [ "$1" = "CL" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR stars random 1 1 4000 1 1

# STAR (FedAvg)
elif [ "$1" = "FedAvg" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR stars random 1 1 40 1 100

# stars (IFCA)
elif [ "$1" = "IFCA" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 PLURAL stars cluster 1 10 40 10 10

# rings (SemiCyclic)
elif [ "$1" = "SemiCyclic" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 PLURAL rings random 1 5 40 10 10

# STAR-stars (HierFAVG)
elif [ "$1" = "HierFAVG" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR stars random 1 5 40 10 10

# STAR-rings (Astraea)
elif [ "$1" = "Astraea" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR rings iid 1 2 40 10 10

# RING-stars (MM-PSGD)
elif [ "$1" = "MM-PSGD" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 RING stars cluster 1 10 40 10 10

# Tornado
elif [ "$1" = "Tornado" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 RING stars iid 2 2 40 10 10

# Tornadoes
elif [ "$1" = "Tornadoes" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR rings cluster 10 10 40 10 10

# Tornado-rings
elif [ "$1" = "Tornado-rings" ]; then
sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 PLURAL rings cluster 10 10 40 10 10

else
echo "$1"
fi



# RING
# sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 STAR rings random 1 1 40 1 100

# RING-rings
# sh run_standalone_pytorch.sh $2 1000 100 10 fed_shakespeare ./../../../data/fed_shakespeare rnn hetero 0.03 sgd $3 RING rings random 1 5 40 10 10
cd ./../../../
