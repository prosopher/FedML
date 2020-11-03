import argparse
import logging
import os
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.standalone.tornado.star_trainer import StarTrainer
from fedml_api.standalone.tornado.ring_trainer import RingTrainer
from fedml_api.standalone.tornado.pluralistic_trainer import PluralisticTrainer
from fedml_experiments.standalone.fedavg.main_fedavg import add_args, load_data, create_model

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='Tornado-standalone'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--global_topology', type=str, default='star', choices=['star', 'ring', 'pluralistic'])
    parser.add_argument('--group_topology', type=str, default='star', choices=['star', 'ring'])
    parser.add_argument('--group_method', type=str, default='random', choices=['random', 'cluster', 'iid'])
    parser.add_argument('--chain_num', type=int, default=1, metavar='N', help='the number of concurrent chains')
    parser.add_argument('--group_num', type=int, default=1, metavar='N', help='the number of groups')
    parser.add_argument('--global_comm_round', type=int, default=10, help='the number of global communications')
    parser.add_argument('--group_comm_round', type=int, default=10,
                        help='the number of group communications within a global communication')
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    alg_name = args.global_topology[0].upper() + args.global_topology[1:len(args.global_topology)] + \
                args.group_topology[0].upper() + args.group_topology[1:len(args.group_topology)]
    wandb.init(
        project="fedml",
        name="{}-{}-{}-{}-{}-{}".format(alg_name, args.chain_num, args.group_num, args.global_comm_round, args.group_comm_round, args.epochs),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    logging.info(model)

    if args.global_topology == 'star':
        trainer = StarTrainer(dataset, model, device, args)
    elif args.global_topology == 'ring':
        trainer = RingTrainer(dataset, model, device, args)
    elif args.global_topology == 'pluralistic':
        trainer = PluralisticTrainer(dataset, model, device, args)
    else:
        raise Exception(args.global_topology)
    trainer.train()
