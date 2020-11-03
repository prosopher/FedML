import torch
import wandb
import logging
import numpy as np

from fedml_api.standalone.tornado.star_group import StarGroup
from fedml_api.standalone.tornado.ring_group import RingGroup
from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.hierarchical_fl.trainer import Trainer as HierarchicalTrainer

class ConsensusTrainer(HierarchicalTrainer):

    def __init__(self, dataset, model, device, args):
        super().__init__(dataset, model, device, args)

        self.c_train_data_local_num_dict = {0: sum(user_train_data_num for user_train_data_num in self.train_data_local_num_dict.values())}
        self.c_train_data_local_dict = {0: [batch for cid in sorted(self.train_data_local_dict.keys()) for batch in self.train_data_local_dict[cid]]}
        self.c_test_data_local_dict = {0: [batch for cid in sorted(self.test_data_local_dict.keys()) for batch in self.test_data_local_dict[cid]]}
        self.c_train_data_local_dict = {cid: self.combine_batches(self.c_train_data_local_dict[cid]) for cid in self.c_train_data_local_dict.keys()}
        self.c_test_data_local_dict = {cid: self.combine_batches(self.c_test_data_local_dict[cid]) for cid in self.c_test_data_local_dict.keys()}

    def combine_batches(self, batches):
        full_x = torch.from_numpy(np.asarray([])).float()
        full_y = torch.from_numpy(np.asarray([])).long()
        for (batched_x, batched_y) in batches:
            full_x = torch.cat((full_x, batched_x), 0)
            full_y = torch.cat((full_y, batched_y), 0)
        return [(full_x, full_y)]

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == 'random':
            self.group_indexes = np.random.randint(0, self.args.group_num, self.args.client_num_in_total)
            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
        elif self.args.group_method == 'cluster':
            pass
        elif self.args.group_method == 'iid':
            pass
        else:
            raise Exception(self.args.group_method)

        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            if self.args.group_topology == 'star':
                self.group_dict[group_idx] = StarGroup(group_idx, client_indexes, train_data_local_dict, test_data_local_dict,
                                                       train_data_local_num_dict, self.args, self.device, self.model)
            elif self.args.group_topology == 'ring':
                self.group_dict[group_idx] = RingGroup(group_idx, client_indexes, train_data_local_dict, test_data_local_dict,
                                                       train_data_local_num_dict, self.args, self.device, self.model)
            else:
                raise Exception(self.args.group_topology)

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        self.client_list = [Client(client_idx, train_data_local_dict[0], test_data_local_dict[0],
                                   train_data_local_num_dict[0], self.args, self.device, self.model)]
        logging.info("############setup_clients (END)#############")

    def train(self):
        pass

    def local_test_on_all_clients(self, model, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
#         train_metrics = {
#             'num_samples' : [],
#             'num_correct' : [],
#             'precisions' : [],
#             'recalls' : [],
#             'losses' : []
#         }

        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        client = self.client_list[0]
        client.update_local_dataset(0, self.c_train_data_local_dict[0],
                                    self.c_test_data_local_dict[0],
                                    self.c_train_data_local_num_dict[0])

        # train data
#         train_local_metrics = client.local_test(model, False)
#         train_metrics['num_samples'].append(train_local_metrics['test_total'])
#         train_metrics['num_correct'].append(train_local_metrics['test_correct'])
#         train_metrics['losses'].append(train_local_metrics['test_loss'])

        # test data
        test_local_metrics = client.local_test(model, True)
        test_metrics['num_samples'].append(test_local_metrics['test_total'])
        test_metrics['num_correct'].append(test_local_metrics['test_correct'])
        test_metrics['losses'].append(test_local_metrics['test_loss'])

        if self.args.dataset == "stackoverflow_lr":
#             train_metrics['precisions'].append(train_local_metrics['test_precision'])
#             train_metrics['recalls'].append(train_local_metrics['test_recall'])
            test_metrics['precisions'].append(test_local_metrics['test_precision'])
            test_metrics['recalls'].append(test_local_metrics['test_recall'])

        # test on training dataset
#         train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
#         train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
#         train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
#         train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
#             stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Pre": train_precision, "round": round_idx})
#             wandb.log({"Train/Rec": train_recall, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})
#             logging.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_precision, "round": round_idx})
            wandb.log({"Test/Rec": test_recall, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)

        else:
#             stats = {'training_acc': train_acc, 'training_loss': train_loss}
#             wandb.log({"Train/Acc": train_acc, "round": round_idx})
#             wandb.log({"Train/Loss": train_loss, "round": round_idx})
#             logging.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)
