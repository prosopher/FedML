import wandb
import logging
import numpy as np

from fedml_api.standalone.tornado.base_trainer import BaseTrainer

NUM_EVAL_CLIENTS = 100

class ConsensusTrainer(BaseTrainer):

    def local_test_on_all_clients(self, model, global_epoch):
        logging.info("################local_test_on_all_clients : {}".format(global_epoch))

        comm_cnt = int((global_epoch+1) / self.args.epochs)
        comm_bytes = self.model_bytes * comm_cnt * self.get_comm_client_num()

        train_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        test_metrics = {
            'num_samples' : [],
            'num_correct' : [],
            'precisions' : [],
            'recalls' : [],
            'losses' : []
        }

        client_indexes = np.random.choice(range(self.args.client_num_in_total), NUM_EVAL_CLIENTS, replace=False)
        client = self.client_list[0]
        for client_idx in client_indexes:
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # train data
            train_local_metrics = client.local_test(model, False)
            train_metrics['num_samples'].append(train_local_metrics['test_total'])
            train_metrics['num_correct'].append(train_local_metrics['test_correct'])
            train_metrics['losses'].append(train_local_metrics['test_loss'])

            # test data
            test_local_metrics = client.local_test(model, True)
            test_metrics['num_samples'].append(test_local_metrics['test_total'])
            test_metrics['num_correct'].append(test_local_metrics['test_correct'])
            test_metrics['losses'].append(test_local_metrics['test_loss'])

            if self.args.dataset == "stackoverflow_lr":
                train_metrics['precisions'].append(train_local_metrics['test_precision'])
                train_metrics['recalls'].append(train_local_metrics['test_recall'])
                test_metrics['precisions'].append(test_local_metrics['test_precision'])
                test_metrics['recalls'].append(test_local_metrics['test_recall'])

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])
        train_precision = sum(train_metrics['precisions']) / sum(train_metrics['num_samples'])
        train_recall = sum(train_metrics['recalls']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        test_precision = sum(test_metrics['precisions']) / sum(test_metrics['num_samples'])
        test_recall = sum(test_metrics['recalls']) / sum(test_metrics['num_samples'])

        if self.args.dataset == "stackoverflow_lr":
            stats = {'training_acc': train_acc, 'training_precision': train_precision, 'training_recall': train_recall, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Train/Pre": train_precision, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Train/Rec": train_recall, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Train/Loss": train_loss, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Test/Pre": test_precision, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Test/Rec": test_recall, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Test/Loss": test_loss, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            logging.info(stats)

        else:
            stats = {'training_acc': train_acc, 'training_loss': train_loss, "global_epoch": global_epoch}
            wandb.log({"Train/Acc": train_acc, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Train/Loss": train_loss, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss, "global_epoch": global_epoch}
            wandb.log({"Test/Acc": test_acc, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            wandb.log({"Test/Loss": test_loss, "global_epoch": global_epoch, "comm_bytes": comm_bytes})
            logging.info(stats)
