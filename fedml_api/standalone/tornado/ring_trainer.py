import logging
import numpy as np

from fedml_api.standalone.tornado.consensus_trainer import ConsensusTrainer

class RingTrainer(ConsensusTrainer):

    def train(self):
        w_global = self.model.state_dict()
        w_chains = [w_global for _ in range(self.args.chain_num)]
        for global_round_idx in range(self.args.global_comm_round):
            logging.info("################Communication Round : {}".format(global_round_idx))
            group_to_client_indexes = self.client_sampling(global_round_idx, self.args.client_num_in_total, self.args.client_num_per_round)

            # generate a stochastic group exploration order
            if global_round_idx % self.args.group_num == 0:
                group_indexes = np.random.choice(self.args.group_num, self.args.group_num, replace=False)
                logging.info("group_indexes : {}".format(group_indexes))

            w_eval_chains_dict = {}
            for chain_idx in range(self.args.chain_num):
                w_chain = w_chains[chain_idx]

                # train each group chain
                group_idx = group_indexes[(global_round_idx+chain_idx) % self.args.group_num]
                sampled_client_indexes = group_to_client_indexes[group_idx]
                group = self.group_dict[group_idx]
                logging.info('group_idx : {}'.format(group.idx))
                w_group_list = group.train(global_round_idx, w_chain, sampled_client_indexes)
                for global_epoch, w in w_group_list:
                    w_chains[chain_idx] = w
                    if not global_epoch in w_eval_chains_dict: w_eval_chains_dict[global_epoch] = []
                    w_eval_chains_dict[global_epoch].append((group.get_sample_number(sampled_client_indexes), w))

            # aggregate chain weights into the global weight
            for global_epoch in sorted(w_eval_chains_dict.keys()):
                if global_epoch % self.args.frequency_of_the_test == 0:
                    w_eval_chains = w_eval_chains_dict[global_epoch]
                    w_eval_global = self.aggregate(w_eval_chains)

                    # evaluate performance
                    self.model.load_state_dict(w_eval_global)
                    self.local_test_on_all_clients(self.model, global_epoch)
