import numpy as np
import logging

from fedml_api.standalone.tornado.base_group import BaseGroup

class RingGroup(BaseGroup):

    def train(self, global_round_idx, w_init, sampled_client_indexes):
        sampled_clients = [self.get_client(client_idx) for client_idx in sampled_client_indexes]

        w_group_list = []
        w_chains = [w_init for _ in range(self.args.chain_num)]
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # generate a stochastic client exploration order
            if group_round_idx % len(sampled_clients) == 0:
                np.random.shuffle(sampled_clients)

            w_eval_chains_dict = {}
            for chain_idx in range(self.args.chain_num):
                w_chain = w_chains[chain_idx]

                # train each client chain
                client_idx = (group_round_idx+chain_idx) % len(sampled_clients)
                client = sampled_clients[client_idx]
                logging.info('Client ID : {}'.format(client.client_idx))
                w_local_list = client.train(global_round_idx, group_round_idx, w_chain)
                for global_epoch, w in w_local_list:
                    w_chains[chain_idx] = w
                    if not global_epoch in w_eval_chains_dict: w_eval_chains_dict[global_epoch] = []
                    w_eval_chains_dict[global_epoch].append((client.get_sample_number(), w))

            # aggregate local weights
            for global_epoch in sorted(w_eval_chains_dict.keys()):
                w_eval_chains = w_eval_chains_dict[global_epoch]
                w_group_list.append((global_epoch, self.aggregate(w_eval_chains)))
        return w_group_list

    def get_comm_client_num(self):
        return self.args.chain_num
