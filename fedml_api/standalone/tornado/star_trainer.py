import logging

from fedml_api.standalone.tornado.consensus_trainer import ConsensusTrainer

class StarTrainer(ConsensusTrainer):

    def train(self):
        w_global = self.model.state_dict()
        for global_round_idx in range(self.args.global_comm_round):
            logging.info("################Global Communication Round : {}".format(global_round_idx))
            group_to_client_indexes = self.client_sampling(global_round_idx, self.args.client_num_in_total,
                                                  self.args.client_num_per_round)

            # train each group
            w_groups_dict = {}
            for group_idx in sorted(group_to_client_indexes.keys()):
                sampled_client_indexes = group_to_client_indexes[group_idx]
                group = self.group_dict[group_idx]
                w_group_list = group.train(global_round_idx, w_global, sampled_client_indexes)
                for global_epoch, w in w_group_list:
                    if not global_epoch in w_groups_dict: w_groups_dict[global_epoch] = []
                    w_groups_dict[global_epoch].append((group.get_sample_number(sampled_client_indexes), w))

            # aggregate group weights into the global weight
            for global_epoch in sorted(w_groups_dict.keys()):
                w_groups = w_groups_dict[global_epoch]
                w_global = self.aggregate(w_groups)

                # evaluate performance
                if global_epoch % self.args.frequency_of_the_test == 0:
                    self.model.load_state_dict(w_global)
                    self.local_test_on_all_clients(self.model, global_epoch)
