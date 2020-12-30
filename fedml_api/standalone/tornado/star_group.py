import logging

from fedml_api.standalone.tornado.base_group import BaseGroup

class StarGroup(BaseGroup):

    def train(self, global_round_idx, w, sampled_client_indexes):
        sampled_client_list = [self.get_client(client_idx) for client_idx in sampled_client_indexes]
        w_group = w
        w_group_list = []
        for group_round_idx in range(self.args.group_comm_round):
            logging.info("Group ID : {} / Group Communication Round : {}".format(self.idx, group_round_idx))
            w_locals_dict = {}

            # train each client
            for client in sampled_client_list:
                w_local_list = client.train(global_round_idx, group_round_idx, w_group)
                for global_epoch, w in w_local_list:
                    if not global_epoch in w_locals_dict: w_locals_dict[global_epoch] = []
                    w_locals_dict[global_epoch].append((client.get_sample_number(), w))

            # aggregate local weights
            for global_epoch in sorted(w_locals_dict.keys()):
                w_locals = w_locals_dict[global_epoch]
                w_group_list.append((global_epoch, self.aggregate(w_locals)))

            # update the group weight
            w_group = w_group_list[-1][1]
        return w_group_list


    def get_comm_client_num(self):
        return int(self.args.client_num_per_round/self.args.group_num)
