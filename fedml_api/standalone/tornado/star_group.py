from fedml_api.standalone.hierarchical_fl.group import Group as HierarchicalGroup

class StarGroup(HierarchicalGroup):

    def train(self, global_round_idx, w_init, sampled_client_indexes):
        return super().train(global_round_idx, w_init, sampled_client_indexes)
