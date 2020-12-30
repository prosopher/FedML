import numpy as np

from fedml_api.standalone.tornado.star_group import StarGroup
from fedml_api.standalone.tornado.ring_group import RingGroup

class Cloud:

    def __init__(self, args, device, model, train_data_local_dict, test_data_local_dict, train_data_local_num_dict):
        self.args = args
        self.device = device
        self.model = model
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.ready = False

        unique_classes = np.unique(np.concatenate([batch[1] for nid in sorted(train_data_local_dict) for batch in train_data_local_dict[nid]]))
        num_classes = max(unique_classes)
        self.dataRatioPerClass = np.zeros(num_classes+1, dtype=np.int32)
        nid_2_dataRatioPerClass = {}
        nid_2_dataNumPerClass = {}
        for nid in train_data_local_dict:
            dataNumPerClass = np.zeros(num_classes+1, dtype=np.int32)
            for batch in train_data_local_dict[nid]:
                for j in range(len(batch[1])):
                    cid = batch[1][j]
                    self.dataRatioPerClass[cid] += 1
                    dataNumPerClass[cid] += 1
            nid_2_dataNumPerClass[nid] = dataNumPerClass
            nid_2_dataRatioPerClass[nid] = dataNumPerClass / sum(dataNumPerClass)
        self.dataRatioPerClass = self.dataRatioPerClass / sum(self.dataRatioPerClass)

        self.group_dict = {}
        for k in range(self.args.group_num):
            if self.args.group_topology == 'stars':
                self.group_dict[k] = StarGroup(args, device, model, k, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                               self.dataRatioPerClass, nid_2_dataRatioPerClass, nid_2_dataNumPerClass)
            elif self.args.group_topology == 'rings':
                self.group_dict[k] = RingGroup(args, device, model, k, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                               self.dataRatioPerClass, nid_2_dataRatioPerClass, nid_2_dataNumPerClass)
            else:
                raise Exception(self.args.group_topology)

    def clone(self):
        c_cloned = Cloud(self.args, self.device, self.model,
                  self.train_data_local_dict, self.test_data_local_dict, self.train_data_local_num_dict)
        c_cloned.digest(self.gid_2_nids)
        return c_cloned

    def get_group(self, k):
        return self.group_dict[k]

    def get_p_ks(self): # 그룹 별 데이터 크기 집합
        if self.ready == False: raise Exception
        return self.p_ks

    def get_emd(self):
        if self.ready == False: raise Exception
        emd_ks = [ g.get_emd_k() for g in self.group_dict.values() ]
        emd = np.average(emd_ks, weights=self.get_p_ks())
        return emd

    def get_EMD(self):
        if self.ready == False: raise Exception
        EMD_ks = [ g.get_EMD_k() for g in self.group_dict.values() ]
        EMD = np.average(EMD_ks, weights=self.get_p_ks())
        return EMD

    def digest(self, gid_2_nids, debugging=False):
        if gid_2_nids == None:
            return False
        else:
            self.gid_2_nids = gid_2_nids
            for k, N_k in enumerate(gid_2_nids):
                self.group_dict[k].set_N_k(N_k)

            for g in self.group_dict.values():
                g.digest(debugging)

            # digest in group_dict
            self.p_ks = []
            for g in self.group_dict.values():
                g.digest(debugging)
                self.p_ks.append(g.get_p_k())

            # Set flag
            self.ready = True
            return True
