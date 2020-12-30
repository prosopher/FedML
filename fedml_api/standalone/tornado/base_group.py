import logging
import numpy as np

from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.hierarchical_fl.group import Group as HierarchicalGroup

class BaseGroup(HierarchicalGroup):

    def __init__(self, args, device, model, idx, train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                dataRatioPerClass, nid_2_dataRatioPerClass, nid_2_dataNumPerClass):
        self.idx = idx
        self.args = args
        self.device = device
        self.model = model
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.dataRatioPerClass = dataRatioPerClass
        self.nid_2_dataRatioPerClass = nid_2_dataRatioPerClass
        self.nid_2_dataNumPerClass = nid_2_dataNumPerClass

        self.N_k = []
        self.ready = False

    def get_client(self, i):
        return Client(i, self.train_data_local_dict[i], self.test_data_local_dict[i],
                       self.train_data_local_num_dict[i], self.args, self.device, self.model)

    def set_N_k(self, N_k):
        if not(self.N_k == N_k):
            self.N_k = N_k
            self.ready = False

    def get_N_k(self): # 그룹 노드 집합
        if self.ready == False: raise Exception
        return self.N_k

    def get_p_k(self): # 그룹 데이터 크기
        if self.ready == False: raise Exception
        return self.p_k

    def get_p_k_is(self):
        if self.ready == False: raise Exception
        return self.p_k_is

    def calcEMD(self, a, b):
        if len(a) != len(b): raise Exception(len(a), len(b))
        return sum([ abs(a[i] - b[i]) for i in range(len(a)) ])

    def get_emd_k(self):
        if self.ready == False: raise Exception
        return self.emd_k

    def get_EMD_k(self):
        if self.ready == False: raise Exception
        return self.EMD_k

    def digest(self, debugging=False):
        if self.ready == True: return # Group 에 변화가 없을 때 연산되는 것을 방지
        self.p_k = 0
        self.p_k_is = []
        for nid in self.N_k:
            p_k_i = sum(len(batch[1]) for batch in self.train_data_local_dict[nid])
            self.p_k += p_k_i
            self.p_k_is.append(p_k_i)

        num_classes = len(self.dataRatioPerClass)
        self.dataRatioPerClass_k = np.zeros(num_classes, dtype=np.int32)
        for nid in self.N_k:
            self.dataRatioPerClass_k += self.nid_2_dataNumPerClass[nid]
        self.dataRatioPerClass_k = self.dataRatioPerClass_k / sum(self.dataRatioPerClass_k)
        emd_k_is = [ self.calcEMD(self.nid_2_dataRatioPerClass[nid], self.dataRatioPerClass_k) for nid in self.N_k ]
        self.emd_k = np.average(emd_k_is, weights=self.p_k_is)

        self.EMD_k = self.calcEMD(self.dataRatioPerClass_k, self.dataRatioPerClass)

        if debugging == True:
            logging.info('digested group ' + str(self.idx))
        self.ready = True
