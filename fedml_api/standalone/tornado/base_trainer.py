import logging
import numpy as np

from fedml_api.standalone.tornado.base_cloud import Cloud
from fedml_api.standalone.hierarchical_fl.client import Client
from fedml_api.standalone.hierarchical_fl.trainer import Trainer as HierarchicalTrainer

GROUPING_MAX_STEADY_STEPS = 1
GROUPING_MAX_ASSOCIATE_NODES_TO_COMPARE = 1000
GROUPING_ERROR_THRESHOLD = 0.01

class BaseTrainer(HierarchicalTrainer):

    def __init__(self, dataset, model, device, args):
        super().__init__(dataset, model, device, args)

        self.model_bytes = self.calc_param_bytes(model)
        logging.info('model_bytes = {}'.format(self.model_bytes))

    def calc_param_bytes(self, model):
        modules = list(self.model.modules())
        param_sizes = []

        for i in range(1,len(modules)):
            m = modules[i]
            p = list(m.parameters())
            for j in range(len(p)):
                param_sizes.append(np.array(p[j].size()))

        total_bytes = 0
        for i in range(len(param_sizes)):
            s = param_sizes[i]
            total_bytes += np.prod(np.array(s)) * 4
        return total_bytes

    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logging.info("############setup_clients (START)#############")

        self.group_indexes = np.random.randint(0, self.args.group_num, self.args.client_num_in_total)
        self.c = Cloud(self.args, self.device, self.model,
                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict)
        gid_2_nids = self.to_gid_2_nids(self.group_indexes)
        self.c.digest(gid_2_nids)

        logging.info('Before Grouping')
        for g in self.c.group_dict.values():
            logging.info('{} {} {}'.format(g.idx, np.unique(np.concatenate([batch[1] for nid in g.get_N_k() for batch in train_data_local_dict[nid]])), len(g.get_N_k())))
        self.c = self.runGrouping(self.c)
        logging.info('After Grouping')
        for g in self.c.group_dict.values():
            logging.info('{} {} {}'.format(g.idx, np.unique(np.concatenate([batch[1] for nid in g.get_N_k() for batch in train_data_local_dict[nid]])), len(g.get_N_k())))

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        self.client_list = [Client(0, train_data_local_dict[0], test_data_local_dict[0],
                       train_data_local_num_dict[0], self.args, self.device, self.model)]
        logging.info("############setup_clients (END)#############")

    def runGrouping(self, c):
        logging.info('Grouping Started')
        if self.args.group_method == 'random':
            logging.info('Grouping Skipped for random group_method')
            return c
        if self.args.group_method == 'cluster' or self.args.group_method == 'iid':
            pass
        else:
            raise Exception(self.args.group_method)

        c = c.clone()
        z_rand = np.random.randint(0, self.args.group_num, self.args.client_num_in_total)
        gid_2_nids = self.to_gid_2_nids(z_rand)
        c.digest(gid_2_nids)

        c_star, cost_star = self.runKMedoidsGrouping(c)

        logging.info('Final cost_star=%.3f, group_num=%d' % (cost_star, len(c_star.group_dict)))
        logging.info('Grouping Finished')

        self.group_indexes = self.to_z(self.args.client_num_in_total, c_star.gid_2_nids)
        return c_star

    def runKMedoidsGrouping(self, c):
        # k-Medoids with Voronoi Iteration
        # https://en.wikipedia.org/wiki/K-medoids

        # Initialize Medoids
        medoidNids = np.random.choice(np.arange(self.args.client_num_in_total), size=self.args.group_num, replace=False)
        logging.info('Initial medoidNids: {}'.format(sorted(medoidNids)))

        # Associate
        c_cur, cost_cur = self.associate(c, medoidNids)
        c_star, cost_star = c_cur, cost_cur
        logging.info('Initial cost_star=%.3f' % (cost_star))

        # Iterate
        cntSteady = 0
        while cntSteady < GROUPING_MAX_STEADY_STEPS:
            cost_prev = cost_star

            # Determine medoids
            medoidNids = self.determineMedoidNids(c_cur)
            logging.info('medoidNids: {}'.format(sorted(medoidNids)))

            # Associate nodes to medoids
            c_cur, cost_cur = self.associate(c_cur, medoidNids)
            if self.isGreaterThan(cost_star, cost_cur):
                c_star, cost_star = c_cur, cost_cur
                cntSteady = 0 # 한 번이라도 바뀌면 Steady 카운터 초기화
            else:
                cntSteady += 1
            cost_new = cost_star
            logging.info('cntSteady=%d, cost_prev=%.3f, cost_new=%.3f' % (cntSteady, cost_prev, cost_new))
        return c_star, cost_star

    def isGreaterThan(self, lhs, rhs):
        return (lhs - rhs) > (GROUPING_ERROR_THRESHOLD * rhs)

    def associate(self, c, medoidNids):
        c = c.clone()
        medoidGid_2_nids = [ [nid] for nid in medoidNids ]
        if c.digest(medoidGid_2_nids) == False: raise Exception(str(medoidGid_2_nids))

        # Shuffle index every iteration
        randNids = np.arange(self.args.client_num_in_total)
        np.random.shuffle(randNids)

        maxNodes_per_group = int(np.ceil(self.args.client_num_in_total / self.args.group_num))

        z = self.to_z(self.args.client_num_in_total, c.gid_2_nids)
        for i, nid in enumerate(randNids):
            # Medoid 일 경우 무시
            if nid in medoidNids: continue

            if i >= GROUPING_MAX_ASSOCIATE_NODES_TO_COMPARE: break

            # Search for candidates with the same minimum cost
            costs = []
            for k, medoidNid in enumerate(medoidNids):
                # 목적지 그룹이 이전 그룹과 같을 경우 무시
                if z[nid] == k:
                    costs.append(float('inf'))
                    continue

                # 이미 많은 노드가 할당된 그룹은 통과 (Fair Association)
                if len(c.get_group(k).get_N_k()) >= maxNodes_per_group:
                    costs.append(float('inf'))
                    continue

                # 그룹 멤버쉽 변경 시도
                z[nid] = k

                # Cloud Digest 시도
                gid_2_nids = self.to_gid_2_nids(z)
                if c.digest(gid_2_nids) == False: raise Exception(str(z), str(gid_2_nids))

                # 다음 후보에 대한 Cost 계산
                costs.append(self.getAssociateCost(c))
            min_k = np.argmin(costs)
            cost_min = costs[min_k]
            z[nid] = min_k # 다음 Iteration 에서 Digest
            logging.info('nid: %3d\tcurrent cost: %.3f' % (nid, cost_min))

        # GROUPING_MAX_ASSOCIATE_NODES_TO_COMPARE 이후 나머지 nid 에 대해서는 Random 초기화
        for nid in range(self.args.client_num_in_total):
            if z[nid] is None:
                z[nid] = np.random.randint(0, self.args.group_num)

        # 마지막 멤버십 초기화가 발생했을 수도 있으므로, Cloud Digest 시도
        gid_2_nids = self.to_gid_2_nids(z)
        if c.digest(gid_2_nids) == False: raise Exception(str(z), str(gid_2_nids))

#         for k, g in enumerate(c.groups):
#             p_k = c.get_p_ks()[k]
#             logging.info('{} {} {}'.format(g.ps_nid, g.get_N_k(), p_k*g.get_DELTA_k()))
        return c, cost_min

    def getAssociateCost(self, c):
        if self.args.group_method == 'cluster':
            return c.get_emd()
        elif self.args.group_method == 'iid':
            return c.get_EMD()
        else:
            raise Exception(self.args.group_method)

    def determineMedoidNids(self, c):
        medoidNids = []
        if self.args.group_method == 'cluster':
            for g in c.group_dict.values():
                N_k = g.get_N_k()
                costs = []
                for nid in N_k:
    #                 p_i = g.get_p_k_is()[nid]
                    emd_i = g.calcEMD(g.nid_2_dataRatioPerClass[nid], g.dataRatioPerClass_k)
                    costs.append(emd_i)
                medoidNids.append(N_k[np.argmin(costs)])
        elif self.args.group_method == 'iid':
            for g in c.group_dict.values():
                N_k = g.get_N_k()
                costs = []
                for nid in N_k:
    #                 p_i = g.get_p_k_is()[nid]
                    EMD_i = g.calcEMD(g.nid_2_dataRatioPerClass[nid], c.dataRatioPerClass)
                    costs.append(EMD_i)
                medoidNids.append(N_k[np.argmin(costs)])
        else:
            raise Exception(self.args.group_method)
        return medoidNids

    def to_gid_2_nids(self, z):
        # 숫자만 필터링(None 제외) 후 Unique 적용
        gids = np.unique([gid for gid in z if isinstance(gid, int) or isinstance(gid, np.int32) or isinstance(gid, np.int64)])
        for gid in range(len(gids)):
            if not(gid in gids): raise Exception(str(gids)) # 모든 Group 이 사용되고 있는 지 검사
        gid_2_nids = [ sorted([ nid for nid, gid in enumerate(z) if gid == gid_ ]) for gid_ in gids ]
        return gid_2_nids

    def to_z(self, client_num_in_total, gid_2_nids):
        # gid_2_nids 에 없는 nid 가 있을 수도 있으므로 client_num_in_total 를 입력받고, 모두 None 으로 초기화
        z = [ None for _ in range(client_num_in_total) ]
        for gid, nids in enumerate(gid_2_nids):
            for nid in nids:
                z[nid] = gid
        return z

    def train(self):
        pass

    def get_comm_client_num(self):
        pass

    def local_test_on_all_clients(self, model, global_epoch):
        pass
