import numpy as np

class Graph() :
    def __init__(self, layout = 'ntu-rgb+d', strategy = 'spatial', hop_size = 1, dilation = 1) :
        self.hop_size = hop_size
        self.dilation = dilation
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, hop_size = hop_size)
        self.get_adjacency(strategy)

    def __str__(self) :
        return self.A

    def get_edge(self, layout) :
        if layout == 'ntu-rgb+d' :
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                             (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                             (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        elif layout == 'ntu-rgb+d_all_1' :
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = []
            for i in range(1, self.num_node + 1) :
                for j in range(1, self.num_node + 1) :
                    if (i != j) and (j, i) not in neighbor_base :
                        neighbor_base.append((i, j))
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'kinetics_skeleton' :
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'SMPL' :
            self.num_node = 22
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_base = [(0, 1), (0, 2), (0, 3), (1, 4), (2,5), (3,6), (4, 7), (5, 8), (6, 9), (7, 10),
                             (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16),  (14, 17), (16, 18),
                             (17, 19), (18,  20), (19, 21)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.edge = self_link + neighbor_link
            self.center = 0
        else :
            raise ValueError("There is no such layout : [{}]".format(layout))

    def get_adjacency(self, strategy) :
        valid_hop = range(0, self.hop_size + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop :
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform' :
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance' :
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop) :
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial' :
            A = []
            for hop in valid_hop :
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node) :
                    for j in range(self.num_node) :
                        if self.hop_dis[j, i] == hop :
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center] :
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center] :
                                a_close[j, i] = normalize_adjacency[j, i]
                            else :
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0 :
                    A.append(a_root)
                else :
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else :
            raise ValueError("This strategy does not xxist.")

def get_hop_distance(num_node, edge, hop_size = 1) :
    A = np.zeros((num_node, num_node))
    for i, j in edge :
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(hop_size, -1, -1) :
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A) :
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node) :
        if Dl[i] > 0 :
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A) :
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node) :
        if Dl[i] > 0 :
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
