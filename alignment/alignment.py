from .dtw import *
import numpy as np
import torch
# from utils import time_elapsed

# @time_elapsed
def optimized_distance_finder(self_subtraction_matrix,embs,query_embs):
    """
    Credit to Jason :D
    Algorithm: 
        A : [a0, a1, a2, a3, a4 ... ... an]
        B : [b0, b1, b2, b3, b4 ... bm] (n > m)
        Compute distance B and sliding window(A)
        np.sum([a0-b0, a1-b1, a2-b2, a3-b3, a4-b4 ... am-bm]) -> dist0
        np.sum([a1-b0, a2-b1, a3-b2, a4-b3, a5-b4 ... a(m+1)-bm]) -> dist1
        np.sum([a2-b0, a3-b1, a4-b2, a5-b3, a6-b4 ... a(m+2)-bm]) -> dist2
        ...
        a1-b0 is equal to a1-a0 + a0-b0, a2-b1 is equal to a2-a1 + a1-b1, a3-b2 is equal to a3-a2 + a2-b2 ...
        Similarly, 
        a2-b0 is equal to a2-a0 + a0-b0, a3-b1 is equal to a3-a1 + a1-b1, a4-b2 is equal to a4-a2 + a2-b2 ...
    So we only need to compute dist0 and a(0~m) - a(0~m)
    Note: This cannot work on l2 distance because square is incoporated, so we use abs to calcualate distances
    """
    ### Compute dist0
    n = self_subtraction_matrix.size(0)
    m = len(query_embs)
    
    ## dist.shape = m x query_embs.size(-1)
    dist0 = embs[:m] - query_embs ## dont do abs and sum here because we need the signed value to do algorithm
    
    distances = [torch.sum((dist0)**2)]
    for i in range(1, n-m+1):
        ## index diagonallly in self_subtraction_matrix
        distances.append(torch.sum((dist0 + (torch.diagonal(self_subtraction_matrix,i).transpose(0,1)[:m]) )**2))
    min_distance = torch.argmin(torch.stack(distances))
    return min_distance

def align(query_embs,key_embs,name) -> (int):
    """
    Compute which time window of key_embs is most similar to the query_embs(user's input)
    inputs:
    @ query_embs: Tu , 512
    @ key_embs: Ts , 512
    """
    def find_min_distance_with_standard(query_embs,emb):
        """
        Deprecated due to high computation time.
        inputs:
        @ emb: T , 512
        @ query_embs: 
        """
        dist_fn = lambda x, y: np.sum((x - y) ** 2)

        naive_distances = []
        min_dists = []

        if len(emb) == len(query_embs):
            return 0
        

        for i in range(len(emb)-len(query_embs)+1): ## * use sliding window
            window_embs = emb[i:i+len(query_embs)]   ## * compare in which window
            min_dist, _, _, _ = dtw(query_embs.detach().cpu().numpy(), window_embs.detach().cpu().numpy(), dist=dist_fn) ## * the dtw yields
            min_dists.append(min_dist)

            ## the aligning cost is approximately the same as the naive distance
            naive_distance = torch.sum(torch.FloatTensor([(torch.sum((query_embs[j]-window_embs[j])**2)) for j in range(len(query_embs))])) 
            naive_distances.append(naive_distance)

        # start_frame = min_dists.index(min(min_dists)) 
        naive_start_frame = naive_distances.index(min(naive_distances)) ## * the smallest value (set it as start frame)
        # print('naive distances : ' , naive_distances)
        # print("min_dists: " , min_dists)
        # print("min distance : ",min_dists.index(min(min_dists)))
        return naive_start_frame
    
    tmp = key_embs.expand(key_embs.size(0),key_embs.size(0),-1)
    self_subtraction_matrix = (tmp - tmp.transpose(0,1))
    """
          a0    a1   a2  a3     a4 ...
    a0  a0-a0 a1-a0 a2-a0 a3-a0 a4-a0
    a1  a0-a1 a1-a1 a2-a1 a3-a1 a4-a1
    a2  a0-a2 a1-a2 a2-a2 a3-a2 a4-a2
    a3  a0-a3 a1-a3 a2-a3 a3-a3 a4-a3
    a4  a0-a4 a1-a4 a2-a4 a3-a4 a4-a4
    ...
    """

    if len(key_embs) < len(query_embs): 
        # print("\033[91m" + f"Typically this shouldn't happen, consider checking the query embs (query embs {name} has shape {query_embs.shape})" + "\033[0m")
        key_embs, query_embs = query_embs, key_embs
    # start_frame = find_min_distance_with_standard(query_embs,key_embs)
    # assert start_frame == opt_start_frame
    # result = input_embs[start_frame:start_frame+len(query_embs)]
    # assert result.shape == query_embs.shape
    opt_start_frame = optimized_distance_finder(self_subtraction_matrix,key_embs,query_embs)
    # print('opt start frame : ',opt_start_frame)
    return opt_start_frame

import unittest

class TestAlign(unittest.TestCase):
    def test_align(self):
        for i in range (100):
            # Create some dummy data for testing
            query_embs = torch.randn(10, 128)
            input_embs = torch.randn(20, 128)

            # Call the align function
            # print(align(query_embs, input_embs,None))
            

if __name__ == '__main__':
    torch.manual_seed(42)
    unittest.main()