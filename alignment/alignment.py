from alignment.dtw import *
import numpy as np
def align(query_embs,input_embs) -> (np.ndarray,np.ndarray):
    """
    1. Use the sliding window method to find the interval which has minimum distance between input and standard. (To find the frames where user is doing target motion)
    2. Return the query_embs[Tx128] (Standard embedding) and the input_embs[Tx128] (User's embedding) which was found in 1.
    """
    def find_min_distance_with_standard(emb,query_embs):
        def dist_fn(x, y):
            dist = np.sum((x-y)**2)
            return dist

        min_dists = []

        if len(emb) == len(query_embs):
            return 0

        for i in range(len(emb)-len(query_embs)): ## * use sliding window
            query_embs = emb[i:i+len(query_embs)]   ## * compare in which window
            min_dist, _, _, _ = dtw(query_embs.cpu(), input_embs.cpu(), dist=dist_fn) ## * the dtw yields
            min_dists.append(min_dist)
        start_frame = min_dists.index(min(min_dists)) ## * the smallest value (set it as start frame)
        return start_frame 
    if len(input_embs) < len(query_embs): ## There is no need to pad the emb, it is guranteed to be played at the first frame. (if we think about the sliding window algo)
        input_embs, query_embs = query_embs, input_embs
    start_frame = find_min_distance_with_standard(input_embs,query_embs)
    result = input_embs[start_frame:start_frame+len(query_embs)]
    assert result.shape == query_embs.shape
    return query_embs, result

import unittest

class TestAlign(unittest.TestCase):
    def test_align(self):
        # Create some dummy data for testing
        query_embs = np.random.rand(10, 128)
        input_embs = np.random.rand(20, 128)

        # Call the align function
        query_result, input_result = align(query_embs, input_embs)

        # Check if the shapes of the returned arrays match the shape of query_embs
        self.assertEqual(query_result.shape, query_embs.shape)
        self.assertEqual(input_result.shape, query_embs.shape)

if __name__ == '__main__':
    unittest.main()