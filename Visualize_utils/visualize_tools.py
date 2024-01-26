import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from . import visualize_config as visualize_config
from . import visualize_graph as visualize_graph

def visualizing(att_node,
                att_A,
                data,
                label,
                predict,
                name,
                frame_size,
                num_att_A,
                vis_dir,
                dataset,
                num_edge=30,
                second_person_skip=True,
                color_map='jet'):

    if second_person_skip:
        skip = 2
    else:
        skip = 1

    if dataset == 'ntu60':
        num_node, small_node_list, node_coordinate_list, adjacency_matrix = visualize_config.NTU_RGB_D()
    elif dataset == 'ntu120':
        num_node, small_node_list, node_coordinate_list, adjacency_matrix = visualize_config.NTU_RGB_D()

    for i in tqdm(range(0, len(att_node), skip), leave=False, desc='# visualizing'):
        dir = vis_dir + '/Label' + str(label[i//2]) + '_Pred' + str(predict[i//2]) + '_Name_' + str(name[i//2])
        if not os.path.isdir(dir):
            os.makedirs(dir)
            np.save(dir + '/att_node.npy', att_node[i].cpu().numpy())
            np.save(dir + '/att_A.npy', att_A[i].cpu().numpy())
            np.save(dir + '/data.npy', data[i//2])

        map = gen_map(att_node[i], dir, color_map)
        A_list = gen_graph(att_A[i, :, :, :].cpu().numpy(), num_att_A, num_edge, num_node, small_node_list, node_coordinate_list, adjacency_matrix, dir)
        gen_map_graph(map, A_list, data[i//2].copy(), color_map, num_att_A, frame_size, num_node, small_node_list, adjacency_matrix, dir)


def gen_map(att_node, dir, color_map):
    map = torch.squeeze(att_node).permute(1, 0).data.cpu()
    map = (map - map.min()) / (map.max() - map.min())
    plt.figure()
    plt.imshow(map, cmap=color_map)
    # plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    # plt.tick_params(bottom=False, left=False, right=False, top=False)
    path = dir + '/att_map.pdf'
    # plt.colorbar()
    plt.savefig(path)
    plt.close()

    return map


def gen_graph(att_A, num_att_A, num_edge, num_node, small_node_list, node_coordinate_list, adjacency_matrix, dir):
    A_list = []
    for i in range(num_att_A):
        A = visualize_graph.build_K(att_A[i].copy(), num_edge, num_node)
        A_list.append(A)
        plot_path = dir + '/A_' + str(i) + '.pdf'
        visualize_graph.plot_A(A, plot_path, num_node, small_node_list, node_coordinate_list, adjacency_matrix,)

        """
        # plot all edge
        A = np.where(att_A[i] > 0, 1, 0)
        num_edges = np.sum(A)
        plot_path = dir + '/A_num_edge' + str(num_edges) + '_' + str(i) + '.pdf'
        plot_A(A, plot_path)
        """

    return A_list


def gen_map_graph(map, A_list, data, color_map, num_att_A, frame_size, num_node, small_node_list, adjacency_matrix, dir):
    cmap = plt.get_cmap(color_map)
    for i in range(num_att_A):
        A = A_list[i]
        p = 0
        for t in range(0, frame_size, 2):
            plot_path = dir + '/A' + str(i) + '_frame' + str(t//2) + '.pdf'
            visualize_graph.plot_map_graph(map[:, p], A, data[:, t, :, 0].copy(), plot_path, cmap, num_node, small_node_list, adjacency_matrix)
            p += 1
