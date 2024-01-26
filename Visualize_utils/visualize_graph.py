import numpy as np
from PIL import Image, ImageDraw


def build_K(M, num_edge, num_node):
    A = np.identity(num_node)
    for i in range(num_node):
        M[i][i] = -100
    for i in range(num_edge):
        leave_index = np.argmax(M)
        x = leave_index % num_node
        y = leave_index // num_node
        if x == y:
            raise ValueError
        else:
            A[y][x] = 1
            A[x][y] = 1
            M[y][x] = M[x][y] = -100

    return A


def plot_A(A, plot_path, num_node, small_node_list, node_coordinate_list, adjacency_matrix, node_size=25, small_node_size=15):
    img = Image.new("RGB", (600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # black edge
    for x in range(num_node):
        for y in range(num_node):
            if 1 == adjacency_matrix[x][y]:
                s_x = node_coordinate_list[x][0]
                s_y = node_coordinate_list[x][1]
                f_x = node_coordinate_list[y][0]
                f_y = node_coordinate_list[y][1]
                draw.line((s_x, s_y, f_x, f_y), fill=(0, 0, 0), width=10)

    # plot edge
    red = (234, 30, 30)
    for y in range(num_node):
        for x in range(num_node):
            if A[y][x] == 1:
                s_x = node_coordinate_list[x][0]
                s_y = node_coordinate_list[x][1]
                f_x = node_coordinate_list[y][0]
                f_y = node_coordinate_list[y][1]
                draw.line((s_x, s_y, f_x, f_y), fill=red, width=4)
    # plot node
    gray = (130, 130, 130)
    for i, (x, y) in enumerate(node_coordinate_list):
        if i + 1 not in small_node_list:
            draw.ellipse(((x - node_size, y - node_size), (x + node_size, y + node_size)), outline=(255, 255, 255),
                         fill=gray)
        else:
            draw.ellipse(((x - small_node_size, y - small_node_size), (x + small_node_size, y + small_node_size)),
                         outline=(255, 255, 255), fill=gray)

    # save
    img.save(plot_path, quality=100)
    img.close()


def plot_map_graph(map, A, data, plot_path, cmap, num_node, small_node_list, adjacency_matrix, node_size=15, small_node_size=10):
    img = Image.new("RGB", (600, 900), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    gray = (130, 130, 130)

    # Inversion and Expansion
    data[0] = data[0] * -400
    data[1] = data[1] * -400


    # node list
    node_list = []
    x_decay = 300 - data[0][0]
    y_decay = 450 - data[1][0]
    for i in range(num_node):
        node_list.append([data[0][i] + x_decay, data[1][i] + y_decay])

    # gray edge
    for x in range(num_node):
        for y in range(num_node):
            if 1 == adjacency_matrix[x][y]:
                s_x = node_list[x][0]
                s_y = node_list[x][1]
                f_x = node_list[y][0]
                f_y = node_list[y][1]
                draw.line((s_x, s_y, f_x, f_y), fill=gray, width=5)

    # plot edge
    red = (204, 0, 0)
    for y in range(num_node):
        for x in range(num_node):
            if A[y][x] == 1:
                s_x = node_list[x][0]
                s_y = node_list[x][1]
                f_x = node_list[y][0]
                f_y = node_list[y][1]
                draw.line((s_x, s_y, f_x, f_y), fill=red, width=3)

    # plot node
    for i, (x, y) in enumerate(node_list):
        col = cmap(map[i])
        color = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
        if i + 1 not in small_node_list:
            draw.ellipse(((x - node_size, y - node_size), (x + node_size, y + node_size)), outline=(255, 255, 255),
                         fill=color)
        else:
            draw.ellipse(((x - small_node_size, y - small_node_size), (x + small_node_size, y + small_node_size)),
                         outline=(255, 255, 255), fill=color)

    # save
    img.save(plot_path, quality=100)
    img.close()
