"""
Intent: Visualize skeleton data
Author: Tom
Last update date: 2023/09/21
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import json
import math
import matplotlib
import pickle
from argparse import ArgumentParser, Namespace


matplotlib.use('Agg')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="/home/tom/sporttech/skating_coach/datasets/Axel2_skeleton/467205292522733572_0/res.pk", help="Path to the pickle file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to the output folder"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Set limit to the number of output gifs (0 = no limit)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Set the starting index of the visualization"
    )
    args = parser.parse_args()
    return args
    

def draw(joints, filename):
    num_frame = len(joints)
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    # 设置初始点和目标点的坐标

    # 创建一个函数，用于在每一帧中绘制线段
    max_x , min_x = -math.inf, math.inf
    max_y , min_y = -math.inf, math.inf
    max_z , min_z = -math.inf, math.inf

    for joint in joints : 
        for i in range(0,22) : 
            joint[i][1] = -joint[i][1]
            x_values = joint[i][0]
            y_values = joint[i][1]
            z_values = joint[i][2] 
            if(x_values > max_x) : max_x = x_values
            if(x_values < min_x) : min_x = x_values
            if(y_values > max_y) : max_y = y_values
            if(y_values < min_y) : min_y = y_values
            if(z_values > max_z) : max_z = z_values
            if(z_values < min_z) : min_z = z_values
            max_dist = max(max_x - min_x, max_y - min_y, max_z - min_z)

    def animate(frame):
        ax.clear()
        
        skeleton_bone = [[15,12],[12,9],[9,6],[6,3],[3,0]]
        skeleton_left_leg = [[0,1],[1,4],[4,7],[7,10]]
        skeleton_right_leg = [[0,2],[2,5],[5,8],[8,11]]
        skeleton_left_hand = [[9,13],[13,16],[16,18],[18,20]]
        skeleton_right_hand = [[9,14],[14,17],[17,19],[19,21]]

        for index in skeleton_bone : 
            first = index[0]
            second = index[1]
            ax.plot([joints[frame][first][0], joints[frame][second][0]], [joints[frame][first][1], joints[frame][second][1]],[joints[frame][first][2], joints[frame][second][2]] ,'y')  
        for index in skeleton_left_leg : 
            first = index[0]
            second = index[1]
            ax.plot([joints[frame][first][0], joints[frame][second][0]], [joints[frame][first][1], joints[frame][second][1]],[joints[frame][first][2], joints[frame][second][2]] ,'c') 
        for index in skeleton_right_leg : 
            first = index[0]
            second = index[1]
            ax.plot([joints[frame][first][0], joints[frame][second][0]], [joints[frame][first][1], joints[frame][second][1]],[joints[frame][first][2], joints[frame][second][2]] ,'m')  
        for index in skeleton_left_hand : 
            first = index[0]
            second = index[1]
            ax.plot([joints[frame][first][0], joints[frame][second][0]], [joints[frame][first][1], joints[frame][second][1]],[joints[frame][first][2], joints[frame][second][2]] ,'g') 
        for index in skeleton_right_hand : 
            first = index[0]
            second = index[1]
            ax.plot([joints[frame][first][0], joints[frame][second][0]], [joints[frame][first][1], joints[frame][second][1]],[joints[frame][first][2], joints[frame][second][2]] ,'r')    
        max_dist = max(max_x - min_x,max_y - min_y,max_z - min_z)
        ax.set_xlim(min_x, min_x + max_dist)
        ax.set_ylim(min_y, min_y + max_dist)
        ax.set_zlim(min_z, min_z + max_dist)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Z coordinate')
        # ax.set_title('从一个点到另一个点的动画') 可以放labels

    # 创建一个动画对象
    ani = animation.FuncAnimation(fig, animate, frames=num_frame, repeat=True)
    # save_path = os.path.join(root, 'animation_skeleton_minus.gif')
    save_path = filename
    # 保存动画为GIF文件
    print(save_path)
    # ani.save(save_path, fps=10)
    ani.save(save_path, fps=30)


def extract_x_y_z(bodys) :
    joint = list()
    for body in bodys :
        body_process = list()
        index = 0
        first = True
        xyz = []
        for joints in body : 
            if index % 3 == 0 and first == False : 
                body_process.append(xyz) 
                xyz = []
            xyz.append(joints)
            index+=1
            first = False
        body_process.append(xyz)
        joint.append(body_process)
    return joint


def main(args):
    filename = args.filename
    output_dir = args.output_dir
    limit = args.limit
    offset = 0 if args.limit < 0 else args.limit  # To prevent negative offset
    if filename.lower().endswith('.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            name_list = [ datum['video_name'] for datum in data ]
            skeleton_list = [ datum['features'] for datum in data ]
            joint_list = [ extract_x_y_z(skeleton) for skeleton in skeleton_list ]
            if len(name_list) != len(joint_list):
                print('Data corrupted!')
                return -1
            if limit > 0:
                name_list = name_list[offset: min(offset + limit, len(name_list))]
                joint_list = joint_list[offset: min(offset + limit, len(joint_list))]
            for i in range(len(joint_list)):
                print(f'Drawing {name_list[i]}')
                draw(joint_list[i], os.path.join(output_dir, name_list[i] + '.gif'))
    else:
        print("Incorrect file format")
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
    