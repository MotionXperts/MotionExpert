import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import json
folder_path = '/home/weihsin/datasets/Axel/HybrIK'
import matplotlib
matplotlib.use('Agg')
def draw(joints,num_frame,root):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')


    def animate(frame):
        ax.clear()
        for i in range(0,22) : 
            x_values = joints[frame][i][0]
            y_values = joints[frame][i][1]
            z_values = joints[frame][i][2]            
            ax.scatter(x_values,y_values,z_values)

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
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_ylabel('Z coordinate')

    ani = animation.FuncAnimation(fig, animate, frames=num_frame, repeat=True)
    save_path = os.path.join(root, 'animation_skeleton.gif')
    # save to gif file
    print(save_path)
    ani.save(save_path, fps=1000)

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
def read_file(file_path,file_name,root) :
    print(root)
    with open(file_path,"rb") as f:
        labeldata = json.load(f)
    joints = extract_x_y_z(labeldata['features'])
    draw(joints,len(joints),root)

def read_folder(folder_path) :
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                file_name = os.path.basename(os.path.dirname(file_path))
                read_file(file_path,file_name,root)

read_folder(folder_path)                
        