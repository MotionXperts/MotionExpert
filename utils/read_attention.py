import json
import os
import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import random
import os, json, math, pickle, matplotlib, torch
import torch.nn as nn

offset = 0.75
matplotlib.use('Agg')
# Specify the path to your .pkl file
dtype=torch.float32
root_name = './'


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

def draw(joints,file_name,color_node,attention_matrix_new):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('z axis')
    ax.set_zlabel('y axis')
    
    num_frame = len(joints)
    joints = joints.reshape(num_frame,22,3)
    joints[:,:,1],joints[:,:,2] = joints[:,:,2].clone(),joints[:,:,1].clone()
    attention_matrix_new = (attention_matrix_new - attention_matrix_new.min()) / (attention_matrix_new.max() - attention_matrix_new.min())

    combinations = [[i,j] for i in range(0,22) for j in range(0,22) ]

    def animate(frame):
        ax.clear()
        skeleton_index = [ [ 0, 1 ], [ 0, 2 ], [ 0, 3 ], [ 1, 4 ], [ 2, 5 ], [ 3, 6 ], [ 4, 7 ], [ 5, 8 ], [ 6, 9 ], 
                           [ 7, 10], [ 8, 11], [ 9, 12], [ 9, 13], [ 9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21]]
        skeleton = joints[frame]
        ax.set_xlim([joints[0][0][0]-offset, joints[0][0][0]+offset])
        ax.set_ylim([joints[0][0][1]-offset, joints[0][0][1]+offset])
        ax.set_zlim([joints[0][0][2]-offset, joints[0][0][2]+offset])
        ax.set_xlabel('x axis')
        ax.set_zlabel('y axis')
        ax.set_ylabel('z axis')
   
        skeleton_bone = [[15,12],[12,9],[9,6],[6,3],[3,0]]
        skeleton_left_leg = [[0,1],[1,4],[4,7],[7,10]]
        skeleton_right_leg = [[0,2],[2,5],[5,8],[8,11]]
        skeleton_left_hand = [[9,13],[13,16],[16,18],[18,20]]
        skeleton_right_hand = [[9,14],[14,17],[17,19],[19,21]]

        color_map = 'jet'
        colornodes = color_node[frame]
       
        colornodes = (colornodes - colornodes.min()) / (colornodes.max() - colornodes.min())
    
        x , y , z , v = skeleton[:,0] , skeleton[:,1] , skeleton[:,2] , colornodes
        c = np.abs(v)
        ax.scatter(x, y, z, v, c=c, s=150,cmap='jet')
           
        for index in skeleton_bone : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'black',linewidth=5.0)   
        
        for index in skeleton_left_leg : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'black',linewidth=5.0)  
      
        for index in skeleton_right_leg : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'black',linewidth=5.0)
          
        for index in skeleton_left_hand : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'black',linewidth=5.0)
   
        for index in skeleton_right_hand : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'black',linewidth=5.0)

        values = attention_matrix_new.flatten()
    
        cNorm  = colors.Normalize(vmin=0, vmax=values.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='rainbow')
        colorVal = scalarMap.to_rgba(values)
        threshold = np.max(attention_matrix_new)*0.9
        i =0
        for index in combinations:
            first ,second = index[0] , index[1]      
            if attention_matrix_new[first][second] > threshold :
                ax.plot([skeleton[first][0], skeleton[second][0]], [skeleton[first][1], skeleton[second][1]], [skeleton[first][2], skeleton[second][2]], linewidth= 3,color=colorVal[i])
            i += 1
        
        ax.view_init(elev=10.)
        ax.grid(True)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
      


    ani = animation.FuncAnimation(fig, animate, frames=num_frame, repeat=True)
    gif = file_name + '_attention_animation.gif' 
    gif_path = os.path.join(root_name,gif)
    ani.save(gif_path, fps=100)
def normal_draw(joints,file_name):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(projection='3d')
    #ax.set_xlabel('x axis')
    #ax.set_ylabel('z axis')
    #ax.set_zlabel('y axis')
    
    num_frame = len(joints)
    joints = joints.reshape(num_frame,22,3)
    joints = joints[..., [2, 0, 1]]

    combinations = [[i,j] for i in range(0,22) for j in range(0,22) ]

    def animate(frame):
        ax.clear()

        skeleton = joints[frame]
        ax.set_xlim([joints[0][0][0]-offset, joints[0][0][0]+offset])
        ax.set_ylim([joints[0][0][1]-offset, joints[0][0][1]+offset])
        ax.set_zlim([joints[0][0][2]-offset, joints[0][0][2]+offset])
        #ax.set_xlabel('x axis')
        #ax.set_zlabel('y axis')
        #ax.set_ylabel('z axis')
   
        skeleton_bone = [[15,12],[12,9],[9,6],[6,3],[3,0]]
        skeleton_left_leg = [[0,1],[1,4],[4,7],[7,10]]
        skeleton_right_leg = [[0,2],[2,5],[5,8],[8,11]]
        skeleton_left_hand = [[9,13],[13,16],[16,18],[18,20]]
        skeleton_right_hand = [[9,14],[14,17],[17,19],[19,21]]

        
           
        for index in skeleton_bone : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'gold',linewidth=5.0)   
        
        for index in skeleton_left_leg : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'cyan',linewidth=5.0)  
      
        for index in skeleton_right_leg : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'fuchsia',linewidth=5.0)
          
        for index in skeleton_left_hand : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'lime',linewidth=5.0)
   
        for index in skeleton_right_hand : 
            first = index[0]
            second = index[1]
            ax.plot([skeleton[first][0], skeleton[second][0]] , 
                    [skeleton[first][1], skeleton[second][1]] ,
                    [skeleton[first][2], skeleton[second][2]] , 'red',linewidth=5.0)
        
        ax.view_init(elev=10.)

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    for frame in range(num_frame):
        animate(frame)  # Call the animate function to update the plot
        frame_path = f"frame_{frame:04d}.png"
        fig.savefig(frame_path)  # Save the current figure
        print(f"Saved frame {frame} to {frame_path}")
    ani = animation.FuncAnimation(fig, animate, frames=num_frame, repeat=True)
    gif = file_name + '_humanML3D_animation.gif' 
    gif_path = os.path.join(root_name,gif)
    print("gif_path",gif_path)
    ani.save(gif_path, fps=100)

if __name__ == '__main__':

    #attention_node_path   = '/home/weihsin/projects/MotionExpertST-GCN/STAGCN_att_node_results_epoch0.json'
    #attention_matrix_path = '/home/weihsin/projects/MotionExpertST-GCN/STAGCN_att_A_results_epoch0.json'
    node_coordinate_path  = '/home/andrewchen/MotionGPT_v2/HumanML3D/HumanML3D/train.pkl'
    attention_node_path   = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune/att_node_results_epoch9.json'
    attention_matrix_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune/att_A_results_epoch9.json'
    # node_coordinate_path  = '/home/weihsin/datasets/VQA/train_local.pkl'

    #with open(attention_node_path) as f:         attention_node   = json.load(f)
    #with open(attention_matrix_path) as f:       attention_matrix = json.load(f)
    with open(node_coordinate_path, 'rb') as f:   node_coordinate = pkl.load(f)
    num_length = 0
    for item in node_coordinate:
        print(item['video_name'])
        if (item['video_name'] == '013958' ):
            '''
           item['video_name'] == '000215' or item['video_name'] == '000274' or item['video_name'] == '000301' or 
           item['video_name'] == '000591' or item['video_name'] == '000632' or item['video_name'] == '000772' or item['video_name'] == '000846' or 
           item['video_name'] == '000940' or item['video_name'] == '001004' or item['video_name'] == '001034' or item['video_name'] == '001236' or
           item['video_name'] == '001478' or item['video_name'] == '001694' or item['video_name'] == '001905' or item['video_name'] == '002109' or
           item['video_name'] == '002338' or item['video_name'] == '002404' or item['video_name'] == '002603' or item['video_name'] == '002701' or
           item['video_name'] == '002863' or item['video_name'] == '003300' or item['video_name'] == '003305' or item['video_name'] == '003517' or
           item['video_name'] == '003603' ):'''
            print("item",item['video_name'])
            print("item_frame",len(item['features']))
            num_length = len(item['features'])
            key = item['video_name']
            print(type(key))
            #print("attention_node",np.shape(attention_node[key]))

            color_map = 'jet'
            color_node = []
            '''
            attention_node = np.array(attention_node[key])
            attention_matrix_new = np.array(attention_matrix[key][0])
            #for k in range(1,4):
            #    attention_matrix_new += np.array(attention_matrix[key][1])
            for i in range(0,num_length,1 ):
                if num_length <= 131 :
                    # take (i/num_length)*160的 floor vlaue
                    index = int(i*(num_length/131))
                    color_node.append(attention_node[0,index])
                if num_length > 131 :
                    index = int(i*(131/num_length))
                    color_node.append(attention_node[0,index])
            '''
            # draw(item['features'],key,color_node,attention_matrix_new)
            normal_draw(item['features'],key)
            


