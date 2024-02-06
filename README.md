# All you need to know in SportTech
[開會紀錄](https://hackmd.io/5zQZLTOYQYGuZ4nn2sWZVw)
把開會的東西記起來，或是修正別人寫的，讓每個人同步

[零散的筆記](https://hackmd.io/thcD77cGSVinURAAFO3bfg)
請記錄你寫的工具python檔在哪裡

-
# run code
python train_t5_stagcn.py --finetune True > outputloss.txt 
# create a virtual env
```
python3 -m venv motion2text_env
```
# activate the env
```
source motion2text_env/bin/activate
```
# pip install requirements
```
pip install -r requirements.txt
```
# train
```
python train.py
```

# finetune train
```
CUDA_VISIBLE_DEVICES=2 python finetune_train.py > output_finetuen_train.txt
```

# finetuen test
```
python finetune_test.py --model_path ./experiments/seq2seq/save/transformer_bin_class_finetune.pth > output_finetune_test.txt
```

# inference
```
python test.py [model_path]
e.g. python test.py --model_path ./experiments/seq2seq/
```
# Prepare work
### Step 0 : Run 3D skeleton with HybrIK
#### In absolute path /home/weihsin/projects/HybrIK
When you want to run the HybrIK, you can create your own conda environment or use my conda environment.

```
source /home/weihsin/miniconda3/etc/profile.d/conda.sh
conda activate hybrik
## cd 過去做事 裡面有README.md 可以看 但我把他寫到Axel.sh
cd /home/weihsin/projects/HybrIK
```
Before running the following two commands, check the Axel.sh first.
You can see mp4_folder in Axel.sh. Then replace it with the video folder you want to get its corresponding 3-D skeleton.

Next, you can get the file structure like this :

folder : your_video_name
- folder : raw_images
- folder : res_2d_images
- folder : res_images
- mp4 : res_2d_your_video_name
- mp4 : res_your_video_name
- pk : res.pk


```
bash Axel.sh
# 如果執行不起來，就換沒人在用的CUDA就好啦
CUDA_VISIBLE_DEVICES=2 bash Axel.sh     
```


### Step 1 : Trim video
#### In /utils/trim_video_unit.py
You can use this to trim the video.Due to the Skating Dataset has a feature that some frames has many labels on it. Also, not whole sequence of frames in a video have the same label.

Hence, we need to trim the video, and also generate the json file which contain the label about this clip video.
```
python trim_video_unit.py
```

### Step 2 : Combine the information of 3D skeleton and label to a json file 
#### In /utils/dealwithAxel.py 
Replace the variable of "folder3Dskeleton" & "folderlabel" with the path you want 
```
python dealwithAxel.py
```
### Step 3 : Package all things into pklfile
run the pkl file
#### In /utils/pklfile.py 
```
python pklfile.py
```
### Additional : Draw 3D skeleton 
####  In /utils/draw_skeleton.py
Replace the variable of "folder_path" with the path which contain the 3D skeleton folder of every video. 
Output : the animation.gif will be placed in the every folder of video in folder_path.
```
python draw_skeleton.py
```

