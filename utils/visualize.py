import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.dtw import dtw
from scipy.spatial.distance import cdist
import numpy as np
from sklearn import manifold
def create_video():return
def dist_fn(x, y):
    dist = np.sum((x-y)**2)
    return dist

def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame

def align_by_start(start_frames, frames,output_name,query,candi,embeddings=None):
    if embeddings is not None:
        viz_tSNE(embeddings,output_name.replace('.mp4','.png'),use_dtw=True)
    # Create subplots
    nrows = len(frames)
    fig, ax = plt.subplots(ncols=nrows,figsize=(10, 10),tight_layout=True)
  
    ims = []
    def init():
        for k in range(nrows):
            ims.append(ax[k].imshow(unnorm(frames[k][0])))
            ax[k].grid(False)
            ax[k].set_xticks([])
            ax[k].set_yticks([])
        return ims
  
    num_total_frames = min([len(frames[query]), len(frames[candi])])
    def update(i):
        ims[0].set_data(unnorm(frames[query][i]))
        ax[0].set_title('START {} '.format(start_frames[query]), fontsize = 14)
        ims[1].set_data(unnorm(frames[candi][start_frames[1]+i]))
        ax[1].set_title('START {}'.format(start_frames[candi]), fontsize = 14)
      
  # Create animation
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=np.arange(num_total_frames),
        interval=300,
        blit=False)
    anim.save(output_name, dpi=80)

    plt.close('all')

def dtw_align(query_embs, query_frames, key_embs, key_frames, video_path, tsNE_only=False,labels=None,cfg=None):
    """Create aligned videos."""
    nns = align(query_embs, key_embs, True)
    kendalls_embs = [query_embs,key_embs]
    viz_tSNE(kendalls_embs,video_path.split('.mp4')[0]+('_tSNE.jpg'),use_dtw=True)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 10), tight_layout=True)

    ims = []
    title = fig.suptitle("Initializing Video", fontsize=16)
    def init():
        """Initialize the plot for animation."""
        for i in range(2):
            img_display = ax[i].imshow(unnorm(query_frames[0] if i == 0 else key_frames[nns[0]]))
            ims.append(img_display)
            if labels is not None:
                ax[i].set_title(f"Label: {labels[i][0]}")

            ax[i].grid(False)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        title = fig.suptitle("Initializing Video", fontsize=16)
        return ims,title

    
    def update(i):
        """Update plot with next frame."""
        title.set_text(f'Frame {i}/{len(query_frames)}')
        ax[0].set_title(f"Label: {labels[0][i]}")
        ax[1].set_title(f"Label: {labels[1][nns[i]]}")
        ims[0].set_data(unnorm(query_frames[i]))
        ims[1].set_data(unnorm(key_frames[nns[i]]))

        return ims,title
    
    anim = FuncAnimation(
        fig,
        update,
        init_func = init,
        frames=(len(query_frames)),
        interval=50,
        blit=False)
    
    anim.save(video_path, dpi=80)
    plt.close('all')


def align(query_feats, candidate_feats, use_dtw):
    """Align videos based on nearest neighbor or dynamic time warping."""
    if use_dtw:
        _, _, _, path = dtw(query_feats, candidate_feats, dist='sqeuclidean')
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
    else:
        dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
        nns = np.argmin(dists, axis=1)
    return nns

def viz_tSNE(embs,output_path,use_dtw=False,query=0,labels=None,cfg=None):
    embs = [embs[1],embs[0]] ## key (which has the longer length) should be put at first to correctly perform the followings)
    nns = []
    distances = []
    idx = np.arange(len(embs))
    # query_valid_frames = np.where(labels[query]>=cfg.EVAL.KENDALLS_TAU_COMPUTE_LABELS)[0]
    for candidate in range(len(embs)):
        idx[candidate] = candidate
        # candidates_valid_frames = np.where(labels[candidate]>=cfg.EVAL.KENDALLS_TAU_COMPUTE_LABELS)[0]
        nn = align(embs[query], embs[candidate], use_dtw)
        nns.append(nn)
        dis = cdist(embs[query], embs[candidate][nn], dist_fn)
        min_index,min_value = np.argmin(dis,axis=1),np.min(dis,axis=1)
        distances.append((min_index,min_value))
    X = np.empty((0, 128))
    y = []
    frame_idx = []

    nns[1] = np.unique(nns[1]) ## * because we set np.unique here, the max length of nns[1] will be the same as nns[0] (0,0,0,1,2,3 -> 0,1,2,3)
    for i, video_emb in zip(idx, embs):
        for j in range(len(nns[i])): ## so we can only iterate to the max of nns[i]
            X = np.append(X, np.array([video_emb[nns[i][j]]]), axis=0)
            y.append(int(i))
            frame_idx.append(nns[i][j])
    y = np.array(y)
    frame_idx = np.array(frame_idx)

    #t-SNE
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=0).fit_transform(X)
    plt.figure(figsize=(8, 8))

    #Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize

    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(frame_idx[i]), color=plt.cm.Set1(y[i]), 
                fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path)
    plt.close('all')