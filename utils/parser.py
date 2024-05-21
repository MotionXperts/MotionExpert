import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', default=0, type=int, help='rank in local processes')
    parser.add_argument('--local',type=bool,default = True)
    parser.add_argument('--finetune', type=bool,default=False)
    parser.add_argument('--data', default='/home/weihsin/datasets/FigureSkate/HumanML3D_g/global_human_train.pkl')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--prefix', default='HumanML', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--pretrained', type=bool,default=False)
    parser.add_argument('--test_data', default='/home/weihsin/datasets/FigureSkate/HumanML3D_g/global_human_test.pkl')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--ckpt', default=None, help='absolute path to the checkpoint')
    parser.add_argument('--cfg_file',default=None, type=str, help='cfg for alignment module, try to use \
                                                                    /home/c1l1mo/projects/VideoAlignment/result/scl_skating_long_50/config.yaml \
                                                                        for now.')

    args = parser.parse_args()
    if(args.local):
        args.data        = '/home/weihsin/datasets/FigureSkate/HumanML3D_l/local_human_train.pkl'
        args.prefix      = 'Local'
        args.test_data   = '/home/weihsin/datasets/FigureSkate/HumanML3D_l/local_human_test.pkl'

    if(args.finetune):
        print("make sure ur using Cindy or Tommy's finetune")
        args.data        = '/home/weihsin/datasets/VQA/train_local.pkl'
        args.prefix      = 'Finetune'
        args.test_data   = '/home/weihsin/datasets/VQA/test_local.pkl'

    # if(args.pretrained):
    #     weight           = '/home/weihsin/projects/MotionExpert/models_local_new/Local_epoch50.pt'
    #     model_state_dict = model.state_dict()
    #     state_dict = torch.load(weight)
    #     pretrained_dict_1 = {k: v for k, v in state_dict.items() if k in model_state_dict}
    #     model_state_dict.update(pretrained_dict_1)
    #     model.load_state_dict(model_state_dict)
    args.out_dir = os.path.join('results', args.out_dir)
    args.result_dir = os.path.join(args.out_dir,'jsons')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir,exist_ok=True)

    return args