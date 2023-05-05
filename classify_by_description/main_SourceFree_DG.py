# from load import *
import torchmetrics
from tqdm import tqdm
from pdb import set_trace as st
import argparse

import json
import numpy as np
import torch
from torch.nn import functional as F

from descriptor_strings import *  # label_to_classname, wordify, modify_descriptor,
import pathlib

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageNet, ImageFolder
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from datasets import _transform, CUBDataset
from collections import OrderedDict
import clip

import builtins

from pdb import set_trace as st

import sys
sys.path.append("..") 

from prompts import *

import utils
from tllib.utils.logger import CompleteLogger
import warnings


def main(args):

    warnings.filterwarnings("ignore")
    logger = CompleteLogger(args.log, args.phase)


    #region
    '''
    load
    '''


    # TODO: fix this... defining global variable to be edited in a function, bad practice
    # unmodify_dict = {}

    # classes_to_load = openai_imagenet_classes
    # hparams['descriptor_fname'] = None

    # IMAGENET_DIR = '/proj/vondrick3/datasets/ImageNet/' # REPLACE THIS WITH YOUR OWN PATH
    # IMAGENETV2_DIR = '/proj/vondrick/datasets/ImageNetV2/' # REPLACE THIS WITH YOUR OWN PATH
    # # CUB_DIR = '/proj/vondrick/datasets/Birds-200-2011/' # REPLACE THIS WITH YOUR OWN PATH
    # CUB_DIR = '/home/lichenxin/data/CUB_200_2011/' # REPLACE THIS WITH YOUR OWN PATH

    # PACS_DIR = '/mnt/Xsky/zyl/dataset/Domainbed/PACS/'


    # PyTorch datasets
    # tfms = _transform(hparams['image_size'])



    # if hparams['dataset'] == 'imagenet':
    #     if hparams['dataset'] == 'imagenet':
    #         dsclass = ImageNet        
    #         hparams['data_dir'] = pathlib.Path(IMAGENET_DIR)
    #         # train_ds = ImageNet(hparams['data_dir'], split='val', transform=train_tfms)
    #         dataset = dsclass(hparams['data_dir'], split='val', transform=tfms)
    #         classes_to_load = None
        
    #         if hparams['descriptor_fname'] is None:
    #             hparams['descriptor_fname'] = 'descriptors_imagenet'
    #         hparams['after_text'] = hparams['label_after_text'] = '.'
            
    #     elif hparams['dataset'] == 'imagenetv2':
    #         hparams['data_dir'] = pathlib.Path(IMAGENETV2_DIR)
    #         dataset = ImageNetV2(location=hparams['data_dir'], transform=tfms)
    #         classes_to_load = openai_imagenet_classes
    #         hparams['descriptor_fname'] = 'descriptors_imagenet'
    #         dataset.classes = classes_to_load

    # elif hparams['dataset'] == 'cub':
    #     # load CUB dataset
    #     hparams['data_dir'] = pathlib.Path(CUB_DIR)
    #     # 这里只用到 test set
    #     dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)

    #     # st()
        
    #     classes_to_load = None #dataset.classes
    #     hparams['descriptor_fname'] = 'descriptors_cub'

    # else:
    #     # hparams['data_dir'] = pathlib.Path(PACS_DIR)

    #     model, preprocess = clip.load(hparams['model_size'], device=hparams['device'], jit=False)

    #     test_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets, split='test',download=True, transform=preprocess, seed=args.seed)

    #     dataset = test_dataset

    #     classes_to_load = None


    #     # hparams['descriptor_fname'] = f'{args.data}/descriptors_{str.lower(args.data)}'

    #     # hparams['descriptor_fname'] = f'{args.data}/descriptors_{str.lower(args.data)}_rank'

    #     hparams['descriptor_fname'] = f'{args.data}/descriptors_{str.lower(args.data)}_ex_domain'

    #     # hparams['descriptor_fname'] = f'PACS/descriptors_pacs_{str.lower(args.targets[0])}'
        
    #     # hparams['descriptor_fname'] = 'PACS/descriptors_pacs_domain_bank_pacs'

    #     # hparams['descriptor_fname'] = 'PACS/descriptors_pacs_domain_bank_pacs_no_merged'

    #     # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    #     print("test_dataset_size: ", len(test_dataset))


    # hparams['descriptor_fname'] = './descriptors/' + hparams['descriptor_fname']


    print("Creating descriptors...")

    # gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load)

    gpt_all_domain_descriptions, unmodify_dict = load_domain_gpt_descriptions(hparams, classes_to_load)


    domain_name = {'P':'photo', 'A': 'art', 'C': 'cartoon', 'S':'sketch' }

    gpt_current_domain_descriptions, unmodify_dict = load_specific_gpt_descriptions(hparams, classes_to_load,domain=domain_name[args.targets[0]])

    gpt_uni_descriptions, unmodify_dict = load_specific_gpt_descriptions(hparams, classes_to_load,domain='')


    label_to_classname = list(gpt_all_domain_descriptions.keys())

    n_classes = len(list(gpt_all_domain_descriptions.keys()))


    seed_everything(hparams['seed'])
    bs = hparams['batch_size']
    '''
    5794
    dataset[0] 
    (image [3,224,224], label 185)
    '''
    dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=16, pin_memory=True)

    print("Loading model...")
    device = torch.device(hparams['device'])
    # load model
    # model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)

    cnt = 0
    for name, parameters in model.named_parameters():
        cnt += parameters.mean()

    model.eval()
    model.requires_grad_(False)

    print("Encoding descriptions...")

    # 得到 description的 text feature
    current_domain_description_encodings = compute_description_encodings(model, gpt_current_domain_descriptions) # description_encodings表示每个class的description的text feature

    all_domain_description_encodings = compute_description_encodings(model, gpt_all_domain_descriptions) # domain_description_encodings表示每个domain的description的text feature

    uni_description_encodings = compute_description_encodings(model, gpt_uni_descriptions) #


    # 得到标准prompt的 text feature
    label_prompt, label_encodings = compute_label_encodings(model)

    sen_prompt, label_sentence_encodings = compute_label_sentence_encodings(model)

    domain_sen_prompt, domain_label_sentence_encodings, class_num, domain_num = compute_domain_label_sentence_encodings(model)
    
    domain_label_sentence_encodings = domain_label_sentence_encodings.view(class_num, domain_num, -1)
    # mean pooling to generate domain unified prompt representations for each class
    avg_domain_label_sentence_encodings = torch.mean(domain_label_sentence_encodings, dim=1)


    _, domain_specific_label_sentence_encodings, _, _ = compute_domain_label_sentence_encodings(model, domain=domain_name[args.targets[0]])


    print("Evaluating...")
    # class_set = [dataset[i][1] for i in range(len(dataset))]
    # num_classes = max(class_set)

    
    # acc = [torchmetrics.Accuracy().to(device)]*6

    acc1 = torchmetrics.Accuracy().to(device)
    acc2 = torchmetrics.Accuracy().to(device)
    acc3 = torchmetrics.Accuracy().to(device)
    acc4 = torchmetrics.Accuracy().to(device)
    acc5 = torchmetrics.Accuracy().to(device)
    acc6 = torchmetrics.Accuracy().to(device)
    acc7 = torchmetrics.Accuracy().to(device)

    for batch_number, (images, labels,_) in enumerate(tqdm(dataloader)):
        # images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)


        image_encodings = model.encode_image(images).float()
        image_encodings = F.normalize(image_encodings) # 和 image_features /= image_features.norm(dim=-1, keepdim=True) 等价


        '''
        计算CLIP_only_label的指标结果
        '''
        
        # image_labels_similarity = 100*image_encodings @ label_encodings.T
        # clip_predictions = image_labels_similarity.argmax(dim=1)
        # clip_acc = clip_label_only_accuracy_metric(image_labels_similarity, labels)
        # clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)

        # image_labels_sen_similarity = 100*image_encodings @ label_sentence_encodings.T

        # clip_sen_predictions = image_labels_sen_similarity.argmax(dim=1)
        # _, clip_sen_predictions_top5 = image_labels_sen_similarity.topk(5,dim=1)

        _ = acc1(image_encodings @ label_sentence_encodings.T, labels) # Standard sentence prompt

        _ = acc2(image_encodings @ domain_specific_label_sentence_encodings.T, labels)

        _ = acc3(image_encodings @ avg_domain_label_sentence_encodings.T, labels) #[640,7] The average prompt of many domains

        
        sim = []
        for ind in range( domain_label_sentence_encodings.shape[1] ):
            sim.append( 100*image_encodings @ domain_label_sentence_encodings[:,ind].T)
        sim = torch.stack(sim, dim=0).mean(dim=0)   #[11,640,7] > [640,7]
        _ = acc4(sim, labels) # The ensemble of prediction by all domain prompts
            
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        for i, (k, v) in enumerate(uni_description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        ## k - class, v - description
            dot_product_matrix = 100*image_encodings @ v.T # 这是一个矩阵
            image_description_similarity[i] = dot_product_matrix # [640,11]
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i]) #这里是取均值
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        _ = acc5(cumulative_tensor, labels)
        

        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        for i, (k, v) in enumerate(current_domain_description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        ## k - class, v - description
            dot_product_matrix = 100*image_encodings @ v.T # 这是一个矩阵
            image_description_similarity[i] = dot_product_matrix # [640,11]
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i]) #这里是取均值
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        _ = acc6(cumulative_tensor, labels)



        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        for i, (k, v) in enumerate(all_domain_description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        ## k - class, v - description
            dot_product_matrix = 100*image_encodings @ v.T # 这是一个矩阵
            image_description_similarity[i] = dot_product_matrix # [640,11]
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i]) #这里是取均值
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        _ = acc7(cumulative_tensor, labels)
        

        if batch_number==0:
            print("\n")

        # show_from_indices(torch.where(clip_sen_predictions!=labels)[0], images, labels, clip_des_predictions, clip_sen_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_sen_similarity)

        # clip_sen_predictions_top5
        # clip_des_predictions_top5

        # st()

        # show_from_indices(torch.where(clip_des_predictions != clip_sen_predictions)[0], images, labels, clip_des_predictions, clip_sen_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_sen_similarity)

        # show_from_indices(torch.where(clip_sen_predictions != labels)[0], images, labels, descr_predictions, clip_sen_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_sen_similarity)

        # st()
        
        

    print("\n")

    print(f'target domain is:{args.targets}')
    print(f'test_dataset size is: {len(dataset)}')
    accuracy_logs = {}


    # accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

#     accuracy_logs["Total CLIP-Label Top-1 Accuracy: "] = 100*clip_label_only_accuracy_metric.compute().item()
# # 
    # accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
    # # accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

    # accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()

    # accuracy_logs["Total Rank Description-based Top-1 Accuracy: "] = 100*ensemble_accuracy_metric.compute().item()

    log_names = ['CLIP', 'CLIP_current_domain', 'CLIP-domain-avg-prompt', 'CLIP-domain-avg', 'Description-Uni', 'Description-current-domain', 'Description-all-domain']
    log_accs = [acc1,acc2,acc3,acc4, acc5, acc6, acc7]

    # for ind, name in enumerate( log_names ):
        # st()
    for ind, name in enumerate( log_names) :
        accuracy_logs[f"Total {name} Top-1 Accuracy: "] = 100*log_accs[ind].compute().item()


    # accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()

    # # accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()
    # accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()

    # accuracy_logs["Total Rank Description-based Top-1 Accuracy: "] = 100*ensemble_accuracy_metric.compute().item()

    # print the dictionary
    for key, value in accuracy_logs.items():
        # print(key, f'{value:.3f}')
          print(key, value, f'** {value:.3f}')


    print("\n")

    #endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS',
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: PACS)')
    parser.add_argument('-s', '--sources', nargs='+', default=None,
                        help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='target domain(s)')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters

    parser.add_argument('-a', '--arch', metavar='ARCH', default='rn50')
    # vitb16, vitb32

    # parser.add_argument('-a', '--arch', metavar='ARCH', default='vitb16')
    # ['RN50',
    #  'RN101',
    #  'RN50x4',
    #  'RN50x16',
    #  'RN50x64',
    #  'ViT-B/32',
    #  'ViT-B/16',
    #  'ViT-L/14',
    #  'ViT-L/14@336px']

    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--freeze-bn', action='store_true', help='whether freeze all bn layers')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='only activated when freeze-bn is True')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--bank_type", type=str, default='Combined')
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--intra', default=0.5, type=float, help='weight of loss intra')
    parser.add_argument('--inter', default=0.05, type=float, help='weight of loss intra')

    parser.add_argument('--rank', action='store_true')

    args = parser.parse_args()

    builtins.args = args

    
    from load import *
    from loading_helpers import *



    main(args)
