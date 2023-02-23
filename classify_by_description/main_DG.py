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

from loading_helpers import *

from pdb import set_trace as st

import sys
sys.path.append("..") 

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

    hparams = {}
    # hyperparameters

    # hparams['model_size'] = "ViT-B/32" 

    convert_dict = {'RN50': 'RN50', 'VITB16':'ViT-B/16', 'VITB32': 'ViT-B/32'}

    hparams['model_size'] = convert_dict[args.arch]
    
    # "RN50" 

    # Options:
    # ['RN50',
    #  'RN101',
    #  'RN50x4',
    #  'RN50x16',
    #  'RN50x64',
    #  'ViT-B/32',
    #  'ViT-B/16',
    #  'ViT-L/14',
    #  'ViT-L/14@336px']

    # hparams['dataset'] = 'cub'
    hparams['dataset'] = 'pacs'

    hparams['batch_size'] = 64*10
    hparams['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # st()

    hparams['category_name_inclusion'] = 'prepend' #'append' 'prepend'

    hparams['apply_descriptor_modification'] = True

    hparams['verbose'] = False
    hparams['image_size'] = 224
    if hparams['model_size'] == 'ViT-L/14@336px' and hparams['image_size'] != 336:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 336.')
        hparams['image_size'] = 336
    elif hparams['model_size'] == 'RN50x4' and hparams['image_size'] != 288:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
        hparams['image_size'] = 288
    elif hparams['model_size'] == 'RN50x16' and hparams['image_size'] != 384:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
        hparams['image_size'] = 384
    elif hparams['model_size'] == 'RN50x64' and hparams['image_size'] != 448:
        print(f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.')
        hparams['image_size'] = 448

    hparams['before_text'] = ""
    
    hparams['label_before_text'] = ""
    # hparams['label_before_text'] = "a photo of a "

    hparams['between_text'] = ', '
    # hparams['between_text'] = ' '
    # hparams['between_text'] = ''
    hparams['after_text'] = ''
    hparams['unmodify'] = True
    # hparams['after_text'] = '.'
    # hparams['after_text'] = ' which is a type of bird.'
    hparams['label_after_text'] = ''
    # hparams['label_after_text'] = ' which is a type of bird.'
    hparams['seed'] = 1

    # TODO: fix this... defining global variable to be edited in a function, bad practice
    # unmodify_dict = {}

    # classes_to_load = openai_imagenet_classes
    hparams['descriptor_fname'] = None

    IMAGENET_DIR = '/proj/vondrick3/datasets/ImageNet/' # REPLACE THIS WITH YOUR OWN PATH
    IMAGENETV2_DIR = '/proj/vondrick/datasets/ImageNetV2/' # REPLACE THIS WITH YOUR OWN PATH
    # CUB_DIR = '/proj/vondrick/datasets/Birds-200-2011/' # REPLACE THIS WITH YOUR OWN PATH
    CUB_DIR = '/home/lichenxin/data/CUB_200_2011/' # REPLACE THIS WITH YOUR OWN PATH

    PACS_DIR = '/home/lichenxin/code/Source-Free-Domain-Generalization/data/PACS/'


    # PyTorch datasets
    tfms = _transform(hparams['image_size'])



    if hparams['dataset'] == 'imagenet':
        if hparams['dataset'] == 'imagenet':
            dsclass = ImageNet        
            hparams['data_dir'] = pathlib.Path(IMAGENET_DIR)
            # train_ds = ImageNet(hparams['data_dir'], split='val', transform=train_tfms)
            dataset = dsclass(hparams['data_dir'], split='val', transform=tfms)
            classes_to_load = None
        
            if hparams['descriptor_fname'] is None:
                hparams['descriptor_fname'] = 'descriptors_imagenet'
            hparams['after_text'] = hparams['label_after_text'] = '.'
            
        elif hparams['dataset'] == 'imagenetv2':
            hparams['data_dir'] = pathlib.Path(IMAGENETV2_DIR)
            dataset = ImageNetV2(location=hparams['data_dir'], transform=tfms)
            classes_to_load = openai_imagenet_classes
            hparams['descriptor_fname'] = 'descriptors_imagenet'
            dataset.classes = classes_to_load

    elif hparams['dataset'] == 'cub':
        # load CUB dataset
        hparams['data_dir'] = pathlib.Path(CUB_DIR)
        # 这里只用到 test set
        dataset = CUBDataset(hparams['data_dir'], train=False, transform=tfms)

        # st()
        
        classes_to_load = None #dataset.classes
        hparams['descriptor_fname'] = 'descriptors_cub'

    else:
        # hparams['data_dir'] = pathlib.Path(PACS_DIR)

        model, preprocess = clip.load(hparams['model_size'], device=hparams['device'], jit=False)

        test_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets, split='test',download=True, transform=preprocess, seed=args.seed)

        dataset = test_dataset

        classes_to_load = None


        hparams['descriptor_fname'] = f'{args.data}/descriptors_{str.lower(args.data)}'

        # hparams['descriptor_fname'] = f'PACS/descriptors_pacs_{str.lower(args.targets[0])}'
        
        # hparams['descriptor_fname'] = 'PACS/descriptors_pacs_domain_bank_pacs'

        # hparams['descriptor_fname'] = 'PACS/descriptors_pacs_domain_bank_pacs_no_merged'

        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        print("test_dataset_size: ", len(test_dataset))

    # st()

    hparams['descriptor_fname'] = './descriptors/' + hparams['descriptor_fname']

        

    print("Creating descriptors...")

    gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, classes_to_load)
    label_to_classname = list(gpt_descriptions.keys())

    # st()

    n_classes = len(list(gpt_descriptions.keys()))

    def compute_description_encodings(model):
        description_encodings = OrderedDict()

        for k, v in gpt_descriptions.items():
            tokens = clip.tokenize(v).to(hparams['device'])
            description_encodings[k] = F.normalize(model.encode_text(tokens).float())
        return description_encodings

    def compute_label_encodings(model):

        # st()

        prompts = [hparams['label_before_text'] + wordify(l) + hparams['label_after_text']  for l in label_to_classname]

        # st()

        # label_encodings = F.normalize(
        #     model.encode_text(
        #     clip.tokenize(
        #     prompts
        #     ).to(hparams['device'])
        #     )
        #     )

        with torch.no_grad():
            text_inputs = torch.cat( [clip.tokenize(prompt) for prompt in prompts]).to(hparams['device'])
            text_features = model.encode_text(text_inputs).float()
            label_encodings = F.normalize(text_features)


        # st()
        # label_encodings > [7, 1024]
        return prompts, label_encodings

    def compute_label_sentence_encodings(model):


        prompts = ["a photo of a "+ wordify(l)  for l in label_to_classname]

        # flag=0

        # st()

        # label_encodings = F.normalize(
        #     model.encode_text( clip.tokenize(prompts).to(hparams['device']) ) #7,1024
        #     )

        # !!!
        # 同时编码多个token 和 一个token的结果是不一样的..

        with torch.no_grad():
            text_inputs = torch.cat( [clip.tokenize(prompt) for prompt in prompts]).to(hparams['device'])
            text_features = model.encode_text(text_inputs).float()
            label_encodings = F.normalize(text_features)


        # with torch.no_grad():
        #     for prompt in prompts:
        #         text = clip.tokenize(prompt).to(device)
        #         text_features = model.encode_text(text).float()

        #         text_features = text_features / text_features.norm(dim=1, keepdim=True)

        #         if flag==0:
        #             # text features have to be placed on cpu beacuse of the limitation of gpu memorys.
        #             text_features_all = text_features.to('cpu')
        #             flag = 1
        #         else:
        #             text_features_all =  torch.cat((text_features_all, text_features.to('cpu')), dim=0)

        # st()
        # label_encodings > [7, 1024]
        return prompts, label_encodings

    def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
        if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
        elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
        elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
        else: raise ValueError("Unknown aggregate_similarity")

    def show_from_indices(indices, images, labels=None, predictions=None, predictions2 = None, n=None, image_description_similarity=None, image_labels_similarity=None):
        if indices is None or (len(indices) == 0):
            print("No indices provided")
            return
        
        if n is not None:
            indices = indices[:n]
        
        for index in indices:
            # show_single_image(images[index])
        
            print(f"Index: {index}")
            if labels is not None:
                true_label = labels[index]
                true_label_name = label_to_classname[true_label]
                print(f"True label: {true_label_name}")
            if predictions is not None:
                predicted_label = predictions[index]
                predicted_label_name = label_to_classname[predicted_label]
                print(f"Predicted label (ours): {predicted_label_name}")
            if predictions2 is not None:
                predicted_label2 = predictions2[index]
                predicted_label_name2 = label_to_classname[predicted_label2]
                print(f"Predicted label 2 (CLIP): {predicted_label_name2}")

            save_single_image(os.path.join(args.log,'visualize',f'{index}_GT-{true_label_name}_des-{predicted_label_name}_clip-{predicted_label_name2}.png'), images[index])
            
            print("\n")
            
            if image_labels_similarity is not None:
                if labels is not None:
                    print(f"Total similarity to {true_label_name} (true label) labels: {image_labels_similarity[index][true_label].item()}")
                if predictions is not None:
                    if labels is not None and true_label_name == predicted_label_name: 
                        print("Predicted label (ours) matches true label")
                    else: 
                        print(f"Total similarity to {predicted_label_name} (predicted label) labels: {image_labels_similarity[index][predicted_label].item()}")
                if predictions2 is not None:
                    if labels is not None and true_label_name == predicted_label_name2: 
                        print("Predicted label 2 (CLIP) matches true label")
                    elif predictions is not None and predicted_label_name == predicted_label_name2: 
                        print("Predicted label 2 (CLIP) matches predicted label 1")
                    else: 
                        print(f"Total similarity to {predicted_label_name2} (predicted label 2) labels: {image_labels_similarity[index][predicted_label2].item()}")
            
                print("\n")
            
            if image_description_similarity is not None:
                if labels is not None:
                    print_descriptor_similarity(image_description_similarity, index, true_label, true_label_name, "true")
                    print("\n")
                if predictions is not None:
                    if labels is not None and true_label_name == predicted_label_name:
                        print("Predicted label (ours) same as true label")
                        # continue
                    else:
                        print_descriptor_similarity(image_description_similarity, index, predicted_label, predicted_label_name, "descriptor")
                    print("\n")
                if predictions2 is not None:
                    if labels is not None and true_label_name == predicted_label_name2:
                        print("Predicted label 2 (CLIP) same as true label")
                        # continue
                    elif predictions is not None and predicted_label_name == predicted_label_name2: 
                        print("Predicted label 2 (CLIP) matches predicted label 1")
                    else:
                        print_descriptor_similarity(image_description_similarity, index, predicted_label2, predicted_label_name2, "CLIP")
                print("\n")

    def print_descriptor_similarity(image_description_similarity, index, label, label_name, label_type="provided"):
        # print(f"Total similarity to {label_name} ({label_type} label) descriptors: {aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
        print(f"Total similarity to {label_name} ({label_type} label) descriptors:")

        # print(f"Average:\t\t{100.*aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")
        print(f"Average:\t\t{aggregate_similarity(image_description_similarity[label][index].unsqueeze(0)).item()}")

        label_descriptors = gpt_descriptions[label_name]
        for k, v in sorted(zip(label_descriptors, image_description_similarity[label][index]), key = lambda x: x[1], reverse=True):
            k = unmodify_dict[label_name][k]
            # print("\t" + f"matched \"{k}\" with score: {v}")
            # print(f"{k}\t{100.*v}")
            print(f"{k}\t{v}")
            
    def print_max_descriptor_similarity(image_description_similarity, index, label, label_name):
        max_similarity, argmax = image_description_similarity[label][index].max(dim=0)
        label_descriptors = gpt_descriptions[label_name]
        print(f"I saw a {label_name} because I saw {unmodify_dict[label_name][label_descriptors[argmax.item()]]} with score: {max_similarity.item()}")
        
    def show_misclassified_images(images, labels, predictions, n=None, 
                                image_description_similarity=None, 
                                image_labels_similarity=None,
                                true_label_to_consider: int = None, 
                                predicted_label_to_consider: int = None):
        misclassified_indices = yield_misclassified_indices(images, labels=labels, predictions=predictions, true_label_to_consider=true_label_to_consider, predicted_label_to_consider=predicted_label_to_consider)
        if misclassified_indices is None: return
        show_from_indices(misclassified_indices, images, labels, predictions, 
                        n=n,
                        image_description_similarity=image_description_similarity, 
                        image_labels_similarity=image_labels_similarity)

    def yield_misclassified_indices(images, labels, predictions, true_label_to_consider=None, predicted_label_to_consider=None):
        misclassified_indicators = (predictions.cpu() != labels.cpu())
        if true_label_to_consider is not None:
            misclassified_indicators = misclassified_indicators & (labels.cpu() == true_label_to_consider)
        if predicted_label_to_consider is not None:
            misclassified_indicators = misclassified_indicators & (predictions.cpu() == predicted_label_to_consider)
            
        if misclassified_indicators.sum() == 0:
            output_string = 'No misclassified images found'
            if true_label_to_consider is not None:
                output_string += f' with true label {label_to_classname[true_label_to_consider]}'
            if predicted_label_to_consider is not None:
                output_string += f' with predicted label {label_to_classname[predicted_label_to_consider]}'
            print(output_string + '.')
                
            return
        
        misclassified_indices = torch.arange(images.shape[0])[misclassified_indicators]
        return misclassified_indices

    #endregion

    #region
    '''
    run
    '''
    from PIL import Image

    def predict_and_show_explanations(images, model, labels=None, description_encodings=None, label_encodings=None, device=None):
        if type(images) == Image:
            images = tfms(images)
            
        if images.device != device:
            images = images.to(device)
            labels = labels.to(device)

        image_encodings = model.encode_image(images)
        image_encodings = F.normalize(image_encodings)
        
        
        image_labels_similarity = image_encodings @ label_encodings.T
        clip_predictions = image_labels_similarity.argmax(dim=1)

        n_classes = len(description_encodings)
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
            
            
            dot_product_matrix = image_encodings @ v.T
            
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i])
            
            
        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
            
        
        descr_predictions = cumulative_tensor.argmax(dim=1)
        
        
        show_from_indices(torch.arange(images.shape[0]), images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)




    seed_everything(hparams['seed'])

    bs = hparams['batch_size']

    # st()
    '''
    5794
    dataset[0] 
    (image [3,224,224], label 185)
    '''
    # st()
    dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=16, pin_memory=True)

    # st()

    print("Loading model...")

    device = torch.device(hparams['device'])
    # load model
    # model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)

    cnt = 0
    for name, parameters in model.named_parameters():
        cnt += parameters.mean()

    # st()
        

    model.eval()
    model.requires_grad_(False)

    print("Encoding descriptions...")

    # 得到 description的 text feature
    description_encodings = compute_description_encodings(model)

    # 得到标准prompt的 text feature
    label_prompt, label_encodings = compute_label_encodings(model)

    sen_prompt, label_sentence_encodings = compute_label_sentence_encodings(model)

    # st()


    print("Evaluating...")
    # class_set = [dataset[i][1] for i in range(len(dataset))]
    # num_classes = max(class_set)

    # lang_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    # lang_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass",num_classes=n_classes, top_k=5).to(device)

    # clip_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes).to(device)
    # clip_accuracy_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes,top_k=5).to(device)

    
    ensemble_accuracy_metric = torchmetrics.Accuracy().to(device)

    lang_accuracy_metric = torchmetrics.Accuracy().to(device)
    # lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    clip_accuracy_metric = torchmetrics.Accuracy().to(device)
    # clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    clip_label_only_accuracy_metric = torchmetrics.Accuracy().to(device)
    


    for batch_number, (images, labels,_) in enumerate(tqdm(dataloader)):
        # images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)

        # st()

   
        image_encodings = model.encode_image(images).float()


        image_encodings = F.normalize(image_encodings) # 和 image_features /= image_features.norm(dim=-1, keepdim=True) 等价



        '''
        计算CLIP_only_label的指标结果
        '''
        
        # image_labels_similarity = 100*image_encodings @ label_encodings.T
        # clip_predictions = image_labels_similarity.argmax(dim=1)
        # clip_acc = clip_label_only_accuracy_metric(image_labels_similarity, labels)
        # clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)

        image_labels_sen_similarity = 100*image_encodings @ label_sentence_encodings.T
        clip_sen_predictions = image_labels_sen_similarity.argmax(dim=1)
        clip_sen_acc = clip_accuracy_metric(image_labels_sen_similarity, labels)
        


        # st()
        
        image_description_similarity = [None]*n_classes
        image_description_similarity_cumulative = [None]*n_classes
        
        for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
        ## k - class, v - description

            dot_product_matrix = 100*image_encodings @ v.T # 这是一个矩阵
            
            image_description_similarity[i] = dot_product_matrix
            image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i]) #这里是取均值
            
            
        # create tensor of similarity means
        cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)

        descr_predictions = cumulative_tensor.argmax(dim=1)
        
        lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
        # lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)

        ensemble_similarity = (image_labels_sen_similarity + cumulative_tensor) / 2

        ensem_acc = ensemble_accuracy_metric(ensemble_similarity.argmax(dim=1), labels)

        if batch_number==0:
            print("\n")

        # show_from_indices(torch.where(descr_predictions != clip_sen_predictions)[0], images, labels, descr_predictions, clip_sen_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_sen_similarity)

        # show_from_indices(torch.where(clip_sen_predictions != labels)[0], images, labels, descr_predictions, clip_sen_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_sen_similarity)

        # st()
        
        

    print("\n")

    print(f'target domain is:{args.targets}')
    print(f'test_dataset size is: {len(dataset)}')
    accuracy_logs = {}


    # accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

#     accuracy_logs["Total CLIP-Label Top-1 Accuracy: "] = 100*clip_label_only_accuracy_metric.compute().item()
# # 
    accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
    # accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

    accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()

    accuracy_logs["Total Ensemble Top-1 Accuracy: "] = 100*ensemble_accuracy_metric.compute().item()

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
    args = parser.parse_args()

    main(args)
