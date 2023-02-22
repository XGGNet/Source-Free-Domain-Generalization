from load import *
import torchmetrics
from tqdm import tqdm
from pdb import set_trace as st
import argparse

import sys
sys.path.append("..") 

import utils
from tllib.utils.logger import CompleteLogger
import warnings



seed_everything(hparams['seed'])

warnings.filterwarnings("ignore")
logger = CompleteLogger('logs/clip_resnet50/CUB', 'train')


bs = hparams['batch_size']

# st()
'''
5794
dataset[0] 
(image [3,224,224], label 185)
'''

dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = torch.device(hparams['device'])
# load model
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False)
model.eval()
model.requires_grad_(False)

print("Encoding descriptions...")

# 得到 description的 text feature
description_encodings = compute_description_encodings(model)

# 得到标准prompt的 text feature
label_encodings = compute_label_encodings(model)


print("Evaluating...")
# class_set = [dataset[i][1] for i in range(len(dataset))]
# num_classes = max(class_set)

lang_accuracy_metric = torchmetrics.Accuracy().to(device)
lang_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

clip_accuracy_metric = torchmetrics.Accuracy().to(device)
clip_accuracy_metric_top5 = torchmetrics.Accuracy(top_k=5).to(device)

for batch_number, batch in enumerate(tqdm(dataloader)):
    images, labels = batch
    
    images = images.to(device)
    labels = labels.to(device)
    
    image_encodings = model.encode_image(images)
    image_encodings = F.normalize(image_encodings)

    '''
    计算CLIP_std_prompt的指标结果
    '''
    
    image_labels_similarity = image_encodings @ label_encodings.T
    clip_predictions = image_labels_similarity.argmax(dim=1)
    
    clip_acc = clip_accuracy_metric(image_labels_similarity, labels)
    clip_acc_top5 = clip_accuracy_metric_top5(image_labels_similarity, labels)
    
    
    image_description_similarity = [None]*n_classes
    image_description_similarity_cumulative = [None]*n_classes
    
    for i, (k, v) in enumerate(description_encodings.items()): # You can also vectorize this; it wasn't much faster for me
    ## k - class, v - description

        dot_product_matrix = image_encodings @ v.T # 这是一个矩阵
        
        image_description_similarity[i] = dot_product_matrix
        image_description_similarity_cumulative[i] = aggregate_similarity(image_description_similarity[i]) #这里是取均值
        
        
    # create tensor of similarity means
    cumulative_tensor = torch.stack(image_description_similarity_cumulative,dim=1)
        
    
    descr_predictions = cumulative_tensor.argmax(dim=1)
    
    
    lang_acc = lang_accuracy_metric(cumulative_tensor.softmax(dim=-1), labels)
    lang_acc_top5 = lang_accuracy_metric_top5(cumulative_tensor.softmax(dim=-1), labels)
    
    

print("\n")

accuracy_logs = {}
accuracy_logs["Total Description-based Top-1 Accuracy: "] = 100*lang_accuracy_metric.compute().item()
accuracy_logs["Total Description-based Top-5 Accuracy: "] = 100*lang_accuracy_metric_top5.compute().item()

accuracy_logs["Total CLIP-Standard Top-1 Accuracy: "] = 100*clip_accuracy_metric.compute().item()
accuracy_logs["Total CLIP-Standard Top-5 Accuracy: "] = 100*clip_accuracy_metric_top5.compute().item()

# print the dictionary
print("\n")
for key, value in accuracy_logs.items():
    print(key, value)

show_from_indices(torch.where(descr_predictions != clip_predictions)[0], images, labels, descr_predictions, clip_predictions, image_description_similarity=image_description_similarity, image_labels_similarity=image_labels_similarity)

