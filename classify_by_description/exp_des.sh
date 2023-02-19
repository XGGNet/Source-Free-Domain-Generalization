# CUDA_VISIBLE_DEVICES=0 python clip_std_prompt.py data/PACS -d PACS -t P -a resnet50 --seed 0 --log logs/clip_std_resnet50/PACS_P


CUDA_VISIBLE_DEVICES=1 python main_DG.py ../data/PACS -d PACS -t P  
CUDA_VISIBLE_DEVICES=1 python main_DG.py ../data/PACS -d PACS -t A 
CUDA_VISIBLE_DEVICES=1 python main_DG.py ../data/PACS -d PACS -t C 
CUDA_VISIBLE_DEVICES=1 python main_DG.py ../data/PACS -d PACS -t S 