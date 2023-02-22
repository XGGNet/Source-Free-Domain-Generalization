# CUDA_VISIBLE_DEVICES=0 python clip_std_prompt.py data/PACS -d PACS -t P -a resnet50 --seed 0 --log logs/clip_std_resnet50/PACS_P


CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py data/PACS -d PACS -t P -a resnet50 --seed 0 --log logs/clip_domainbank_mean_resnet50/PACS_P 
CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py data/PACS -d PACS -t A -a resnet50 --seed 0 --log logs/clip_domainbank_mean_resnet50/PACS_A 
CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py data/PACS -d PACS -t C -a resnet50 --seed 0 --log logs/clip_domainbank_mean_resnet50/PACS_C 
CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py data/PACS -d PACS -t S -a resnet50 --seed 0 --log logs/clip_domainbank_mean_resnet50/PACS_S