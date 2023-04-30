# CUDA_VISIBLE_DEVICES=0 python clip_std_prompt.py /mnt/Xsky/zyl/dataset/Domainbed/PACS -d PACS -t P -a vitb32 --seed 0 --log logs/clip_std_vitb32/PACS_P


CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py /mnt/Xsky/zyl/dataset/Domainbed/PACS -d PACS -t P -a vitb32 --seed 0 --log logs/clip_domainbank_mean_vitb32/PACS_P 
CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py /mnt/Xsky/zyl/dataset/Domainbed/PACS -d PACS -t A -a vitb32 --seed 0 --log logs/clip_domainbank_mean_vitb32/PACS_A 
CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py /mnt/Xsky/zyl/dataset/Domainbed/PACS -d PACS -t C -a vitb32 --seed 0 --log logs/clip_domainbank_mean_vitb32/PACS_C 
CUDA_VISIBLE_DEVICES=0 python cae_mean_pool.py /mnt/Xsky/zyl/dataset/Domainbed/PACS -d PACS -t S -a vitb32 --seed 0 --log logs/clip_domainbank_mean_vitb32/PACS_S