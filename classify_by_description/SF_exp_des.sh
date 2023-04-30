# #!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python clip_std_prompt.py data/PACS -d PACS -t P -a resnet50 --seed 0 --log logs/clip_std_resnet50/PACS_P


# PACS

# CUDA_VISIBLE_DEVICES=0 python main_DG.py ${data_dir} -d ${dataset} -t P --log logs/clip_resnet50/PACS/P/description
# CUDA_VISIBLE_DEVICES=0 python main_DG.py /home/lichenxin/data/Domainbed/PACS-t A --log logs/clip_resnet50/PACS/A/description
# CUDA_VISIBLE_DEVICES=1 python main_DG.py /home/lichenxin/data/Domainbed/PACS -d PACS -t C --log logs/clip_resnet50/PACS/C/description
# CUDA_VISIBLE_DEVICES=1 python main_DG.py  /home/lichenxin/data/Domainbed/PACS -d PACS -t S --log logs/clip_resnet50/PACS/S/description




# PACS
dataset='PACS'
# domain_list=('P')
domain_list=('P' 'A' 'C' 'S')


# VLCS
# dataset='VLCS'
# domain_list=('C' 'L' 'S' 'V')

# Office_home
# dataset='OfficeHome'
# domain_list=('Pr' 'Rw' 'Cl' 'Ar')

#Terra
# dataset='Terra'
# domain_list=('100' '38' '43' '46')

arch='VITB32' #    #  'ViTB16', 'ViTB32',

GPU_ID='4'
data_dir='/mnt/Xsky/zyl/dataset//Domainbed/'${dataset}

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_SourceFree_DG.py ${data_dir} -d ${dataset} -t ${domain_list[0]} -a ${arch} --log logs/SourceFree_${arch}/${dataset}/${domain_list[0]}/description

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_SourceFree_DG.py ${data_dir} -d ${dataset} -t ${domain_list[1]} -a ${arch} --log logs/SourceFree_${arch}/${dataset}/${domain_list[1]}/description

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_SourceFree_DG.py ${data_dir} -d ${dataset} -t ${domain_list[2]} -a ${arch} --log logs/SourceFree_${arch}/${dataset}/${domain_list[2]}/description

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_SourceFree_DG.py ${data_dir} -d ${dataset} -t ${domain_list[3]} -a ${arch} --log logs/SourceFree_${arch}/${dataset}/${domain_list[3]}/description




