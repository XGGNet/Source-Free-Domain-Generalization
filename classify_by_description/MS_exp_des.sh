# #!/bin/bash


# PACS
# dataset='PACS'
# domain_list=('P' 'A' 'C' 'S')

# VLCS
# dataset='VLCS'
# domain_list=('C' 'L' 'S' 'V')

# Office_home
dataset='OfficeHome'
domain_list=('Pr' 'Rw' 'Cl' 'Ar')

#Terra
# dataset='Terra'
# domain_list=('100' '38' '43' '46')

arch='RN50' #    #  'ViTB16', 'ViTB32',

GPU_ID='0'
data_dir='/home/lichenxin/data/Domainbed/'${dataset}

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_MultiSource_DG.py ${data_dir} -d ${dataset} -s ${domain_list[1]} ${domain_list[2]} ${domain_list[3]} -t ${domain_list[0]} -a ${arch} --log logs/MultiSource_${arch}/${dataset}/${domain_list[0]}/description

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_MultiSource_DG.py ${data_dir} -d ${dataset} -s ${domain_list[0]} ${domain_list[2]} ${domain_list[3]} -t ${domain_list[1]} -a ${arch} --log logs/MultiSource_${arch}/${dataset}/${domain_list[1]}/description

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_MultiSource_DG.py ${data_dir} -d ${dataset} -s ${domain_list[0]} ${domain_list[1]} ${domain_list[3]} -t ${domain_list[2]} -a ${arch} --log logs/MultiSource_${arch}/${dataset}/${domain_list[2]}/description

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_MultiSource_DG.py ${data_dir} -d ${dataset} -s ${domain_list[0]} ${domain_list[1]} ${domain_list[2]} -t ${domain_list[3]} -a ${arch} --log logs/MultiSource_${arch}/${dataset}/${domain_list[3]}/description




