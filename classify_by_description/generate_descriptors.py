import os
import openai
import json
from openai_key import key

import itertools
from pdb import set_trace as st

from descriptor_strings import stringtolist

import sys
sys.path.append("..") 
from prompts import *


openai.api_key = key #FILL IN YOUR OWN HERE


def generate_domain_invariant_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful invariant visual features for distinguishing a lemur in a photo with different styles?
A: There are several useful invariant visual features to tell there is a lemur in a photo with different styles:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful invariant visual features for distinguishing a television in a photo with different styles?
A: There are several useful visual features to tell there is a television in a photo with different styles:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo with different styles?
A: There are several useful visual features to tell there is a {category_name} in a photo with different styles:
-
"""


def generate_domain_specific_prompt(domain: str, category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a {domain} photo?
A: There are several useful visual features to tell there is a lemur in a {domain} photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a {domain} photo?
A: There are several useful visual features to tell there is a television in a {domain} photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful visual features for distinguishing a {category_name} in a {domain} photo?
A: There are several useful visual features to tell there is a {category_name} in a {domain} photo:
-
"""


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful visual features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)

def obtain_di_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    
    
    prompts = [generate_domain_invariant_prompt(category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)

def obtain_ds_descriptors_and_save(domain, filename, class_list):
    responses = {}
    descriptors = {}
    
    
    prompts = [generate_domain_specific_prompt(domain, category.replace('_', ' ')) for category in class_list]
    
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)

def obtain_domain_bank_descriptors_and_save(filename, domain_bank_list, class_list):
    responses = {}
    descriptors = {}
    
    prompts = [generate_domain_specific_prompt(domain, category.replace('_', ' ')) for category in class_list for domain in domain_bank_list ]

    # st()
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    

    # st()
    response_texts = [r["text"] for resp in responses for r in resp['choices']] #28

    descriptors_list = [stringtolist(response_text) for response_text in response_texts]

    merged_descriptors_list = []

    for i in range(len(class_list)):
        split_list = descriptors_list[i*len(domain_bank_list):(i+1)*len(domain_bank_list)]
        merged_descriptors_list.append( list(set([item for sublist in split_list for item in sublist])) ) # 存在过的只记录一次
        
    # st()

    descriptors = {cat: descr for cat, descr in zip(class_list, merged_descriptors_list)}



    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)

def obtain_no_merged_domain_bank_descriptors_and_save(filename, domain_bank_list, class_list):
    responses = {}
    descriptors = {}
    
    prompts = [generate_domain_specific_prompt(domain, category.replace('_', ' ')) for category in class_list for domain in domain_bank_list ]

    # st()
    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API
    responses = [openai.Completion.create(model="text-davinci-003",
                                            prompt=prompt_partition,
                                            temperature=0.,
                                            max_tokens=100,
                                            ) for prompt_partition in partition(prompts, 20)]
    

    # st()
    response_texts = [r["text"] for resp in responses for r in resp['choices']] #28

    descriptors_list = [stringtolist(response_text) for response_text in response_texts]

    merged_descriptors_list = []

    for i in range(len(class_list)):
        split_list = descriptors_list[i*len(domain_bank_list):(i+1)*len(domain_bank_list)]
        temp =[]
        for sub in split_list:
            temp.extend(sub)

        merged_descriptors_list.append( temp)
            
    # st()

    descriptors = {cat: descr for cat, descr in zip(class_list, merged_descriptors_list)}



    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    


if __name__ == '__main__':
    # obtain_descriptors_and_save('example', ["bird", "dog", "cat"])

    # obtain_descriptors_and_save('descriptors/descriptors_PACS', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    # obtain_descriptors_and_save('descriptors/VLCS/descriptors_VLCS',categories_list['VLCS'] )

    obtain_descriptors_and_save('descriptors/descriptors_officehome',categories_list['OfficeHome'] )


    # obtain_ds_descriptors_and_save('','descriptors/PACS/descriptors_pacs_p', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    # obtain_ds_descriptors_and_save('art-painting','descriptors/PACS/descriptors_pacs_a', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    # obtain_ds_descriptors_and_save('cartoon','descriptors/PACS/descriptors_pacs_c', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    # obtain_ds_descriptors_and_save('sketch','descriptors/PACS/descriptors_pacs_s', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    # obtain_di_descriptors_and_save('descriptors/descriptors_di_pacs', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])



    # obtain_domain_bank_descriptors_and_save('descriptors/PACS/descriptors_pacs_domain_bank_pacs',domain_banks_list['PACS'], categories_list['PACS'])

    # obtain_domain_bank_descriptors_and_save('descriptors/PACS/descriptors_pacs_domain_bank_combined',domain_banks_list['Combined'], categories_list['PACS'])

    # obtain_domain_bank_descriptors_and_save('descriptors/PACS/descriptors_pacs_domain_bank_expanded',domain_banks_list['Expanded'], categories_list['PACS'])



    # obtain_no_merged_domain_bank_descriptors_and_save('descriptors/PACS/descriptors_pacs_domain_bank_pacs_no_merged',domain_banks_list['PACS'], categories_list['PACS'])


    #  obtain_descriptors_and_save_dg('descriptors/descriptors_PACS_DG', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])