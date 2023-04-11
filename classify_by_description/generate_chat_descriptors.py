import os
import openai
import json

from pdb import set_trace as st

import itertools

from descriptor_strings import stringtolist

import sys
sys.path.append("..") 
from prompts import *

openai.api_key = "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" #FILL IN YOUR OWN HERE


def generate_dg_prompt(category_name: str):
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

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
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

Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

def generate_domain_prompt(domain_name: str, category_name: str):
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

Q: What are useful features for distinguishing a {category_name} in a {domain_name} photo?
A: There are several useful visual features to tell there is a {category_name} in a {domain_name} photo:
-
"""

def generate_nopre_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""
Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""


# def generate_prompt(category_name: str):
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""
# Q: What are useful visual features for distinguishing a television in a photo?
# A: There are several useful visual features to tell there is a television in a photo:
# - electronic device
# - black or grey
# - a large, rectangular screen
# - a stand or mount to support the screen
# - one or more speakers
# - a power cord
# - input ports for connecting to other devices
# - a remote control

# Q: What are useful features for distinguishing a {category_name} in a photo?
# A: There are several useful visual features to tell there is a {category_name} in a photo:
# -
# """


def generate_rank_prompt(category_name: str):
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

Q: What are useful features for distinguishing a {category_name} in a photo? List these features in descending order of importance for identifying this category”.
A: There are several useful visual features to tell there is a {category_name} in a photo, which are listed according to the order of importance:
-
"""


# def generate_rank_prompt(category_name: str):
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     return f"""
# {category_name}”.
# """


# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list, domain_list=[]):
    responses = {}
    descriptors = {}

    if len(domain_list) ==0:
        prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    else:
        prompts = [generate_domain_prompt(domain_name=domain,category_name = category.replace('_', ' ')) for category in class_list for domain in domain_list ]

    # st()


    # rank_prompts = [generate_rank_prompt(category.replace('_', ' ')) for category in class_list]

    # aaa = [ prompt_partition for prompt_partition in partition(prompts, 20)]
    # bbb = [ prompt_partition for prompt_partition in partition(rank_prompts, 20)]



    
    # most efficient way is to partition all prompts into the max size that can be concurrently queried from the OpenAI API

    # responses = [openai.Completion.create(model="text-davinci-003",
    #                                         prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         ) for prompt_partition in partition(prompts, 20)]
    
    # rank_responses = [openai.Completion.create(model="text-davinci-003",
    #                                         prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=200,
    #                                         ) for prompt_partition in partition(rank_prompts, 20)]


    # responses = [openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                         prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         ) for prompt_partition in partition(prompts, 20)]
    
    # rank_responses = [openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                         prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         ) for prompt_partition in partition(rank_prompts, 20)]
    # a = [prompt_partition for prompt_partition in partition(prompts, 20)]

    
    # responses = [openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                         # prompt=prompt_partition,
    #                                         messages = [{'role': 'user', 'content':'test'}],
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         top_p=1,
    #                                         presence_penalty=0,
    #                                         ) for i in range(0,2)]
    

    # responses = [openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                         # prompt=prompt_partition,
    #                                         messages = [{'role': 'user', 'content':prompt_partition[0]}],
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         top_p=1,
    #                                         presence_penalty=0,
    #                                         ) for prompt_partition in partition(prompts, 20)]
    
    # st()

    # aaa = [prompt_partition for prompt_partition in partition(rank_prompts, 20)]
    rank_responses= []

    for i in range(0,len(prompts),50):

        rank_responses.extend([openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages = [{'role': 'user', 'content':prompt_partition[0]}],
                                                # prompt=prompt_partition,
                                                temperature=0.,
                                                max_tokens=100,
                                                top_p=1,
                                                presence_penalty=0,
                                                ) for prompt_partition in partition(prompts[i:i+50], 1)])
        # st()

    
    # rank_responses = openai.ChatCompletion.create(model="gpt-3.5-turbo",
    #                                         messages = [{'role': 'user', 'content':aaa[0]}],
    #                                         # prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=1000,
    #                                         top_p=1,
    #                                         presence_penalty=0,
    #                                         ) 

    # st()

    #  responses[0]['choices'][0]['message']['content']

    # response_texts = [r["message"]['content'] for resp in responses for r in resp['choices']]
    # descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    # descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # [descriptor[2:] for descriptor in response_texts[0].split('\n') if (descriptor != '')]

    rank_response_texts = [r["message"]['content'] for resp in rank_responses for r in resp['choices']]
    rank_descriptors_list = [stringtolist(response_text) for response_text in rank_response_texts]

    if len(domain_list) == 0:
        rank_descriptors = {cat: descr for cat, descr in zip(class_list, rank_descriptors_list)}
    else:
        rank_descriptors = {}
        # domain_and_class = []
        # for domain in domain_list:
        #     for cat in class_list:
        #         domain_and_class.append(domain+'_'+cat)
        # domain_and_class = [(domain,cat) for domain in domain_list for cat in class_list]
        # st()
        for i,cat in enumerate( class_list ):
            temp = {}
            for j,dom in enumerate( domain_list ):
                temp[dom] = rank_descriptors_list[i*len(domain_list)+j]
            rank_descriptors[cat] = temp
    # st()

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(rank_descriptors, fp)
    
if __name__ == '__main__':
    # obtain_descriptors_and_save('example', ["bird", "dog", "cat"])
    # obtain_descriptors_and_save('descriptors/descriptors_PACS', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])
    #  obtain_descriptors_and_save_dg('descriptors/descriptors_PACS_DG', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    # obtain_descriptors_and_save('descriptors/descriptors_cub_test1', ["bird", "dog", "cat"])

    # with open('descriptors/descriptors_cub.json', 'r') as f:
    #     data = json.load(f)
    
    # class_name = list(data.keys())
    # st()

    obtain_descriptors_and_save('descriptors/PACS/descriptors_pacs_ex_domain', categories_list['PACS'], domain_list = domain_banks_list['Combined'])