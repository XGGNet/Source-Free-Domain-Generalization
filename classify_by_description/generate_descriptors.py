import os
import openai
import json

import itertools

from descriptor_strings import stringtolist

openai.api_key = "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" #FILL IN YOUR OWN HERE


def generate_domain_invariant_prompt(category_name: str):
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


def generate_domain_specific_prompt(domain: str, category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a {domain} photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a {domain} photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a {domain} photo?
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

Q: What are useful features for distinguishing a {category_name} in a photo?
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
    
if __name__ == '__main__':
    # obtain_descriptors_and_save('example', ["bird", "dog", "cat"])

    # obtain_descriptors_and_save('descriptors/descriptors_PACS', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    obtain_ds_descriptors_and_save('','descriptors/PACS/descriptors_pacs_p', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    obtain_ds_descriptors_and_save('art-painting','descriptors/PACS/descriptors_pacs_a', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    obtain_ds_descriptors_and_save('cartoon','descriptors/PACS/descriptors_pacs_c', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])

    obtain_ds_descriptors_and_save('sketch','descriptors/PACS/descriptors_pacs_s', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])



    #  obtain_descriptors_and_save_dg('descriptors/descriptors_PACS_DG', ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"])