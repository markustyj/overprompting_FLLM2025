### functions that are used to support few-shot prompting

# F: functional, A: availability, FT: fault tolerance, LF: look & feel, MN: maintainability, O: operational,
# PE: performance, PO: portability, SC: scalability, SE: security, US: usability, RE: reliability

# 0: availability, 1: performance, 2: maintainability, 3: portability, 4: scalability, 5: security, 6: fault tolerance

import numpy as np
import csv


def get_all_prompt( dataset, method, num_shot, bi_classification):
    """ loaded_data is a N x 3 array for promise dataset, N x 2 array for nfr_so dataset
    input
        dataset, string name of the evaluated dataset
        method, string name of the used method   
        num_shot, int, the number of few-shot examples in the prompt
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        the list of prompts, string list
    """
    prompt_list = []

    loaded_test_data = load_test_dataset(dataset)
    for requirement_index in range (len(loaded_test_data)):
        input_requirement = loaded_test_data[requirement_index][0]
        prompt = prompt_construction( dataset, method, num_shot, bi_classification, requirement_index, input_requirement )
        prompt_list.append(prompt)

    return prompt_list



def load_test_dataset(dataset):
    """load the input requirement/sentence in the test dataset for prompt construction
    input
        dataset, string name
    output
        loaded_test_data: the loaded list file in format [ [first-row requirement, first-row multi-class category, first-row binary-class category,.. ], [second-row requirement, second-row multi-class category, .. ], [], ... ]
    """
    loaded_test_data = []  
    if dataset == "nfr":
        file_path = '/Users/yongjiantang/Desktop/tang/code/re_genai/data/processed_nfr_so_test.csv'
    elif dataset == "promise":
        file_path = '/Users/yongjiantang/Desktop/tang/code/re_genai/data/processed_promise_test.csv'
    elif dataset == "pure":
        file_path = '/Users/yongjiantang/Desktop/tang/code/re_genai/data/processed_pure_test.csv'
    else: 
        raise ValueError ( "dataset could only be 'nfr' or 'promise' or 'pure' " )

    with open(file_path, 'r') as file:  
        csv_reader = csv.reader(file)  
        for row in csv_reader:  
            loaded_test_data.append(row)  
    return loaded_test_data[1:]  # the first row of title is not loaded, only the content is necessary.



def prompt_construction(dataset, method, num_shot, bi_classification, requirement_index, input_requirement):
    """ construct the prompt for each single input sentence/requirement
    input
        dataset, string name of the evaluated dataset
        method, string name of the used method    
        num_shot, int, the number of few-shot examples in the prompt
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        string_prompt: the string prompt with each corresponding input sentence/requirement.

    """

    if bi_classification:
        prompt_base = ("Please classify the given software requirements into functional requirement or non-functional requirement. "
                #"The answer should be in format {the given requirement: functional requirement or non-functional requirement}."
                "The answer should be one word, i.e. Functional or Non-functional. \n"
                )
    elif bi_classification == False:    
        prompt_base = ("Please classify the given software requirements into the following categories: "
                       "Functional, Availability, Fault Tolerance, Legal, Look and Feel, Maintainability, Operational, "
                       "Performance, Portability, Scalability, Security, Usability. "
                       #"The answer should be in format {the given requirement: the name of classified category}."
                       "The answer should be very concise and short, i.e. only one of the above-mentioned categories."
                        )
    else: 
        raise ValueError("bi_classification must be True or False.")    

    if dataset == "nfr":
        if bi_classification:
            raise ValueError("nfr does not has binary classification data'.")
        
        prompt_base = ("Please classify the given nonfunctional software requirements into the following categories: "
                       "Availability, Fault Tolerance, Maintainability, "
                       "Performance, Portability, Scalability, Security. "
                       #"The answer should be in format {the given requirement: the name of the classified category}."
                       "The answer should be very concise and short, i.e. only one of the above-mentioned categories."

                        ) 

    prompt = ""
    if num_shot == 0:
        prompt = prompt_base + "The given requirement: " + input_requirement
    elif num_shot > 0:    
        prompt = prompt_base + get_str_few_shot_examples(requirement_index, dataset, method, num_shot, bi_classification) + "\nNow, classify the following given requirement: " + input_requirement
    else: 
        raise ValueError("num_shot has to be a decimal number.")    
 

    return prompt       



def get_str_few_shot_examples(requirement_index, dataset, method, num_shot, bi_classification):
    """ get the part of textual few-shot examples for the prompt
    input
        requirement_index, the index of the requirement sentence, which means we construct the prompt for each single test requirement sentence one by one.
        dataset, string name of the evaluated dataset
        method, string name of the used method          
        num_shot, int, the number of few-shot examples in the prompt
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        string_few_shot_examples: the string of selected few_shot examples, which will be integrated into the formal prompt.

    """
    example_str = ""
    few_shot_list = load_few_shot_list(dataset, method)

    for i in range(num_shot):
        if bi_classification:       # the requirement                               # the category of this requirement
            example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][2] + "\n"
        else: 
            example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][1] + "\n"

    return "\nBelow are some demonstration examples for you to learn, which consist of a software requirement and its category: \n" + example_str




def load_few_shot_list(dataset, method):
    ''' a simple function to load the processed numpy few_shot_list
    input
        dataset, string name of the evaluated dataset
        method, string name of the used method    
    output
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    '''
    if dataset == "promise":
        if method == "random":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/promise_random.npy')  
        elif method == "embedding":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/promise_embedding.npy')  
        elif method == "tfidf":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/promise_tfidf.npy')  
        else:
            raise ValueError("method could only be 'random', 'embedding', or 'tfidf' ")    
    elif dataset == "pure":
        if method == "random":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/pure_random.npy')  
        elif method == "embedding":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/pure_embedding.npy')  
        elif method == "tfidf":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/pure_tfidf.npy')  
        else:
            raise ValueError("method could only be 'random', 'embedding', or 'tfidf' ")  
    elif dataset == "nfr":
        if method == "random":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/nfr_random.npy')  
        elif method == "embedding":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/nfr_embedding.npy')  
        elif method == "tfidf":
            few_shot_list = array = np.load('/Users/yongjiantang/Desktop/tang/code/re_genai/data/few_shot_list/nfr_tfidf.npy')  
        else:
            raise ValueError("method could only be 'random', 'embedding', or 'tfidf' ")  
    else:        
        raise ValueError("dataset could only be 'promise' or 'nfr' or 'pure' ")    

    return few_shot_list


def save_prompt_list(path, prompt_list):
    """save the constructed prompts with few-shot examples in a list 
    input
        path, string path to save the list
        prompt_list, list of constructed prompts from the first requirement/sentence in test dataset to the last one
    """
    with open(path, 'w', newline='\n') as file:  
        for i, prompt in enumerate(prompt_list):
            if i+1 == len(prompt_list):
                file.write(prompt)
            else:     
                file.write(prompt + "\n\n\n")



def read_prompt_list(path):
    """read the saved list of prompts 
    input
        path, string path to save the list
    """
    with open(path, 'r') as file:  
        content = file.read()  
    prompt_list_read = content.split('\n\n\n')   

    return prompt_list_read


# def read_prompt_list(path):
#     """read the saved list of prompts 
#     input
#         path, string path to save the list
#     """
#     with open(path, 'r') as file:  
#         prompt_list_read = []
#         prompt = ""                            
#         num_line_n = 0
#         for line in file:
#             # read the prompt one by one based on the double empty line /n/n/n
#             if line == "\n":
#                 num_line_n = num_line_n + 1
#                 if num_line_n == 2:
#                     prompt_list_read.append(prompt)
#                     prompt = ""
#                     num_line_n = 0
#                 else:
#                     prompt = prompt + "\n" + line  

#     return prompt_list_read
