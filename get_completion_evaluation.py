
### functions that are used to support few-shot prompting

### save and read the generated completions from LLMs
### functions to process the completions, filter unnecessary
### functions to evaluate the processed completions 

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.metrics import precision_score, recall_score, f1_score



def save_completion_list(path, completion_list):
    """save the constructed prompts with few-shot examples in a list 
    input
        path, string path to save the list
        prompt_list, list of constructed prompts from the first requirement/sentence in test dataset to the last one
    """
    with open(path, 'w', newline='\n') as file:  
        for i, completion in enumerate(completion_list):
            if i+1 == len(completion_list):
                file.write(completion)
            else:     
                file.write(completion + "\n\n\n")



def read_completion_list(path):
    """read the saved list of prompts 
    input
        path, string path to save the list
    """
    with open(path, 'r') as file:  
        content = file.read()  
    completion_list_read = content.split('\n\n\n')   

    return completion_list_read



def get_evaluation_results(ground_truth, processed_completion_list):
    """input
        ground_truth: list, true requirement classifications
        processed_completion_list: list, completions from LLMs
    """
    # Calculate precision, recall, and F1 score  
    precision = precision_score(ground_truth, processed_completion_list, zero_division = True, average='weighted')  
    recall = recall_score(ground_truth, processed_completion_list, zero_division = 0, average='weighted')  
    f1 = f1_score(ground_truth, processed_completion_list, average='weighted')  
    
    # Print the results  
    print("F1 Score:", f1 )  
    print("Precision:", precision)  
    print("Recall:", recall, "\n###########\n\n")  

    return f1, precision, recall


def get_evaluation_results_macro(ground_truth, processed_completion_list):
    """input
        ground_truth: list, true requirement classifications
        processed_completion_list: list, completions from LLMs
    """
    # Calculate precision, recall, and F1 score  
    precision = precision_score(ground_truth, processed_completion_list, zero_division = True, average='macro')  
    recall = recall_score(ground_truth, processed_completion_list, zero_division = 0, average='macro')  
    f1 = f1_score(ground_truth, processed_completion_list, average='macro')  
    
    # Print the results  
    print("F1 Score:", f1 )  
    print("Precision:", precision)  
    print("Recall:", recall, "\n###########\n\n")  

    return f1, precision, recall





def separate_class_evaluation(ground_truth, predictions):
    """evaluate separate class respectively """

    # Unique classes  
    classes = set(ground_truth)  
      
    # Initialize dictionaries to store metrics  
    precision_dict = {}  
    recall_dict = {}  
    f1_score_dict = {}  
      
    # Calculate precision, recall, and F1 score for each class  
    for cls in classes:  
        # Binary representation for current class  
        binary_truth = [1 if x == cls else 0 for x in ground_truth]  
        binary_pred = [1 if x == cls else 0 for x in predictions]  
          
        # Precision, Recall, and F1 score  
        precision = precision_score(binary_truth, binary_pred, zero_division=0)  
        recall = recall_score(binary_truth, binary_pred, zero_division=0)  
        f1 = f1_score(binary_truth, binary_pred, zero_division=0)  
          
        # Store in dictionaries  
        precision_dict[cls] = np.round(precision*100,2)
        recall_dict[cls] = np.round(recall*100,2)
        f1_score_dict[cls] = np.round(f1*100,2)  

    return precision_dict, recall_dict, f1_score_dict



def process_completion_list(completion_list):

    return completion_list





def precision_recall_f1(ground_truth, predictions):  
    ''' might be not useful anymore give the function above separate_class_evaluation()
    given two list of binary values, e.g. functional, non-functional,
    calculate their precision, recall, and F1 score respectively.
    '''

    assert len(ground_truth) == len(predictions), "The length of ground truth and predictions must be the same."  
  
    def calculate_metrics(true_positive, false_positive, false_negative):  
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0  
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0  
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  
        return precision, recall, f1_score  
  
    # Functional (1)  
    TP_functional = sum((gt == 1 and pred == 1) for gt, pred in zip(ground_truth, predictions))  
    FP_functional = sum((gt == 0 and pred == 1) for gt, pred in zip(ground_truth, predictions))  
    FN_functional = sum((gt == 1 and pred == 0) for gt, pred in zip(ground_truth, predictions))  
      
    precision_functional, recall_functional, f1_functional = calculate_metrics(TP_functional, FP_functional, FN_functional)  
  
    # Non-functional (0)  
    TP_non_functional = sum((gt == 0 and pred == 0) for gt, pred in zip(ground_truth, predictions))  
    FP_non_functional = sum((gt == 1 and pred == 0) for gt, pred in zip(ground_truth, predictions))  
    FN_non_functional = sum((gt == 0 and pred == 1) for gt, pred in zip(ground_truth, predictions))  
      
    precision_non_functional, recall_non_functional, f1_non_functional = calculate_metrics(TP_non_functional, FP_non_functional, FN_non_functional)  
  
    return {  
        "functional": {  
            "precision": precision_functional,  
            "recall": recall_functional,  
            "f1_score": f1_functional  
        },  
        "non_functional": {  
            "precision": precision_non_functional,  
            "recall": recall_non_functional,  
            "f1_score": f1_non_functional  
        }  
    }  
           




def process_ground_truth_list(loaded_test_data, bi_classification):
    """input
        The loaded dataset_test.csv dataframe
       output
        The ground_truth, i.e. list of categories
    """
    if bi_classification:
        ground_truth = loaded_test_data.iloc[:, 2].tolist()  
    elif bi_classification == False:
        ground_truth = loaded_test_data.iloc[:, 1].tolist()  
    else:
        raise ValueError ("bi_classification could only be True or False")

    return ground_truth



def load_csv_for_evaluation(file_path):

    # loaded_test_data = []
    # with open(file_path, 'r') as file:  
    #     csv_reader = csv.reader(file)  
    #     for row in csv_reader:  
    #         loaded_test_data.append(row)  

    loaded_test_data = pd.read_csv(file_path)  

    return loaded_test_data  # the first row of title is not loaded, only the content is necessary.




def plot_value(values):  
  
    # Create a figure and axis  
    fig, ax = plt.subplots()  
  
    num_shot_list = [0,5,10,20,40,80,120,160]
    # Plot the change of values  
    ax.plot(num_shot_list, values)  

    # set y_axis limit
    ax.set_ylim([0, 1.0])
  
    # Set labels and title  
    ax.set_xlabel('number of few shots')  
    ax.set_ylabel('F1 score')  
    ax.set_title('Relationship between F1 score and number of few-shot examples')  
  
    # Show the plot  
    plt.show() 



def plot_line_graphs_with_values(x_values, *lists, labels=None, title='Line Graph', xlabel='X-axis', ylabel='Y-axis'):
    ## not used in paper figures

    """
    Plots multiple line graphs from provided lists and annotates each point with its value.

    Parameters:
    x_values: list of numbers representing the x-axis values for all lists.
    *lists: arbitrary number of lists, each representing a line to plot.
    labels: list of strings, labels corresponding to the lists.
    title: title of the plot.
    xlabel: label for the x-axis.
    ylabel: label for the y-axis.
    """

    # Verify that all data lists have the same length as x_values
    for data in lists:
        if len(data) != len(x_values):
            raise ValueError("All input lists must have the same length as the x_values list. length of input list: {}. length of x_value list: {}.".format( len(data), len(x_values) ))
    
    # Check if labels are provided and match the number of lists
    if labels and len(labels) != len(lists):
        raise ValueError("Number of labels must match the number of lists provided. Number of labels: {}. Number of lists: {}.".format(len(labels), len(lists)))
    
    # Set a style for the plot
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Plot each list
    for i, data in enumerate(lists):
        label = labels[i] if labels else f'Line {i+1}'
        ax.plot(x_values, data, label=label, linewidth=2)

        # Annotate each point with its value
        for x, y in zip(x_values, data):
            ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5), textcoords='offset points', fontsize=9, color='black', ha='center')

    # Add grid, legend, and labels
    ax.grid(True, which='both', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Optimize layout
    fig.tight_layout()

    # Display the plot
    plt.show()


# Example usage:
# x_values = [0, 5, 10, 20, 40, 80, 120, 160]
# plot_line_graphs_with_values(x_values, [1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1], labels=['List 1', 'List 2'], title='Example Line Plot', xlabel='Index', ylabel='Value')




def plot_line_graphs_with_values_mul(save_path, x_values, *lists, labels=None, title='Line Graph', xlabel='X-axis', ylabel='Y-axis'):
    """
    Plots multiple line graphs from provided lists and annotates each point with its value.

    Parameters:
    x_values: list of numbers representing the x-axis values for all lists.
    *lists: arbitrary number of lists, each representing a line to plot.
    labels: list of strings, labels corresponding to the lists.
    title: title of the plot.
    xlabel: label for the x-axis.
    ylabel: label for the y-axis.
    """
    # def custom_transform(y):  
    #     if y <= 60:  
    #         return y * 0.5
    #     elif y <= 80:  
    #         return 30 + (y - 60)
    #     else:  
    #         return 50 + (y - 80) * 2
  
    # def inverse_transform(y):  
    #     if y <= 30:  
    #         return y * 2  
    #     elif y <= 50:  
    #         return 60 + (y - 30)  
    #     else:  
    #         return 80 + (y - 50) / 2  
        
    # Verify that all data lists have the same length as x_values
    for data in lists:
        if len(data) != len(x_values):
            raise ValueError("All input lists must have the same length as the x_values list. length of input list: {}. length of x_value list: {}.".format( len(data), len(x_values) ))
    
    # Check if labels are provided and match the number of lists
    if labels and len(labels) != len(lists):
        raise ValueError("Number of labels must match the number of lists provided. Number of labels: {}. Number of lists: {}.".format(len(labels), len(lists)))
    
    # Set a style for the plot
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create a figure and an axes
    fig, ax = plt.subplots( figsize=(10, 4.5) )

    # Plot each list
    for i, data in enumerate(lists):
        label = labels[i] if labels else f'Line {i+1}'
        ax.plot(x_values, data, '--', label=label, linewidth=1.2)

        # Annotate each point with its value
        for x, y in zip(x_values, data):
            if y == max(data):
                ax.annotate(f'{y}', xy=(x, y), xytext=(0, 5), textcoords='offset points', fontsize=9, color='black', ha='center')
                ax.text(x, y, 'x', fontsize=12, color='red', ha='center', va='center')  

    # Add grid, legend, and labels
    ax.grid(True, which='both', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)


    # Optimize layout
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight') 

    # Display the plot
    plt.show()




def transform_f1_format(f1_list_random, f1_list_embedding, f1_list_tfidf):
    '''
    for drawing bar graphs, transform the format from one method with different number of shots to three methods with same number of shots
    input: saved lists for a specific dataset using three few-shot selection methods
    return list_of_list, the combined list of zero_shot_weighted, five_shots_weighted, ten_shots_weighted, ...

    '''
    list_of_list = []
    
    for i in range(len(f1_list_random)):
        temp_list = []
        temp_list.append(round(f1_list_random[i],1))
        temp_list.append(round(f1_list_embedding[i],1))
        temp_list.append(round(f1_list_tfidf[i],1))
        list_of_list.append(temp_list)
        
    return list_of_list[0], list_of_list[1], list_of_list[2], list_of_list[3], list_of_list[4], list_of_list[5], list_of_list[6], list_of_list[7]


                         
