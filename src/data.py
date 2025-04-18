import random
import re
from itertools import chain, islice

from datasets import (Dataset, concatenate_datasets, enable_progress_bar,
                      load_dataset)
from tqdm.auto import tqdm

from .utils import CACHE_DIR

SEED = 11785

enable_progress_bar()
# ----------------------
# CNN
# ----------------------


def load_cnn_dataset(num_samples=10):
    try:
        # Try with a specific cache directory
        dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=CACHE_DIR)
        print("Dataset loaded successfully")

        # Verify the structure - this helps debug
        if num_samples > 0:
            print("Example dataset item:", dataset['train'][0])

        # Take only a small sample for testing
        if hasattr(dataset, 'train'):
            return dataset['train'].select(range(min(num_samples, len(dataset['train']))))

        return dataset['train'][:num_samples]

    except Exception as e:
        print(f"Error loading full dataset: {e}")

        # Create a tiny synthetic dataset for testing
        print("Creating synthetic test dataset instead...")

        sample_data = {
            'article': [
                "John likes to play basketball. He goes to the court every evening. His friends join him on weekends.",
                "The company announced record profits. Investors were pleased. The stock price increased by 10%."
            ],
            'highlights': [
                "John plays basketball regularly with friends.",
                "Company profits lead to stock price increase."
            ],
            'id': ['test1', 'test2']  # Added ID field
        }

        return Dataset.from_dict(sample_data)


# ----------------------
# FLAN
# ----------------------
def load_flan_dataset(source="Open-Orca/FLAN",
                      subset=None,
                      split='train',
                      streaming=True):
    '''
    Wrapper to load FLAN dataset
    '''

    print(f"Downloading data to {CACHE_DIR}")

    if not subset:
        dataset = load_dataset(source,
                               split=split,
                               streaming=streaming,
                               cache_dir=CACHE_DIR)
    else:
        dataset = load_dataset(source,
                               split=split,
                               data_files=f"{subset}/*",
                               streaming=streaming,
                               cache_dir=CACHE_DIR)

    print(f"Dataset loaded successfully: {subset if subset else source}")

    return dataset


# ----------------------
# Summarization
# ----------------------
def load_summarization_datasets(subset_names=None, subset_frac=1, streaming=False, p=0):
    '''
    Load and filter summarization-related tasks from FLAN zsopt datasets
    
    Arguments:
    subset_names: set of FLAN folders that you'd like to download. If set to None, downloads everything. 
    subset_frac: based on the concatenated dataset / folders, further filter a fraction of it
    streaming:
    p: whether to take only summarization tasks (0) or all tasks (1)
    '''

    # List for storing summarization datasets
    summarization_datasets = []

    # Keywords that may indicate summarization tasks
    summarization_keywords = [
        "summarize", "summary", "summarization", "summarize this",
        "tldr", "summarise", "abstract", "synopsis",
        "condense", "key points", "main points",
        "gist", "highlight", "overview", "recap",
    ]

    # CNN/DailyMail exclusion pattern
    cnn_pattern = re.compile(
        r'cnn|daily mail|dailymail|cnn_dailymail', re.IGNORECASE)
    
    # Function to filter for summarization tasks
    def is_valid_task(example):
        if random.random() < p:
            # allow other tasks to pass through by p probability
            return True

        # Check if it's a summarization task
        if not isinstance(example.get("inputs", ""), str):
            return False

        is_summarization = any(keyword in example.get(
            "_task_name", "") for keyword in summarization_keywords)

        # If not a summarization task, return False
        if not is_summarization:
            return False

        # filter out CNN/DailyMail examples
        # Check in inputs, task_name or any other relevant field
        input_text = example.get("inputs", "")
        task_name = example.get("_task_name", "")

        # Return False if CNN/DailyMail is found in inputs or task_name
        if (cnn_pattern.search(input_text) is not None or
                cnn_pattern.search(task_name) is not None):
            return False

        return True
    
    # Function to process and filter a dataset
    def process_dataset(dataset, cache_name):
        print(f"Original {cache_name} size: {len(dataset)}")
        subset_size = int(len(dataset) * subset_frac)
        dataset = dataset.select(range(subset_size))
        
        intermediate_path = f"{CACHE_DIR}/{cache_name}_p_{p}.arrow"
        
        filtered_dataset = dataset.filter(
            is_valid_task,
            num_proc=20,
            batch_size=1600,
            cache_file_name=intermediate_path,
        )
        
        return filtered_dataset
    
    # Load and process datasets based on subset_names
    if subset_names:
        for subset in tqdm(subset_names):
            try:
                print(f"Processing {subset}...")
                # Load dataset with specific subset
                dataset = load_flan_dataset(
                    subset=subset,
                    streaming=streaming
                )
                
                summarization_subset = process_dataset(dataset, f"subset_{subset}")
                summarization_datasets.append(summarization_subset)
                print(f"Found summarization examples in {subset}")
                
            except Exception as e:
                print(f"Error processing {subset}: {e}")
    else:
        try:
            print("No subset names provided, downloading all datasets...")
            # Load dataset without specifying subset
            dataset = load_flan_dataset(
                streaming=streaming  # No subset parameter, downloads everything
            )
            
            summarization_subset = process_dataset(dataset, "all_datasets")
            summarization_datasets.append(summarization_subset)
            print(f"Found summarization examples in all datasets")
            
        except Exception as e:
            print(f"Error processing all datasets: {e}")

    # Concatenate all datasets
    if summarization_datasets:
        final_dataset = concatenate_datasets(summarization_datasets)
        print(f"Combined dataset with {len(final_dataset)} summarization examples")
        return final_dataset.shuffle(SEED)
    else:
        print("No summarization examples found")
        return None


# def load_summarization_datasets(subset_names=None, subset_frac=1, streaming=False, p=0):
#     '''
#     Load and filter summarization-related tasks from FLAN zsopt datasets
    
#     Arguments:
#     subset_names: set of FLAN folders that you'd like to download. If set to None, downloads everything. 
#     subset_frac: based on the concatenated dataset / folders, further filter a fraction of it
#     streaming:
#     p: whether to take only summarization tasks (0) or all tasks (1)
#     '''

#     # List for storing summarization datasets
#     summarization_datasets = []

#     # Prioritize smaller subsets first
#     if not subset_names:
#         subset_names = [
#             "niv2_zsopt_data",    # 2.4GB
#             "cot_zsopt_data",     # 20MB
#             "dialog_zsopt_data",  # 984MB
#             "flan_zsopt_data",    # 11GB
#             "t0_zsopt_data"       # 17GB
#         ]

#     # Keywords that may indicate summarization tasks
#     summarization_keywords = [
#         "summarize", "summary", "summarization", "summarize this",
#         "tldr", "summarise", "abstract", "synopsis",
#         "condense", "key points", "main points",
#         "gist", "highlight", "overview", "recap",
#     ]

#     # Create regex pattern for keyword matching
#     pattern = re.compile(
#         r'\b(' + '|'.join(summarization_keywords) + r')\b', re.IGNORECASE)

#     # CNN/DailyMail exclusion pattern
#     cnn_pattern = re.compile(
#         r'cnn|daily mail|dailymail|cnn_dailymail', re.IGNORECASE)

#     for subset in tqdm(subset_names):
#         print(f"Processing {subset}...")

#         try:
#             # Load dataset - assuming load_flan_dataset is a custom function
#             # Replace with standard HuggingFace datasets loading
#             dataset = load_flan_dataset(
#                 subset=subset,
#                 streaming=streaming  # Always use cache/download
#             )

#             print(f"Original {subset} size: {len(dataset)}")
#             subset_size = int(len(dataset) * subset_frac)
#             dataset = dataset.select(range(subset_size))

#             # Filter for summarization tasks using standard .filter() method
#             def is_valid_task(example):

#                 if random.random() < p:
#                     # allow other tasks to pass through by p probability
#                     return True

#                 # Check if it's a summarization task
#                 if not isinstance(example.get("inputs", ""), str):
#                     return False

#                 is_summarization = any(keyword in example.get(
#                     "_task_name", "") for keyword in summarization_keywords)

#                 # If not a summarization task, return False
#                 if not is_summarization:
#                     return False

#                 # filter out CNN/DailyMail examples
#                 # Check in inputs, task_name or any other relevant field
#                 input_text = example.get("inputs", "")
#                 task_name = example.get("_task_name", "")

#                 # Return False if CNN/DailyMail is found in inputs or task_name
#                 if (cnn_pattern.search(input_text) is not None or
#                         cnn_pattern.search(task_name) is not None):
#                     return False

#                 return True

#             intermediate_path = f"{CACHE_DIR}/subset_{subset}_p_{p}.arrow"

#             summarization_subset = dataset.filter(
#                 is_valid_task,
#                 num_proc=20,
#                 batch_size=1600,
#                 # keep_in_memory=True,
#                 cache_file_name=intermediate_path,
#             )

#             # summarization_subset = dataset.filter(
#             #     lambda example: isinstance(example.get("inputs", ""), str) and
#             #     pattern.search(example.get("inputs", ""))
#             # )

#             summarization_datasets.append(summarization_subset)
#             print(f"Found summarization examples in {subset}")

#         except Exception as e:
#             print(f"Error processing {subset}: {e}")

#     # Concatenate all datasets
#     if summarization_datasets:
#         final_dataset = concatenate_datasets(summarization_datasets)
#         print(
#             f"Combined dataset with {len(final_dataset)} summarization examples")
#         return final_dataset.shuffle(SEED)
#     else:
#         print("No summarization examples found")
#         return None
