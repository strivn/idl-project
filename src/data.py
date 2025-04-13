import re
from itertools import chain, islice

from datasets import Dataset, concatenate_datasets, load_dataset, enable_progress_bar
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
def load_summarization_datasets(subset_names=None, subset_frac=1, streaming=False):
    '''
    Load and filter summarization-related tasks from FLAN zsopt datasets
    '''

    # List for storing summarization datasets
    summarization_datasets = []

    # Prioritize smaller subsets first
    if not subset_names: 
        subset_names = [
            "niv2_zsopt_data",    # 2.4GB
            "cot_zsopt_data",     # 20MB
            "dialog_zsopt_data",  # 984MB
            "flan_zsopt_data",    # 11GB
            "t0_zsopt_data"       # 17GB
        ]

    # Keywords that may indicate summarization tasks
    summarization_keywords = [
        "summarize", "summary", "summarization", "summarize this",
        "tldr", "summarise", "abstract", "synopsis",
        "condense", "extract", "key points", "main points",
        "gist", "highlight", "overview", "recap", "sum"
    ]

    # Create regex pattern for efficient keyword matching
    pattern = re.compile(
        r'\b(' + '|'.join(summarization_keywords) + r')\b', re.IGNORECASE)

    for subset in tqdm(subset_names):
        print(f"Processing {subset}...")

        try:
            # Load dataset - assuming load_flan_dataset is a custom function
            # Replace with standard HuggingFace datasets loading
            dataset = load_flan_dataset(
                subset=subset,
                streaming=streaming  # Always use cache/download
            )
            
            print(type(dataset))

            subset_size = int(len(dataset) * subset_frac)
            dataset = dataset.select(range(subset_size))
            
            print(type(dataset))
            # Filter for summarization tasks using standard .filter() method
            def is_summarization_task(example):
                if not isinstance(example.get("inputs", ""), str):
                    return False
                return pattern.search(example.get("inputs", "")) is not None
            
            summarization_subset = dataset.filter(
                is_summarization_task,
                num_proc=4,
                batch_size=1600,  # Process examples in batches
            )

            # summarization_subset = dataset.filter(
            #     lambda example: isinstance(example.get("inputs", ""), str) and
            #     pattern.search(example.get("inputs", ""))
            # )

            summarization_datasets.append(summarization_subset)
            print(f"Found summarization examples in {subset}")

        except Exception as e:
            print(f"Error processing {subset}: {e}")

    # Concatenate all datasets
    if summarization_datasets:
        final_dataset = concatenate_datasets(summarization_datasets)
        print(
            f"Combined dataset with {len(final_dataset)} summarization examples")
        return final_dataset.shuffle(SEED)
    else:
        print("No summarization examples found")
        return None
