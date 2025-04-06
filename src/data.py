from datasets import load_dataset, Dataset

CACHE_DIR = '.cache'

# Load dataset
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


def load_flan_dataset(source="Open-Orca/FLAN", streaming=True):
    dataset = load_dataset(source, split="train",
                           streaming=streaming, cache_dir=CACHE_DIR)
    
    print("Dataset loaded successfully")

    return dataset
