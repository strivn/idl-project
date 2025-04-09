from datasets import load_dataset, Dataset
from .utils import CACHE_DIR
from nltk.tokenize import sent_tokenize

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


def load_flan_dataset(source="Open-Orca/FLAN", split='train', streaming=True):
    dataset = load_dataset(source, split=split,
                           streaming=streaming, cache_dir=CACHE_DIR)
    
    print("Dataset loaded successfully")

    return dataset



def prepare_cnn_dataset(dataset, min_article_sent_len=1):
    """
    Prepares sentence-level query-answer pairs for batched scoring from the CNN-DailyMail Dataset

    Returns:
        all_pairs: list of (highlight, article sentence) tuples
        meta: list of dicts with example id, sentence, highlight, and article_idx
    """
    all_pairs = []
    meta = []

    # changed to use itertuples as it is much faster
    
    # itertuples 
    #   CPU times: user 2.4 ms, sys: 81 μs, total: 2.48 ms
    #   10k CPU times: user 224 ms, sys: 13.3 ms, total: 237 ms

    # iterrows 
    #   CPU times: user 20.8 ms, sys: 3.13 ms, total: 23.9 ms
    #   10k CPU times: user 1.19 s, sys: 54.5 ms, total: 1.25 s
    for row in dataset.itertuples(index=True):
        
        article_sents = sent_tokenize(row.article)
        highlights = sent_tokenize(row.highlights)
        
        if not highlights:
            continue
        
        highlight = highlights[0]
        for sent in article_sents:
            if len(sent.split()) < min_article_sent_len:
                continue
            all_pairs.append((highlight, sent))
            meta.append({
                'id': row.id,
                'highlight': highlight,
                'article_sentence': sent,
                'article_idx': row.Index
            })
    
    return all_pairs, meta