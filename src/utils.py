import os
import torch as t
import numpy as np
import torch.nn.functional as F


# ----------------------
# Utils
# ----------------------

def ensure_directory_exists(directory_path):
    """
    Check if a directory exists, and create it if it doesn't.

    Args:
        directory_path: Path to the directory to check/create

    Returns:
        The path to the directory (which now definitely exists)
    """
    if not os.path.exists(directory_path):
        # Directory doesn't exist, so create it
        # makedirs creates all intermediate directories too
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")

    return directory_path


# ----------------------
# Confgs
# ----------------------

# Load models
if t.cuda.is_available():
    DEVICE = "cuda"
elif t.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


if os.path.exists("/ocean"):
    CACHE_DIR = "/ocean/projects/cis250068p/shared/caches"
else:
    CACHE_DIR = ".cache/"

ensure_directory_exists(CACHE_DIR)


# ----------------------
# Scoring Functions
# ----------------------


def calculate_score(query, answer, model, tokenizer, backward=False, query_direction="reverse", task='citation', debug=False):
    """
    Calculate log probability of response given prompt or vice versa based on query_direction.

    Args:
        query_info (dict): Dictionary containing query, answer, and task type
        model: The language model
        tokenizer: The corresponding tokenizer
        query_direction (str): "reverse" for response->query, "normal" for query->response
        debug (bool): If True, print debug information

    Returns:
        dict: Contains token-wise and sequence log probabilities
    """

    # First, prepare the texts
    if query_direction == "normal":
        assert backward == False, "normal query direction does not support backward model - only for forward model"

        conditioning_prompt = 'is a summary of' if task == 'citation' else 'has an answer to'
        # there's no discussion on the conditioning prompt for this, so we assume its:
        # context [query + conditioning prompt] and target [answer]
        # where conditioning prompt -> "is summarized by"
        # e.g. "Harry potter star... gains access to reported... __ is summarized by __" -> ""

        context = query + conditioning_prompt
        target = answer

    elif query_direction == "reverse":
        if not backward:  # Forward model (Fo) - Algorithm 3
            conditioning_prompt = 'is a summary of' if task == 'citation' else 'has an answer to'
            context = answer + conditioning_prompt
            target = query
        else:  # Backward model (Ba) - Algorithm 2
            conditioning_prompt = 'is summarized by' if task == 'citation' else 'is answered by'
            context = conditioning_prompt + answer
            target = query

    # DEBUG
    if debug:
        print(f"Settings: {backward} {query_direction}")
        print(f"Context: {context}")
        print(f"Target: {target}")
        print(
            f"Full sentence: {context + target if not backward else target + context}")
        print("\n")

    # tokenize
    context_ids = tokenizer.encode(context, return_tensors="pt")
    target_ids = tokenizer.encode(target, return_tensors="pt")

    # Store length to "divide" the texts later
    context_len = context_ids.shape[1]
    target_len = target_ids.shape[1]

    # reverse the token for backward
    if backward:
        context_ids = t.flip(context_ids, (1,))
        target_ids = t.flip(target_ids, (1,))

    # Combine the context and target tokens
    input_ids = t.cat((context_ids, target_ids), dim=1).to(model.device)

    # regardless of the query direction and backward/forward model, the scoring is the same:
    return compute_token_probabilities(input_ids, model, tokenizer, context_len, target_len)


def calculate_score_batch(queries, answers, model, tokenizer, backward=False, query_direction="reverse", task='citation', debug=False):
    """
    Calculate log probability of responses given prompts or vice versa for a batch of pairs.

    Args:
        queries (list): List of query strings
        answers (list): List of answer strings
        model: The language model
        tokenizer: The corresponding tokenizer
        backward (bool): If True, use backward model, otherwise forward model
        query_direction (str): "reverse" for response->query, "normal" for query->response
        task (str): The task type, either 'citation' or something else
        debug (bool): If True, print debug information

    Returns:
        list: List of dicts containing token-wise and sequence log probabilities for each pair
    """
    batch_size = len(queries)
    assert batch_size == len(
        answers), "Number of queries and answers must match"

    # Prepare the texts for each query-answer pair
    context_ids_list = []
    target_ids_list = []
    context_lengths = []
    target_lengths = []

    for query, answer in zip(queries, answers):
        # Prepare the texts based on query direction and model type
        if query_direction == "normal":
            assert backward == False, "normal query direction does not support backward model - only for forward model"

            conditioning_prompt = 'is a summary of' if task == 'citation' else 'has an answer to'
            context = query + conditioning_prompt
            target = answer

        elif query_direction == "reverse":
            if not backward:  # Forward model (Fo) - Algorithm 3
                conditioning_prompt = 'is a summary of' if task == 'citation' else 'has an answer to'
                context = answer + conditioning_prompt
                target = query
            else:  # Backward model (Ba) - Algorithm 2
                conditioning_prompt = 'is summarized by' if task == 'citation' else 'is answered by'
                context = conditioning_prompt + answer
                target = query

        # DEBUG
        if debug:
            print(f"Settings: {backward} {query_direction}")
            print(f"Context: {context}")
            print(f"Target: {target}")
            print(
                f"Full sentence: {context + target if not backward else target + context}")
            print("\n")

        # Tokenize
        context_ids = tokenizer.encode(context, return_tensors="pt")
        target_ids = tokenizer.encode(target, return_tensors="pt")

        # Store length to "divide" the texts later
        context_len = context_ids.shape[1]
        target_len = target_ids.shape[1]

        # Reverse the token for backward
        if backward:
            context_ids = t.flip(context_ids, (1,))
            target_ids = t.flip(target_ids, (1,))

        context_ids_list.append(context_ids.squeeze(0))
        target_ids_list.append(target_ids.squeeze(0))
        context_lengths.append(context_len)
        target_lengths.append(target_len)

    # Pad sequences to the same length
    max_context_len = max(context_lengths)
    max_target_len = max(target_lengths)
    max_len = max_context_len + max_target_len

    input_ids_batch = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer does not have a pad_token_id defined.")

    for i in range(batch_size):
        combined = t.cat((context_ids_list[i], target_ids_list[i]))

        if combined.shape[0] < max_len:
            padding = t.full((max_len - combined.shape[0],), fill_value=pad_id, dtype=t.long)
            combined = t.cat((combined, padding))

        input_ids_batch.append(combined)

    # Stack into a batch tensor
    input_ids_batch = t.stack(input_ids_batch).to(model.device)

    # Compute token probabilities for the batch
    return compute_token_probabilities_batch(input_ids_batch, model, tokenizer, context_lengths, target_lengths)


def compute_token_probabilities(input_ids, model, tokenizer, context_len, target_len):
    """
    Compute token-wise and sequence probabilities from model outputs.

    Args:
        input_ids: Combined context and target token IDs
        model: The language model
        tokenizer: The corresponding tokenizer
        context_len: Length of the context tokens
        target_len: Length of the target tokens

    Returns:
        dict: Contains token-wise and sequence log probabilities
    """
    # Get model output
    with t.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Extract token probabilities for the target text
    token_probs = []
    # Process tokens in the target portion of the combined input
    for i in range(context_len - 1, context_len + target_len - 1):
        # Get the logits for the current position
        next_token_logits = logits[0, i, :]  # no batch, sequence i, all vocab

        # Get the actual token that should follow
        next_token_id = input_ids[0, i+1].item()

        # Convert logits to probabilities
        next_token_probs = F.softmax(next_token_logits, dim=0)
        prob = next_token_probs[next_token_id].item()
        log_prob = np.log(prob)

        token_text = tokenizer.decode([next_token_id])
        token_probs.append({
            'token': token_text,
            'token_id': next_token_id,
            'log_prob': log_prob
        })

    # Calculate sequence probability
    sequence_log_prob = sum(tp['log_prob'] for tp in token_probs)
    # Normalize by length to get per-token average
    normalized_log_prob = sequence_log_prob / len(token_probs)
    # Convert to perplexity if needed
    perplexity = np.exp(-sequence_log_prob / len(token_probs))

    return {
        'token_log_probs': token_probs,
        'sequence_log_prob': sequence_log_prob,
        'normalized_log_prob': normalized_log_prob,
        'perplexity': perplexity
    }


def compute_token_probabilities_batch(input_ids_batch, model, tokenizer, context_lens, target_lens):
    """
    Compute token-wise and sequence probabilities for a batch of inputs.

    Args:
        input_ids_batch: Batch of combined context and target token IDs [batch_size, seq_len]
        model: The language model
        tokenizer: The corresponding tokenizer
        context_lens: List of context token lengths for each sequence in the batch
        target_lens: List of target token lengths for each sequence in the batch

    Returns:
        list: List of dicts containing token-wise and sequence log probabilities for each input
    """
    batch_size = input_ids_batch.shape[0]

    # Get model output for the entire batch in a single forward pass
    with t.no_grad():
        outputs = model(input_ids_batch)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    results = []

    # Process each sequence in the batch
    for batch_idx in range(batch_size):
        context_len = context_lens[batch_idx]
        target_len = target_lens[batch_idx]
        input_ids = input_ids_batch[batch_idx]

        # Extract token probabilities for the target text
        token_probs = []

        # Process tokens in the target portion of the combined input
        for i in range(context_len - 1, context_len + target_len - 1):
            # Get the logits for the current position
            # batch_idx, sequence i, all vocab
            next_token_logits = logits[batch_idx, i, :]

            # Get the actual token that should follow
            next_token_id = input_ids[i+1].item()

            # Convert logits to probabilities
            next_token_probs = F.softmax(next_token_logits, dim=0)
            prob = next_token_probs[next_token_id].item()
            log_prob = np.log(prob)

            token_text = tokenizer.decode([next_token_id])
            token_probs.append({
                'token': token_text,
                'token_id': next_token_id,
                'log_prob': log_prob
            })

        # Skip if no tokens to evaluate (empty target)
        if not token_probs:
            results.append({
                'token_log_probs': [],
                'sequence_log_prob': 0.0,
                'normalized_log_prob': 0.0,
                'perplexity': float('inf')
            })
            continue

        # Calculate sequence probability
        sequence_log_prob = sum(tp['log_prob'] for tp in token_probs)
        # Normalize by length to get per-token average
        normalized_log_prob = sequence_log_prob / len(token_probs)
        # Convert to perplexity if needed
        perplexity = np.exp(-sequence_log_prob / len(token_probs))

        results.append({
            'token_log_probs': token_probs,
            'sequence_log_prob': sequence_log_prob,
            'normalized_log_prob': normalized_log_prob,
            'perplexity': perplexity
        })

    return results





def example_texts():
    # Example Texts
    articles = [
        # 0 - Harry Potter
        "Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him.",
        
        # 1 - Apple
        "In a widely anticipated event on Tuesday, Apple unveiled its newest iPhone model. The device features a significantly improved camera system with low-light enhancements, a faster A17 chip promising better battery life and performance, and subtle design changes. Analysts expect strong demand heading into the holiday season.",
        
        # 2 - NASA
        "NASA’s Perseverance rover successfully touched down on the surface of Mars after a seven-month journey through space. The rover is equipped with a suite of scientific instruments aimed at detecting signs of ancient microbial life and collecting rock samples that may eventually be returned to Earth.",
        
        # 3 - Stock Market
        "The U.S. stock market closed higher on Wednesday, buoyed by a rally in technology stocks and signs that inflation may be cooling. The Federal Reserve hinted at a potential pause in interest rate hikes, which contributed to investor optimism. Major indices, including the Nasdaq and S&P 500, posted strong gains.",
        
        # 4 - Olympics
        "Simone Biles returned to competition after a two-year break and delivered a stunning performance at the national championships, winning gold in the all-around category. Her comeback is being hailed as a major moment in gymnastics history.",
        
        # 5 - COVID Booster
        "The CDC has recommended a new round of COVID-19 booster shots for people over 60 and those with compromised immune systems, citing the recent emergence of new subvariants. Pharmacies across the country are expected to begin offering doses next week.",
        
        # 6 - AI Regulation
        "The European Parliament passed a preliminary AI regulation bill requiring transparency for generative models and stricter risk assessment for high-impact applications. The law aims to protect users while supporting innovation in artificial intelligence.",
        
        # 7 - Climate Report
        "A new report from the United Nations warns that global temperatures could surpass the 1.5°C threshold within the next decade if greenhouse gas emissions aren’t drastically reduced. The report urges immediate policy action to avoid severe climate disruption.",
        
        # 8 - Movie Release
        "Christopher Nolan's new historical thriller 'Oppenheimer' opened to strong box office numbers this weekend, bringing in over $80 million globally. Critics have praised the film's storytelling and lead performances.",
        
        # 9 - EV Expansion
        "Ford announced a $2 billion investment to expand electric vehicle production in the Midwest, including new battery facilities. The move is part of Ford’s effort to compete with Tesla and meet rising consumer demand for EVs.",
        
        # 10 - Mental Health App
        "A recent study found that a new mobile app designed for managing anxiety showed significant improvement in users' stress levels after just four weeks of use. The app offers CBT-based exercises and journaling features.",
        
        # 11 - Crypto News
        "Bitcoin prices spiked to their highest level in over a year following speculation that a major asset manager may receive approval for a spot ETF. The crypto market saw a general uptick across multiple tokens.",
        
        # 12 - SpaceX Launch
        "SpaceX successfully launched another batch of Starlink satellites into orbit aboard a Falcon 9 rocket, marking the company’s 65th mission this year. The reusable booster landed safely on a drone ship in the Atlantic.",
        
        # 13 - Food Prices
        "Grocery prices have stabilized for the third month in a row, according to the latest consumer price index data. Analysts say easing supply chain issues and fuel costs are helping to curb food inflation.",
        
        # 14 - Education Policy
        "The Department of Education announced plans to introduce a national tutoring program aimed at helping students catch up on learning gaps caused by the COVID-19 pandemic. Funding will come from federal relief packages.",
        
        # 15 - Streaming Growth
        "Netflix reported a surprising 12% subscriber growth in the last quarter, driven largely by its crackdown on password sharing and introduction of a new ad-supported tier. Revenue also exceeded analyst expectations."
    ]

    summaries = [
        "Harry Potter star Daniel Radcliffe gets £20M fortune as he turns 18 Monday",
        "Apple launches new iPhone with better camera and chip",
        "NASA rover lands on Mars for ancient life mission",
        "Tech stocks boost market as inflation concerns drop",
        "Simone Biles wins all-around gold in comeback",
        "CDC recommends new COVID booster for seniors",
        "EU passes bill to regulate AI transparency and risk",
        "UN warns global temps may exceed 1.5°C soon",
        "'Oppenheimer' earns $80M on strong opening weekend",
        "Ford to invest $2B to boost EV production",
        "Mental health app reduces anxiety in 4 weeks",
        "Bitcoin hits yearly high on ETF speculation",
        "SpaceX launches Starlink satellites, lands booster",
        "Food prices hold steady for third month",
        "New U.S. tutoring program to address pandemic learning gaps",
        "Netflix grows 12% after curbing password sharing"
    ]

    adverse_summaries = [
        "Daniel Craig is recasted as James Bond again",
        "Samsung releases folding phone with improved battery",
        "NASA delays rover mission due to mechanical failure",
        "Market crashes amid tech layoffs and inflation fears",
        "Olympic committee bans Simone Biles from competition",
        "FDA pulls COVID boosters citing lack of demand",
        "AI startup fined for violating European data laws",
        "New study denies impact of CO2 on global warming",
        "Nolan's 'Oppenheimer' flops on opening weekend",
        "Ford cuts EV funding amid low consumer interest",
        "Mental health app criticized for lack of scientific backing",
        "Bitcoin plummets after regulatory investigation",
        "SpaceX mission fails due to rocket malfunction",
        "Food prices soar after global wheat shortage",
        "U.S. cuts education funding following budget crisis",
        "Netflix loses subscribers despite new ad tier"
    ]
    
    return articles, summaries, adverse_summaries