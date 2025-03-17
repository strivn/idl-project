# Can backward models improve citation?


## Overview

[Work is performed as part of course project for 11785-S25 Intro to Deep Learning]

Language models have demonstrated remarkable capabilities in text generation across various domains. However, their probabilistic nature often leads to the generation of content that are coherent but may lack factual accuracy. This phenomenon is commonly referred as "hallucination". Therefore this raises the motivation to explore, how may we improve language models factual accuracy capabilities?

Recent work by (Varun et al., 2025) introduced [Time-Reversed Language Models (TRLMs)](papers/TRLM_2412.02626.pdf). It is shown in the paper that TRLMs may achieve better performance in citations. So this project attempts to examine whether reverse language models can improve citation accuracy.


## Getting Started
```python
# Clone the repository
git clone https://github.com/strivn/idl-project
cd idl-project

# Install requirements
pip install -r requirements.txt
```

## Repository Structure
The repository is yet to be refactored at this stage.
```
idl-project
|- scoring_citation_experiment_v3.ipynb : Citation attribution notebook
|- pythia_fine_tune.ipynb               : Fine tuning notebook
|- papers/                              : references
|- .archive/                            : old files
```


## Fine-Tuning Data

At this stage of the project, fine tuning models on the full FLAN 2022 dataset as described on the TRLM paper would be costly and time consuming. So we use [FLAN Subset Mini](https://huggingface.co/datasets/pszemraj/flan-subsets-mini) (around 300 mb based on deduped) to test the fine tuning process.



## Team
- Ivan Wiryadi (iwiryadi)
- Janbol Jangabyl (jjangby)
- Jihu Hwang (jihuh)
- Xuanyi Shen
