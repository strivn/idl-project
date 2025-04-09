# PSC Readme

## Setting your environment in PSC

1. Use `-A` to set up the allocate resources to use a specific project, e.g. `interact [usual commands] -A cis250068p`. This is so we don't use the homework allocation.
2. Initially, set up your environment
    ```
    module load AI/pytorch_23.02-1.13.1-py3
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install transformers==4.50.3 datasets tokenizers sentence_transformers nltk rouge_score accelerate
    ```
3. Pull/clone from git if havent https://github.com/strivn/idl-project


## Running
1. `interact [usual commands] -A cis250068p` for interactive notebook
2. `cd idl-project` and `sbatch run_training.sh` for batch job (untested)

## Some notes
1. Check what cuda version is installed through nvcc / nvidia-smi and may need to adjust the torch installation accordingly (whl/cu124 -> cuda 124)
    
## Debugging

**RuntimeError: operator torchvision::nms does not exist
**
Clue: if it sees /.local/lib/python then perhaps there are rogue installations somewhere that is overriding the environment's packages. Simply remove, e.g. 
`rm -rf ~/.local/lib/python3.10/site-packages/torchvision*`