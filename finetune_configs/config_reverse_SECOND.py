PROJECT_NAME                = "janbol-reverse"
RUN_NAME                    =f"{PROJECT_NAME} - second config reverse"
CACHE_DIR                   = "/ocean/projects/cis250068p/jangabyl/caches"
HUGGING_FACE                = "ENTER THE TOKEN HERE"
MAX_LENGTH                  = 1024
PREPROCESS_BATCH_SIZE       = 1000

# Data parameters
SUBSET_FRAC                 = 1
FLAN_SUBSET                 = ['niv2_zsopt_data', 'flan_zsopt_data', 't0_zsopt_data', 'cot_zsopt_data', 'dialog_zsopt_data']

TEST_SIZE                   = 0.01
TASK_DIVERSITY_P            = 0.1

# Training parameters
NUM_EPOCHS                  = 2
BATCH_SIZE                  = 32
GRADIENT_ACCUMULATION_STEPS = 4
OUTPUT_DIR                  = "/ocean/projects/cis250068p/jangabyl/caches/training/pythia-reverse-finetuned"
LOG_DIR                     = "/ocean/projects/cis250068p/jangabyl/caches/training/pythia-reverse-finetuned/logs"

WEIGHT_DECAY                = 0.01
LEARNING_RATE               = 6e-4
LR_SCHEDULER_TYPE           = "cosine"
WARMUP_RATIO                = 0.1
SAVE_STEPS                  = 5000
EVAL_STEPS                  = 5000
LOGGING_STEPS               = 50

MAX_SEQ_LENGTH              = 1024
SAVE_TOTAL_LIMIT            = 3
FP16                        = True
# BUFFER_SIZE                 = 25000  # For dataset processing

