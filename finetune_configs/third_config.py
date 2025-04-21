## TRAINING CONFIG

FT_TYPE         = 'full'            #full or lora

RUN_NAME        = f'{FT_TYPE}-tuning-config3'


# Data parameters
SUBSET_FRAC                 = 1
# FLAN_SUBSET                 = ['niv2_zsopt_data', 'flan_zsopt_data', 't0_zsopt_data']
# FLAN_SUBSET                 = ['niv2_zsopt_data', 'cot_zsopt_data', 'dialog_zsopt_data']

# FLAN_SUBSET                 = ['niv2_zsopt_data', 'flan_zsopt_data', 'cot_zsopt_data']
FLAN_SUBSET                 = ['niv2_zsopt_data', 'flan_zsopt_data', 't0_zsopt_data', 'cot_zsopt_data', 'dialog_zsopt_data']

TEST_SIZE                   = 0.01
TASK_DIVERSITY_P            = 0.2

# Training parameters
NUM_EPOCHS                  = 1
BATCH_SIZE                  = 32
GRADIENT_ACCUMULATION_STEPS = 4

WEIGHT_DECAY                = 0.01
LEARNING_RATE               = 5e-7
LR_SCHEDULER_TYPE           = "cosine"
WARMUP_RATIO                = 0.1
SAVE_STEPS                  = 5000
EVAL_STEPS                  = 5000
LOGGING_STEPS               = 50

MAX_SEQ_LENGTH              = 1024
SAVE_TOTAL_LIMIT            = 3
FP16                        = True
# BUFFER_SIZE                 = 25000  # For dataset processing

# LoRA configuration
LORA_RANK                   = 8
LORA_ALPHA                  = 32
LORA_DROPOUT                = 0.05
TARGET_MODULES              = ["query_key_value"]  # Target specific attention modules