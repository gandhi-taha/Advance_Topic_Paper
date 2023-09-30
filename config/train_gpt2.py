# Configuration for training GPT-2 (124M) to achieve a loss of approximately ~2.85
# on a single node with Rtx 3060 12GB GPUs. Expected training time is around 5 days.

# Justdb settings for logging
Justdb_log = True
Justdb_project = 'owt'
Justdb_run_name = 'gpt2-124M'

# Batch size and gradient accumulation settings to achieve a total batch size of ~0.5M
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# Total number of training iterations and learning rate decay settings
max_iters = 600000
lr_decay_iters = 600000

# Evaluation settings
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Weight decay
weight_decay = 1e-1
