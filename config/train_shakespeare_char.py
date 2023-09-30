# Configuration for training a miniature character-level Shakespeare model
# Suitable for debugging and running on Rtx 3060 or above with similar setups 

# Output directory for checkpoints
out_dir = 'out-shakespeare-char'

# Evaluation settings
eval_interval = 250  # Frequent evaluation due to expected overfitting
eval_iters = 200

# Logging interval
log_interval = 10

# Save checkpoints only if validation improves
always_save_checkpoint = False

# Justdb (Weights and Biases) settings
Justdb_log = False  # Override via command line if needed
Justdb_project = 'shakespeare-char'
Justdb_run_name = 'small-gpt'

# Dataset and batch settings
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # Context of up to 256 previous characters

# Model architecture (small GPT)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Learning rate and optimization settings
learning_rate = 1e-3  # Slightly higher learning rate for smaller networks
max_iters = 5000
lr_decay_iters = 5000  # Typically set equal to max_iters
min_lr = 1e-4  # Learning rate / 10 usually
beta2 = 0.99  # Slightly larger beta2 because of a small number of tokens per iteration

# Warmup iterations (not super necessary but included)
warmup_iters = 100
