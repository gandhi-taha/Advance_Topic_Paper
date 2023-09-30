import time

# Define the output directory for checkpoints
out_dir = 'out-shakespeare'

# Evaluation settings
eval_interval = 5
eval_iters = 40

# Wandb (optional) settings
wandb_log = False  # You can change this to True if desired
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

# Dataset and initialization settings
dataset = 'shakespeare'
init_from = 'gpt2'  # Use the largest GPT-2 model

# Checkpoint saving
always_save_checkpoint = False  # Save checkpoints only if validation loss improves

# Training parameters
# Define batch size, gradient accumulation steps, and maximum iterations
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# Learning rate and LR decay
learning_rate = 3e-5
decay_lr = False
