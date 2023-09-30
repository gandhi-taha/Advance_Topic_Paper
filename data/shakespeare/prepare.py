import os
import requests
import tiktoken
import numpy as np

# Define the URL for the Shakespeare dataset
data_url = 'https://raw.githubusercontent.com/gandhi-taha/Advance_Topic_Paper/main/min_shakespeare.txt'

# Download the dataset if it doesn't exist
input_file_path = 'input.txt'
if not os.path.exists(input_file_path):
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

# Read the dataset from the file
with open(input_file_path, 'r') as f:
    data = f.read()

n = len(data)

# Create train and validation splits
train_data = data[:int(n * 0.99)]
val_data = data[int(n * 0.99):]

# Encode the text using GPT-2 BPE with tiktoken
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Validation has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')
