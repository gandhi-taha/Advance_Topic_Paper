import os
import pickle
import requests
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

print(f"Length of dataset in characters: {len(data):,}")

# Get all unique characters in the text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("All the unique characters:", ''.join(chars))
print(f"Vocabulary size: {vocab_size:,}")

# Create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Create train and validation splits
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Encode both splits to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Validation has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

# Save meta information to help with encoding/decoding later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
