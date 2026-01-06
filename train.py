import torch
import torch.optim as optim
import tiktoken
import numpy as np
import os
from model import NexaGPT, NexaConfig
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 4
BLOCK_SIZE = 64
MAX_ITERS = 500
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Data
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Creating a dummy file.")
        with open(file_path, 'w') as f:
            f.write("Nexa is a great model. " * 100)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Encode Data
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, data):
    out = {}
    model.eval()
    losses = torch.zeros(100)
    for k in range(100):
        X, Y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

def main():
    print(f"Training Nexa on {DEVICE}")
    
    # We are now training on chat data to make it behave like a chatbot
    text = load_data('data/chat_data.txt')
    enc = tiktoken.get_encoding("gpt2")
    data = np.array(enc.encode(text))
    
    if len(data) <= BLOCK_SIZE:
        print("Data is too short for the block size. Please add more text to data/sample.txt")
        return

    # Model Setup (Using smaller config for demo)
    config = NexaConfig(vocab_size=50304, n_embd=256, n_head=8, n_layer=4, block_size=BLOCK_SIZE)
    model = NexaGPT(config)
    model.to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in tqdm(range(MAX_ITERS)):
        if iter % EVAL_INTERVAL == 0:
            loss = estimate_loss(model, data)
            print(f"step {iter}: loss {loss:.4f}")

        xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training finished.")
    
    # Save Model
    torch.save(model.state_dict(), "nexa_model.pt")
    print("Model saved to nexa_model.pt")

if __name__ == "__main__":
    main()
