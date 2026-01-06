import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import sys
import os

# Add project root to path so we can import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import NexaRNN

def train():
    # Paths
    sp_model_path = os.path.join(os.path.dirname(__file__), '../tokenizer/nexa.model')
    data_path = os.path.join(os.path.dirname(__file__), '../data/train.txt')
    save_path = os.path.join(os.path.dirname(__file__), '../nexa_llm.pth')

    # Load Tokenizer
    sp = spm.SentencePieceProcessor()
    if not os.path.exists(sp_model_path):
        print("Tokenizer model not found. Please run tokenizer/tokenizer.py first.")
        return
    sp.load(sp_model_path)
    
    vocab_size = sp.get_piece_size()
    print(f"Vocab Size: {vocab_size}")

    # Load Data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Encode Data
    token_ids = sp.encode_as_ids(text)
    data = torch.tensor(token_ids, dtype=torch.long)
    
    # Create Sequences (x -> y)
    # x: tokens[0..n-1], y: tokens[1..n]
    seq_length = 10
    inputs = []
    targets = []
    
    for i in range(len(data) - seq_length):
        inputs.append(data[i:i+seq_length])
        targets.append(data[i+1:i+seq_length+1])
        
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    # Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    model = NexaRNN(vocab_size, embed_size=64, hidden_size=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Training Loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(inputs) # (batch, seq, vocab)
        
        # Reshape for loss: (batch * seq, vocab) vs (batch * seq)
        loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
    # Save Model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
