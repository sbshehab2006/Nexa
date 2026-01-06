import torch
import torch.nn as nn

class NexaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super(NexaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embed = self.embedding(x)  # (batch, seq, embed_size)
        output, _ = self.gru(embed) # (batch, seq, hidden_size)
        
        # Predict next token for each step
        logits = self.fc(output) # (batch, seq, vocab_size)
        return logits
