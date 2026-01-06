import torch
import tiktoken
from model import NexaGPT, NexaConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_text(model, start_text, max_new_tokens=100):
    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(start_text)
    idx = torch.tensor(start_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    model.eval()
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return enc.decode(idx[0].tolist())

def main():
    # Load Model (Match config from train.py)
    # Note: In a real project, save config to file
    config = NexaConfig(vocab_size=50304, n_embd=256, n_head=8, n_layer=4, block_size=64)
    model = NexaGPT(config)
    
    try:
        model.load_state_dict(torch.load("nexa_model.pt", map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No trained model found! Please run train.py first.")
        print("Using initialized weights (garbage output expected).")

    model.to(DEVICE)
    
    start_text = input("Enter prompt (default: 'Nexa is'): ") or "Nexa is"
    
    print("\nGenerating...")
    generated = generate_text(model, start_text)
    print("\n--- Generated Text ---\n")
    print(generated)
    print("\n----------------------")

if __name__ == "__main__":
    main()
