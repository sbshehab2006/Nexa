import sentencepiece as spm
import os

def train_tokenizer():
    input_file = os.path.join(os.path.dirname(__file__), '../data/train.txt')
    model_prefix = os.path.join(os.path.dirname(__file__), 'nexa')
    
    # Train SentencePiece BPE model
    # vocab_size=100 as requested (small for testing/demo)
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=500,
        model_type='bpe',
        character_coverage=0.9995, 
        # Making it fall back to bytes to avoid unknown char errors in small data
        byte_fallback=True 
    )
    
    print(f"Tokenizer trained. Files saved to {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    train_tokenizer()
