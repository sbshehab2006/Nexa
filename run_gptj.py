import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def run_gptj():
    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    model_name = "EleutherAI/gpt-j-6B"
    
    # Check hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detected: {device.upper()}")

    # ------------------------------------------------------------------
    # 2. Optimization Strategy
    # ------------------------------------------------------------------
    # If on GPU, we load in float16 to save memory (~12GB VRAM required)
    # If on CPU, we load in default float32 (Requires ~24GB RAM)
    
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # ------------------------------------------------------------------
    # 3. Load Tokenizer & Model
    # ------------------------------------------------------------------
    print(f"Loading model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision="float16" if device == "cuda" else "main",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            # device_map="auto" # Uncomment this if using bitsandbytes/accelerate
        ).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded successfully!")

    # ------------------------------------------------------------------
    # 4. Generation Loop
    # ------------------------------------------------------------------
    while True:
        prompt = input("\nEnter prompt (or 'q' to quit): ").strip()
        if prompt.lower() == 'q':
            break
            
        if not prompt:
            continue

        print("\nGenerating...")
        start_time = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=100,      # Max length of generated text
                do_sample=True,          # Enable sampling (creative)
                temperature=0.9,         # Higher = more creative
                top_p=0.95,              # Nucleus sampling
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        end_time = time.time()

        print(f"--- Result ({end_time - start_time:.2f}s) ---")
        print(generated_text)
        print("-----------------------------------------")

if __name__ == "__main__":
    run_gptj()
