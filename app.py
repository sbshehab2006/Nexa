import torch
import tiktoken
import streamlit as st
from model import NexaGPT, NexaConfig

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "nexa_model.pt"

# Page Config
st.set_page_config(page_title="Nexa AI", page_icon="🤖", layout="centered")

# Custom CSS for chat interface
st.markdown("""
<style>
    .stTextInput {
        position: fixed;
        bottom: 30px;
        width: 100%;
        max-width: 700px;
    }
    .chat-container {
        margin-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load Model Configuration (Must match train.py)
    config = NexaConfig(vocab_size=50304, n_embd=256, n_head=8, n_layer=4, block_size=64)
    model = NexaGPT(config)
    
    try:
        map_loc = DEVICE
        model.load_state_dict(torch.load(MODEL_PATH, map_location=map_loc))
        model.eval()
        model.to(DEVICE)
        return model
    except FileNotFoundError:
        return None

def generate_response(model, history):
    enc = tiktoken.get_encoding("gpt2")
    
    # Format the prompt with history
    # Simple format: User: ... Assistant: ...
    prompt = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Assistant: "

    start_ids = enc.encode(prompt)
    idx = torch.tensor(start_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Generation parameters
    max_new_tokens = 50
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            
            # Simple temperature sampling
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Stop if newline is generated (simple stop condition for chat)
            if idx_next.item() == enc.encode('\n')[0]:
                 break
                 
            idx = torch.cat((idx, idx_next), dim=1)
    
    # Decode only the new part
    generated_ids = idx[0].tolist()[len(start_ids):]
    response = enc.decode(generated_ids)
    return response.strip()

# Initialize State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header
st.title("🤖 Nexa AI Chat")
st.caption("A custom LLM built from scratch with PyTorch")

# Load Model
model = load_model()
if model is None:
    st.error("Model file not found! Please run `python train.py` first.")
    st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Message Nexa..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(model, st.session_state.messages)
            st.markdown(response)
    
    # Add assistant message to state
    st.session_state.messages.append({"role": "assistant", "content": response})
