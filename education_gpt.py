import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up the app
st.set_page_config(page_title="Education GPT", page_icon="📚")
st.title("📚 Education GPT")
st.caption("Your personal AI tutor for academic doubts")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["tinyllama/TinyLlama-1.1B-Chat-v1.0", "microsoft/phi-2"],
        index=0,
        help="Choose the LLM model to use"
    )
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, help="Controls randomness")
    max_length = st.slider("Max Length", 100, 1000, 300, help="Maximum response length")

    st.divider()
    st.markdown("""
    **About Education GPT**

    This is a mini LLM designed to help students with:
    - Math problems
    - Science concepts
    - Programming questions
    - History facts
    - Literature analysis
    - And more!
    """)


# Initialize the model (with caching)
@st.cache_resource
def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model


# Load the selected model
try:
    tokenizer, model = load_model(model_choice)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Education GPT. How can I help you with your studies today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your academic question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Prepare the prompt with chat history
        chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        input_text = f"{chat_history}\nassistant:"

        # Tokenize and generate
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})