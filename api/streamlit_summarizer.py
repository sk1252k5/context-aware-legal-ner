import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load your fine-tuned T5 model
MODEL_PATH = "summarization/t5_model.pth"
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load trained weights
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

st.title("📄 Legal Document Summarizer")
st.write("Upload your **legal text** below and get a concise summary powered by T5.")

# Text input
user_input = st.text_area("Enter Legal Document Text Here:", height=300)

if st.button("Generate Summary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please input some text.")
    else:
        # Prepare input for T5
        input_ids = tokenizer.encode(
            "summarize: " + user_input,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Display output
        st.subheader("📑 Summary:")
        st.success(summary)
