import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Загрузка модели с кэшированием
@st.cache_resource
def load_model():
    model_name = "black-forest-labs/FLUX.1-dev"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def main():
    st.title("FLUX.1-dev Text Generation")
    
    # Загрузка модели и токенизатора
    model, tokenizer = load_model()
    
    # Выбор устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Поле для ввода текста
    prompt = st.text_area("Enter your prompt:", height=100)
    
    # Параметры генерации
    max_length = st.slider("Max length", 50, 500, 100)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8)
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating..."):
                # Токенизация и генерация
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Генерация текста
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Декодирование результата
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.subheader("Generated Text:")
                st.write(generated_text)
        else:
            st.warning("Please enter a prompt")

if __name__ == "__main__":
    main()