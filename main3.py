import sys
import locale
import codecs

# Фикс кодировки для POSIX систем
if sys.platform.startswith("linux"):
    locale.getpreferredencoding = lambda: "UTF-8"
    sys.stdin = codecs.getreader("UTF-8")(sys.stdin.detach())
    sys.stdout = codecs.getwriter("UTF-8")(sys.stdout.detach())
    
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

login(token="ТОКЕН")

# Загрузка модели с кэшированием
@st.cache_resource
def load_model():
    model_name = "black-forest-labs/FLUX.1-dev"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
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