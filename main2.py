import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import os

# Авторизация
login(token=os.getenv("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"))

@st.cache_resource
def load_model():
    model_name = "black-forest-labs/FLUX.1-dev"
    
    # Явно указываем класс модели для загрузки
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=True,
        trust_remote_code=True  # Добавляем для кастомных моделей
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=True,
        trust_remote_code=True,  # Критически важно для этой модели
        torch_dtype=torch.float16  # Для экономии памяти
    )
    return model, tokenizer

def main():
    st.title("FLUX.1-dev Text Generation")
    model, tokenizer = load_model()
    
    # Автоматическое определение устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Интерфейс
    prompt = st.text_area("Input text:", height=150)
    max_length = st.slider("Max length", 100, 2000, 512)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    
    if st.button("Generate") and prompt:
        with st.spinner("Generating..."):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Генерация с адаптированными параметрами
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            st.subheader("Result:")
            st.markdown(f"```\n{generated_text}\n```")

if __name__ == "__main__":
    main()