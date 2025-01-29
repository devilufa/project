import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

def main():
    st.title("Генерация изображений с помощью текстового запроса с использованием модели FLUX.1-dev")
    
    # Конфигурация через Secrets.toml
    api_key = st.secrets.get("HF_API_KEY", "your-api-key-here")
    
    # Инициализация клиента
    client = InferenceClient(token=api_key)
    
    # Интерфейс
    prompt = st.text_area("Что нарисовать? (запрос должен быть на английском):", 
                         "Например: Astronaut riding a horse",
                         height=100)
    
    # Параметры генерации (только поддерживаемые)
    with st.expander("Расширенные настройки"):
        width = st.slider("Ширина", 256, 1024, 512)
        height = st.slider("Высота", 256, 1024, 512)
        steps = st.slider("Шаг", 10, 100, 25)
        cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 7.0)
    
    if st.button("Нарисовать!"):
        if not prompt:
            st.error("Нужен запрос!")
            return
            
        with st.spinner("Рисую..."):
            try:
                # Генерация изображения с актуальными параметрами
                image = client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-dev",
                    width=width,
                    height=height,
                    guidance_scale=cfg_scale,
                    num_inference_steps=steps
                )
                
                # Конвертация в BytesIO
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                
                st.image(img_byte_arr.getvalue(),
                       caption=prompt,
                       use_column_width=True)
                
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    main()