import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

def main():
    st.title("FLUX.1-dev Image Generation")
    
    # Конфигурация через Secrets.toml
    api_key = st.secrets.get("HF_API_KEY", "your-api-key-here")
    
    # Инициализация клиента
    client = InferenceClient(token=api_key)
    
    # Интерфейс
    prompt = st.text_area("Enter your prompt:", 
                         "Astronaut riding a horse",
                         height=100)
    
    # Параметры генерации (только поддерживаемые)
    with st.expander("Advanced Settings"):
        width = st.slider("Width", 256, 1024, 512)
        height = st.slider("Height", 256, 1024, 512)
        steps = st.slider("Steps", 10, 100, 25)
        cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 7.0)
    
    if st.button("Generate Image"):
        if not prompt:
            st.error("Please enter a prompt")
            return
            
        with st.spinner("Generating image..."):
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