from huggingface_hub import InferenceClient
client = InferenceClient("strangerzonehf/Flux-Super-Realism-LoRA", token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# output is a PIL.Image object
image = client.text_to_image("Astronaut riding a horse")
