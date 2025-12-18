import ollama

# Load image
image_path = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Field_Images/Tomato_Bacterial_spot/bacterial_spot_tomato-836x576.jpg"

with open(image_path, "rb") as f:
    image_bytes = f.read()

# Query VLM model
response = ollama.chat(
    model="llava:7b",   # or "qwen2.5-vl"
    messages=[
        {
            "role": "user",
            "content": (
                "Identify the plant species and detect if the leaf has any disease. "
                "Explain symptoms and give treatment steps."
            ),
            "images": [image_bytes]
        }
    ]
)

print("\n========== RESULT ==========\n")
print(response["message"]["content"])
