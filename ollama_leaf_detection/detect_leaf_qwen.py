# import ollama

# image_path = "/home/aic_u3/aic_u3/ComputerVision/DINO_large/Benchmark_Dataset-CDDM_images/Benchmark_Dataset-CDDM_images/images/Potato_Early_blight/plant_64484.jpg"

# with open(image_path, "rb") as f:
#     img = f.read()

# response = ollama.chat(
#     model="bakllava",      # change model here
#     messages=[
#         {
#             "role": "user",
#             "content": (
#                 "This is a crop leaf. Identify the plant species and the exact disease name. "
#                 "Give: (1) Correct plant name, (2) Disease name, (3) Symptoms, (4) Treatment steps."
#             ),
#             "images": [img]
#         }
#     ]
# )

# print(response["message"]["content"])


# # model="qwen2-vl"
# # model="llava-phi3"
# # model="moondream"
# # model="bakllava"
# model="qwen3-vl:8b"

# ###############################################
import ollama
import base64

image_path = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V/val/Tomato Yellow Leaf Curl Virus/0ec33982-87fc-49c4-a61e-5459a2497045___UF.GRC_YLCV_Lab_01749.JPG"

# convert image → base64
with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = ollama.chat(
    model="qwen3-vl:8b",   # <--- change here
    messages=[
        {
            "role": "user",
            "content": (
                "This is a crop leaf. Identify the plant species and the exact disease name. "
                "Give: (1) Plant name, (2) Disease name, (3) Symptoms, (4) Treatment steps."
            ),
            "images": [img_b64]
        }
    ]
)

print(response["message"]["content"])
