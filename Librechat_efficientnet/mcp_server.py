# import os
# from inference import predict

# # ---------- Handle messages ----------
# # def handle_message(message):
# #     """
# #     Check if message contains 'image' keyword, then predict
# #     """
# #     if "image" in message.lower():
# #         # Expecting format: "Identify this plant disease: /path/to/image.jpg"
# #         parts = message.split(":")
# #         if len(parts) > 1:
# #             image_path = parts[1].strip()  # get image path
# #             if not os.path.exists(image_path):
# #                 return f"Error: File not found: {image_path}"
# #             try:
# #                 result = predict(image_path)
# #                 return f"Prediction: {result}"
# #             except Exception as e:
# #                 return f"Error: {str(e)}"
# #     return "Send an image like: 'Identify this plant disease: /path/to/image.jpg'"

# def handle_message(message):
#     """
#     Check if message contains 'image' keyword, then predict
#     """
#     if "image" in message.lower():
#         # Expecting format: "Identify this plant disease: /path/to/image.jpg"
#         parts = message.split(":")
#         if len(parts) > 1:
#             image_path = parts[1].strip()  # get image path
#             if not os.path.exists(image_path):
#                 return f"Error: File not found: {image_path}"
#             try:
#                 result = predict(image_path)
#                 # Split plant name and disease name
#                 plant_name, disease_name = result.split(" ", 1)
#                 return f"Plant: {plant_name} | Disease: {disease_name}"
#             except Exception as e:
#                 return f"Error: {str(e)}"
#     return "Send an image like: 'Identify this plant disease: /path/to/image.jpg'"

# # ---------- Simple local test loop ----------
# if __name__ == "__main__":
#     print("✅ MCP server started (local test loop)")
#     print("Type your message like: Identify this plant disease: /path/to/image.jpg")
#     while True:
#         msg = input("\nUser: ")
#         response = handle_message(msg)
#         print(f"LibreChat: {response}")
# ###########################################################
import os
from inference import predict

# ---------- Handle messages ----------
def handle_message(message):
    """
    Check if message contains 'image' keyword, then predict
    """
    if "image" in message.lower():
        # Expecting format: "Identify this plant disease: /path/to/image.jpg"
        parts = message.split(":")
        if len(parts) > 1:
            image_path = parts[1].strip()  # get image path
            if not os.path.exists(image_path):
                return f"Error: File not found: {image_path}"
            try:
                # Call predict() which now returns (plant_name, disease_name)
                plant_name, disease_name = predict(image_path)
                return f"Plant: {plant_name} | Disease: {disease_name}"
            except Exception as e:
                return f"Error: {str(e)}"
    return "Send an image like: 'Identify this plant disease: /path/to/image.jpg'"

# ---------- Simple local test loop ----------
if __name__ == "__main__":
    print("✅ MCP server started (local test loop)")
    print("Type your message like: Identify this plant disease: /path/to/image.jpg")
    while True:
        msg = input("\nUser: ")
        response = handle_message(msg)
        print(f"LibreChat: {response}")
