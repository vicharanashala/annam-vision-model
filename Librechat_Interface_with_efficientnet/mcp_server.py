
# ###########################################################
# import os
# from inference import predict

# # ---------- Handle messages ----------
# def handle_message(message: str) -> str:
#     """
#     Check if message contains 'image' keyword, then predict
#     """
#     if "image" in message.lower():
#         parts = message.split(":")
#         if len(parts) > 1:
#             image_path = parts[1].strip()
#             if not os.path.exists(image_path):
#                 return f"Error: File not found: {image_path}"
#             try:
#                 plant_name, disease_name = predict(image_path)
#                 return f"Plant: {plant_name} | Disease: {disease_name}"
#             except Exception as e:
#                 return f"Error: {str(e)}"
#     return "Send an image like: 'Identify this plant disease: /path/to/image.jpg'"

# # ---------- Entry point ----------
# if __name__ == "__main__":
#     # Check if we are in LibreChat mode (set by config.json or env variable)
#     librechat_mode = os.environ.get("LIBRECHAT_MODE", "false").lower() == "true"

#     if librechat_mode:
#         print("✅ MCP server started in LibreChat mode")
#         # No input() loop here — LibreChat will send/receive messages
#         # Just keep the process alive
#         import time
#         while True:
#             time.sleep(1)
#     else:
#         print("✅ MCP server started (local test loop)")
#         print("Type your message like: Identify this plant disease: /path/to/image.jpg")
#         while True:
#             try:
#                 msg = input("\nUser: ")
#             except EOFError:
#                 break
#             response = handle_message(msg)
#             print(f"LibreChat: {response}")

# #####################################
import os
import sys
import json
import time
from inference import predict

# ---------- Handle messages ----------
def handle_message(message: str) -> str:
    """
    Check if message contains 'image' keyword, then predict
    """
    if "image" in message.lower() or "identify this plant disease" in message.lower():
        parts = message.split(":")
        if len(parts) > 1:
            image_path = parts[1].strip()
            if not os.path.exists(image_path):
                return f"Error: File not found: {image_path}"
            try:
                plant_name, disease_name = predict(image_path)
                return f"Plant: {plant_name} | Disease: {disease_name}"
            except Exception as e:
                return f"Error: {str(e)}"
    return "Send an image like: 'Identify this plant disease: /path/to/image.jpg'"

# ---------- Entry point ----------
if __name__ == "__main__":
    # Check if we are in LibreChat mode (set by config.json or env variable)
    librechat_mode = os.environ.get("LIBRECHAT_MODE", "false").lower() == "true"

    if librechat_mode:
        print("✅ MCP server started in LibreChat mode")
        # Read messages from stdin (sent by LibreChat backend)
        for line in sys.stdin:
            try:
                data = json.loads(line.strip())
                message = data.get("content", "")
                response = handle_message(message)
                # Send back JSON response
                sys.stdout.write(json.dumps({"role": "assistant", "content": response}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(json.dumps({"role": "assistant", "content": f"Error: {str(e)}"}) + "\n")
                sys.stdout.flush()
    else:
        print("✅ MCP server started (local test loop)")
        print("Type your message like: Identify this plant disease: /path/to/image.jpg")
        while True:
            try:
                msg = input("\nUser: ")
            except EOFError:
                break
            response = handle_message(msg)
            print(f"LibreChat: {response}")