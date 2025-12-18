# scripts/dataset_to_jsonl.py
import os
import jsonlines

DATASET_ROOT = "/home/aic_u3/aic_u3/ComputerVision/Perception_Models/Potato_Tomato_G-Models/Dataset_Tomato-Potato_split_T_V"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset_json")
os.makedirs(OUT_DIR, exist_ok=True)

USER_INSTRUCTION = (
    "Identify the plant and the disease from the image and provide:\n"
    "- Plant name\n"
    "- Disease name\n"
    "- Symptoms\n"
    "- Treatment steps"
)

def write_split(split):
    split_path = os.path.join(DATASET_ROOT, split)
    out_file = os.path.join(OUT_DIR, f"{split}.jsonl")

    with jsonlines.open(out_file, "w") as writer:
        for cls in sorted(os.listdir(split_path)):
            class_dir = os.path.join(split_path, cls)
            if not os.path.isdir(class_dir):
                continue

            for img in sorted(os.listdir(class_dir)):
                if not img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                img_path = os.path.join(class_dir, img)

                text = (
                    "<|vision_start|><|image_pad|><|vision_end|>\n"
                    f"{USER_INSTRUCTION}\n\n"
                    f"{cls}"
                )

                writer.write({
                    "image": img_path,
                    "text": text
                })

    print(f"✓ Wrote JSONL: {out_file}")


if __name__ == "__main__":
    write_split("train")
    write_split("val")
