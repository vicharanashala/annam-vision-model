import os
import json
from PIL import Image
from torch.utils.data import Dataset
from unsloth import FastVisionModel
from transformers import Trainer, TrainingArguments

# MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_JSONL = os.path.join(BASE_DIR, "..", "dataset_json", "train.jsonl")
VAL_JSONL   = os.path.join(BASE_DIR, "..", "dataset_json", "val.jsonl")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "outputs", "leaf_vlm_lora")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- JSONL loader ---
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(l) for l in f]

# --- Dataset ---
class LeafVLMdataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        prompt_text = item["text"]

        processed = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt"
        )

        # add labels for causal LM
        processed["labels"] = processed["input_ids"].clone()

        # squeeze batch dimension
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        return processed

# --- Load model ---
model, processor = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=False,
)

# --- Add LoRA ---
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- Load dataset ---
train_data = load_jsonl(TRAIN_JSONL)
val_data = load_jsonl(VAL_JSONL)

train_ds = LeafVLMdataset(train_data, processor)
val_ds = LeafVLMdataset(val_data, processor)

# --- Training arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=20,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

# --- Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("\n🎉 Training complete. Saved at:", OUTPUT_DIR)
