import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading VLM...")
print("Device:", DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)

model = model.to(DEVICE)
model.eval()

print("VLM loaded successfully.")


def generate_explanation(
    image_path,
    condition="Unknown",
    detections=None
):
    image = Image.open(image_path).convert("RGB")

    if detections is None:
        detections = []

    detection_text = ", ".join(detections) if detections else "No objects detected"

    prompt = f"""
You are an assistant for a computer vision system.

Environmental condition detected: {condition}
Detected objects: {detection_text}

Analyze the image and give a short explanation.
Mention:
1. The environmental condition.
2. The main visible objects.
3. How the environmental condition affects visibility or detection.

Do not invent objects that are not visible.
Keep the explanation concise and professional.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text