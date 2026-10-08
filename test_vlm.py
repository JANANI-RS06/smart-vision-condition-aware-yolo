from pipeline.vlm import generate_explanation


IMAGE = "samples/image1.png"

result = generate_explanation(
    IMAGE,
    condition="Fog",
    detections=["car", "car", "truck"]
)

print("\n========== VLM RESULT ==========")
print(result)
print("================================")