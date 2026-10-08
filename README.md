# Smart Vision - Condition-Aware Object Detection

    pip install -r requirements.txt
    streamlit run app.py

The dashboard runs even before any model is connected.

## Connect your models
- `models/classifier.pth` - 6-class ResNet-18 (state dict). Class order in `config.json` must match training.
- `models/detector.pt` - Ultralytics YOLO weights.
- `models/RealESRGAN_x4plus.pth` - optional; also `pip install realesrgan basicsr`.

## Results files (all optional, shown on the Performance page)
| File | Columns |
|------|---------|
| `results/classifier_results.csv` | class, accuracy, precision, recall, f1 (one `overall` row for the metric cards) |
| `results/confusion_matrix.csv` | first column = true class, other columns = predicted class counts (or drop in `confusion_matrix.png`) |
| `results/detection_results.csv` | model, precision, recall, map50, map50_95, inference_ms, fps, model_size_mb |
| `results/per_condition_results.csv` | model, condition, precision, recall, map50, map50_95, (optional) setting |

## Sample gallery
`python make_samples.py path/to/test_images` fills `samples/<condition>/` automatically.

## Reliability check
Plug in your logic with `register_reliability_checker(fn)` in `pipeline/detector.py`.
