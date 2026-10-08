"""Condition classifier: load weights, preprocess, predict condition + confidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MISSING_MSG = (
    "Model weights are not available yet.\n"
    "Please place the trained model inside the models/ folder."
)


def load_classifier(cfg: dict):
    """Return (bundle, error_message). bundle is None when weights are missing or fail to load."""
    ccfg = cfg["classifier"]
    path = ROOT / ccfg["weights"]
    if not path.exists():
        return None, MISSING_MSG
    try:
        import torch
        import torchvision

        classes = cfg["classes"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location="cpu")

        if isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            arch = ccfg.get("arch", "resnet18")
            model = getattr(torchvision.models, arch)(weights=None, num_classes=len(classes))
            state = ckpt
            if isinstance(ckpt, dict):
                state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state)

        model.to(device).eval()
        return {"model": model, "device": device, "cfg": ccfg, "classes": classes}, None
    except Exception as exc:  # never crash the dashboard
        return None, f"Classifier could not be loaded: {exc}"


def preprocess(pil_image, ccfg: dict):
    """PIL image -> normalized tensor of shape (1, 3, H, W)."""
    from torchvision import transforms

    size = ccfg.get("input_size", 224)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(ccfg["mean"], ccfg["std"]),
    ])
    return tf(pil_image.convert("RGB")).unsqueeze(0)


def predict_condition(bundle: dict, pil_image):
    """Return (condition, confidence 0-1, {class: probability})."""
    import torch

    x = preprocess(pil_image, bundle["cfg"]).to(bundle["device"])
    with torch.no_grad():
        probs = torch.softmax(bundle["model"](x), dim=1)[0].cpu().tolist()
    classes = bundle["classes"]
    best = max(range(len(probs)), key=probs.__getitem__)
    return classes[best], float(probs[best]), dict(zip(classes, probs))
