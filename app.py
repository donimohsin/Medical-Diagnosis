"""Gradio demo: upload a scan -> predicted class + probabilities + Grad-CAM heat-map.
Run: python app.py   (needs models/best_model.pt from train.py).  Research/education only, NOT a medical device."""
import os, torch, gradio as gr
from src.model import load_checkpoint, eval_transform
from src.gradcam import GradCAM
from src.utils import overlay_heatmap

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, classes = load_checkpoint(MODEL_PATH, device)
cam_gen, tf = GradCAM(model, model.layer4[-1]), eval_transform()

def diagnose(img):
    if img is None:
        return None, None
    img = img.convert("RGB")
    cam, idx, probs = cam_gen(tf(img).unsqueeze(0).to(device))
    return {c: float(probs[i]) for i, c in enumerate(classes)}, overlay_heatmap(img, cam)

demo = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="pil", label="Upload a scan (MRI / X-ray)"),
    outputs=[gr.Label(num_top_classes=len(classes), label="Prediction"),
             gr.Image(label="Grad-CAM: regions that influenced the decision")],
    title="🩻 Medical Image Diagnosis Assistant",
    description="Deep-learning classifier (ResNet transfer learning) with Grad-CAM explainability. "
                "**For education and research only. Not a medical device. Do not use for real diagnosis.**",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
