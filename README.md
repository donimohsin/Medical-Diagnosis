# 🩻 Medical Image Diagnosis with Explainable AI (ResNet + Grad-CAM)

A deep-learning assistant that classifies medical scans (brain-tumor MRI or chest X-ray) using **transfer learning**
and explains each prediction with **Grad-CAM heat-maps** that highlight the image regions that drove the decision.

**Developed by Mohammad Mohsin Doni**

> ⚠️ **Disclaimer:** This is an educational project. It is **not a medical device** and must not be used for real diagnosis.

---

## Features
- **Transfer learning** with an ImageNet-pretrained ResNet50 (ResNet18 option for faster runs)
- **Two-phase training:** train a new classifier head first, then fine-tune the whole network at a lower learning rate
- **Class-weighted loss** to handle imbalanced classes
- **Honest evaluation:** best epoch chosen on a validation split, final numbers reported on an untouched test set
  (accuracy, per-class precision / recall / F1, confusion matrix)
- **Grad-CAM implemented from scratch** (no external library) for explainability
- **Gradio web app:** upload a scan, get class probabilities plus the heat-map
- Works with any `Training/<class>/` + `Testing/<class>/` image dataset

## How it works
```
Scan (JPG/PNG) -> resize 224x224 + normalise -> ResNet50 (ImageNet weights, new classifier head)
                                                   |-- softmax -> class probabilities
                                                   '-- last conv block + gradients -> Grad-CAM heat-map overlay
```
**Grad-CAM:** the gradients of the predicted class score are averaged per feature map to get importance weights;
the weighted sum of the last convolutional feature maps (passed through ReLU) shows where the model "looked".

## Project structure
```
medical-diagnosis/
├── train.py                      # training, evaluation, plots, Grad-CAM examples
├── app.py                        # Gradio demo
├── src/
│   ├── model.py                  # ResNet builder, transforms, checkpoint loader
│   ├── gradcam.py                # Grad-CAM from scratch
│   └── utils.py                  # dataset folder discovery, heat-map overlay
├── notebooks/train_on_gpu.ipynb  # ready-to-run Kaggle / Colab notebook
├── models/                       # best_model.pt is saved here (not committed)
├── outputs/                      # metrics.json + plots (generated)
└── requirements.txt
```

## Quick start: train on a free GPU (recommended)
Training a ResNet50 on a laptop CPU is very slow, so use a free GPU.

1. Push this project to GitHub (or upload the folder).
2. Open `notebooks/train_on_gpu.ipynb` on **[Kaggle](https://www.kaggle.com/code)** (Accelerator: GPU) or **Google Colab** (T4 GPU).
3. Add the dataset:
   - **Brain Tumor MRI Dataset** (Masoud Nickparvar, Kaggle), 4 classes: glioma, meningioma, pituitary, no tumor
   - or **Chest X-Ray Images (Pneumonia)** (Paul Mooney, Kaggle), 2 classes
4. Run all cells. Training takes roughly 10-20 minutes on a T4.
5. Download `models/best_model.pt` and the `outputs/` folder.

Or from a terminal with a GPU:
```bash
pip install -r requirements.txt
python train.py --data /path/to/dataset --arch resnet50 --head_epochs 2 --epochs 8
```
(On Windows add `--workers 0`.)

## Run the demo
```bash
pip install -r requirements.txt
python app.py        # open http://127.0.0.1:7860 , needs models/best_model.pt
```
If PyTorch doesn't install on your Python version, run the demo from the notebook (last cell, creates a public link) or use Python 3.11/3.12.

## Results
*Fill in from `outputs/metrics.json` after training.*

| Metric | Value |
|---|---|
| Dataset | _e.g. Brain Tumor MRI (4 classes)_ |
| Test accuracy | _e.g. 0.xx_ |
| Macro F1 | _e.g. 0.xx_ |
| Best validation accuracy | _e.g. 0.xx_ |

**Screenshots to add:** `outputs/confusion_matrix.png`, `outputs/training_curves.png`, `outputs/gradcam_examples.png`, and the Gradio app.

## Scope and limitations (be upfront about these)
- This is a **single-label image classifier**. A full radiology assistant would also need multi-label pathology
  detection, lesion **segmentation** and automated **report generation**; those are future work.
- Public datasets are curated, so accuracy on them overstates real-world performance; there is no external validation here.
- Possible data leakage: some public datasets contain near-duplicate slices, which can inflate test accuracy.
- Heat-maps show where the model looked, not proof that the model reasoned like a clinician.
- No DICOM handling, HIPAA controls or audit logging.

## Future improvements
Vision Transformer (ViT) comparison, multi-label classification on NIH ChestX-ray14, U-Net / Mask R-CNN segmentation,
report generation, calibration and uncertainty estimates, DICOM support, FastAPI + Docker deployment.

## Author
**Mohammad Mohsin Doni**, developed as part of an AI internship project.
