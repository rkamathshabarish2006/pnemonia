# Pneumonia Detection from Chest X-Ray Images

A deep-learning project that classifies chest X-ray images into two categories:

- **NORMAL** — no pneumonia detected by the model
- **PNEUMONIA** — pneumonia predicted by the model

The project is implemented in a Google Colab Jupyter Notebook using TensorFlow/Keras. It combines an ImageNet-pretrained **EfficientNetB0** convolutional neural network with **Transformer blocks** for image-feature learning and binary classification.

> **Disclaimer:** This is an educational project and research prototype. It is not a medical device and must not be used to diagnose, treat, or rule out pneumonia. Always consult a qualified healthcare professional.

## Project features

- Downloads the Chest X-Ray Images (Pneumonia) dataset using `kagglehub`.
- Locates the training, validation, and test folders and counts the images.
- Displays sample NORMAL and PNEUMONIA X-ray images.
- Loads images at **224 × 224** pixels in RGB format.
- Uses data augmentation (random flip, rotation, zoom, translation, and contrast).
- Calculates class weights to help address class imbalance.
- Uses EfficientNetB0 for CNN feature extraction.
- Projects CNN features into Transformer embeddings and processes them with two Transformer blocks.
- Trains a binary classifier and evaluates it using accuracy, precision, recall, and AUC.
- Generates a confusion matrix and classification report.
- Includes a single-image prediction workflow and a Grad-CAM visualization section in the notebook.
- Saves the trained model as `pneumonia_cnn_transformer_final.keras`.

## Model architecture

The notebook's model pipeline is:

1. Input chest X-ray image (224 × 224 × 3)
2. Data augmentation
3. EfficientNetB0 (ImageNet pretrained; fine-tuned in the later training stage)
4. 1 × 1 convolution feature projection (256 channels)
5. Reshape feature map into image tokens
6. Transformer block 1 (multi-head self-attention and feed-forward network)
7. Transformer block 2
8. Global average pooling
9. Dropout and dense classification layers
10. Sigmoid output for binary classification

The prediction threshold used in the notebook is **0.5**:
- Probability ≥ 0.5 → PNEUMONIA
- Probability < 0.5 → NORMAL

## Dataset

The notebook downloads the dataset from Kaggle using:

`paultimothymooney/chest-xray-pneumonia`

Dataset page: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The expected directory structure is:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

The notebook searches for these folders after downloading the dataset, so the local Kaggle cache path does not need to be hard-coded.

## Requirements

The notebook is designed to run in **Google Colab**. Main Python packages used include:

- TensorFlow / Keras
- kagglehub
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- Seaborn

Install the additional packages in a Colab cell if required:

```python
!pip install -q -U kagglehub scikit-learn seaborn
```

TensorFlow is commonly preinstalled in Google Colab. If running outside Colab, install a TensorFlow version compatible with your Python version and system.

## How to run

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook**.
3. Upload `newPnemonia.ipynb` from this repository.
4. Select a runtime. A GPU runtime is recommended.
5. Run the notebook cells in order.
6. Allow `kagglehub` to download the dataset.
7. Train the model and review the test metrics and classification report.
8. Run the prediction section to upload a chest X-ray image and view the model output.
9. The notebook saves the model to:

   `pneumonia_cnn_transformer_final.keras`

### Important notes before running

- The notebook contains repeated dataset-loading and augmentation cells; running the later definitions will replace the earlier variables.
- The training cell uses a `callbacks` variable. Make sure the callbacks are defined in the notebook before running that training cell (for example, EarlyStopping and ModelCheckpoint).
- The notebook's Grad-CAM section checks for specific layer names. Layer names can differ depending on the model version/build, so verify the actual layer names if that section reports missing layers.
- The notebook uses Google Colab's `files.upload()` and `files.download()` utilities for image/model transfer. These cells are Colab-specific.

## Evaluation

The notebook evaluates the model on the test dataset and reports:

- Accuracy
- Precision
- Recall
- AUC
- Confusion matrix
- Per-class precision, recall, and F1-score

**Add your actual final test results here after training.** Do not report training or validation accuracy as test accuracy.

| Metric | Test result |
|---|---:|
| Accuracy | Add result |
| Precision | Add result |
| Recall | Add result |
| AUC | Add result |

## Repository structure

```text
Pneumonia-Detection/
├── newPnemonia.ipynb
└── README.md
```

If you choose to add the trained model, place `pneumonia_cnn_transformer_final.keras` in the repository or provide a separate download link. Model files can be large; Git LFS or a cloud storage link may be more suitable. Do not upload private data or large dataset archives to GitHub.

## Future improvements

- Add a separate inference script or web application.
- Validate the model on an independent dataset from a different source.
- Tune the classification threshold using validation data.
- Add ROC and precision-recall curves.
- Improve Grad-CAM layer selection and explainability validation.
- Assess class-wise performance and possible dataset bias.

## Author

Add your name and contact or GitHub profile here.

---

If you use this repository, please cite the dataset source and clearly state that the model is an educational prototype, not a clinical diagnostic tool.
