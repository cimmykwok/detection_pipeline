# 📄 Container Identification and Recognition Pipeline

This repository contains the `detection_pipeline.py` script, which is designed for the automatic **identification and recognition of container units (TUs)** from images.

The pipeline performs object detection, text extraction using OCR, and critical validation against the ISO 6346 standard to ensure data accuracy.

---

## ✨ Features

* **Object Detection (MMDetection):** Utilizes an **MMDetection** model to locate and classify objects of interest, including container units, associated logos, and text blocks (`id_text`).
* **Hierarchical Association:** Links detected text (`id_text`) and logo objects to their containing **container** object using bounding box containment logic.
* **Logo Mapping:** Employs a JSON configuration to map detected logo labels to their corresponding 4-letter container owner codes (e.g., *SUDU*, *MAEU*).
* **Optical Character Recognition (OCR) (MMOCR):** Integrates an **MMOCR** inferencer for precise text detection and recognition on cropped container regions.
* **ISO 6346 Validation & Correction:** Implements the **ISO 6346 check digit calculation** and validation. The detected logo code is used to correct potential misrecognitions in the owner code section of the container ID.
* **Output Management:** Organizes and saves cropped images of containers and text, and stores the final detected and validated container information in a structured format via the `DetectedItems.save_to_file` method.

---

## ⚙️ Setup and Prerequisites

This pipeline relies on the **OpenMMLab** ecosystem for computer vision tasks and requires a specific environment setup.

### 1. Python Environment

Ensure you have a Python environment (e.g., using `conda` or `venv`) set up. The script is configured to use a **CUDA-enabled GPU** (`device='cuda:0'`).

### 2. Dependencies

The following core libraries are required. Please refer to the official OpenMMLab documentation for the exact installation steps, especially for CUDA compatibility.

| Category | Libraries | Installation Example |
| :--- | :--- | :--- |
| **OpenMMLab** | `mmcv`, `mmdet`, `mmocr` | `mim install mmcv mmdet mmocr` |
| **Scientific** | `torch`, `torchvision`, `torchaudio` | `pip install torch torchvision torchaudio` |
| **Other** | `numpy`, `opencv-python` (`cv2`) | `pip install numpy opencv-python` |

```bash
# Recommended OpenMMLab Installation Flow
pip install torch torchvision torchaudio
pip install openmim
mim install mmcv
mim install mmdet
mim install mmocr
pip install numpy opencv-python
```

## 🚀 Execution Guide

### 1. Dataset Configuration

Before running the pipeline, you **must** configure the datasets for both the object detection and OCR models.

#### For Object Detection (MMDetection):

This step involves modifying the core MMDetection installation to recognize the pipeline's custom dataset format.

1.  **Add Dataset Config:** Copy the custom configuration file `dataset/config/coco_detection_igd.py` to the following folder in your MMDetection installation:
    ```
    mmdet/configs/_base_/datasets/
    ```
2.  **Add Dataset Class:** Copy the custom dataset class file `dataset/config/coco.py` to the MMDetection datasets folder:
    ```
    mmdet/datasets
    ```
3.  **Annotation Files:** Ensure your dataset annotation files are located in:
    ```
    dataset/igd_dataset/annotation
    ```

#### For Text Detection and Recognition (MMOCR):

* Refer to the **TextSpotting** project from InteGreatDrones Projects for instructions on converting the **TRUDI dataset** for use with MMOCR

> **IMPORTANT:** You must also update the corresponding dataset paths in the `config.py` file for each model (located in the `model` folder) to reflect your file system.

### 2. Running the Pipeline

Once the MMDetection and MMOCR environments are properly set up and all dataset configurations are complete, you can adjust the pipeline settings in `config.py` and execute the script:

```bash
python detection_pipeline.py
```
