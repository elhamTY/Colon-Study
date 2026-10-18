# Explainability and Feature Fusion for Colon CT Image Classification

Code for the manuscript *Explainability and Feature Fusion for Colon CT Image Classification*
(Elham Tahsin Yasin, Sinop University).

The study uses the public **ColonCancerCT-2025: Abdominal CT Scans** dataset (Kaggle, CC BY 4.0):
<https://www.kaggle.com/datasets/orvile/coloncancerct-2025-abdominal-ct-scans>.
No images are redistributed in this repository.

## Contents

| Path | Purpose |
|---|---|
| `notebooks/XAI_Radiomics_ColonCancer_Study.ipynb` | Original study: EfficientNet-B0 training, evaluation, Grad-CAM++ / Integrated Gradients / GradientSHAP / LIME, deletion–insertion evaluation, parameter randomization, Grad-CAM++ ROI extraction, handcrafted descriptors and the six XGBoost feature strategies. |
| `revision/prep_images.py` | Step 0: reads every PNG of `Split_Data/{train,val,test}/{Non_Cancer,cancer}`, records image metadata and MD5 hashes, and stores the 224 × 224 network inputs. |
| `revision/common.py` | Model definition, checkpoint loading, Grad-CAM++ ROI extractor and handcrafted descriptors (original and corrected shape calculation). |
| `revision/s1_verify.py` | Reproduces the test-set predictions of the saved checkpoint (3 errors, ROC-AUC 0.999997). |
| `revision/s2_dups.py`, `revision/s2b_nn.py` | Exact-duplicate analysis (MD5 of decoded pixels), perceptual hashing, nearest-neighbour SSIM between each test image and the training/validation images, and the raw-pixel 1-nearest-neighbour baseline. |
| `revision/bodymask.py`, `revision/s4_mask.py`, `revision/s4b_report.py` | Body/background masking experiment. |
| `revision/s3_features.py` | Deterministic feature extraction: one row per original image, no augmentation or resampling, file names retained; corrected shape descriptors. |
| `revision/s5_xgb.py` | Six XGBoost strategies on the deterministic features; `--check` reproduces the first-version Table 3 from the first-version feature exports (`train/test_all_features_combined.csv`, available on request). |
| `revision/stats_util.py` | Wilson and Hanley–McNeil intervals, exact McNemar test. |

## Data layout

```
Split_Data/
  train/{Non_Cancer,cancer}/image_*.png   7,937 images
  val/{Non_Cancer,cancer}/image_*.png     1,701 images
  test/{Non_Cancer,cancer}/image_*.png    1,702 images
```

The partition used in the paper is defined by these folders; the file lists are given in `splits/`.

## Running the revision analyses

```bash
pip install -r requirements.txt
python revision/prep_images.py --root /path/to/Split_Data --out revision/data
# copy best_model_efficientnet_b0.pth (trained with the notebook) into revision/data/
cd revision
python s1_verify.py                 # reproduce test predictions
python s2_dups.py && python s2b_nn.py
python s4_mask.py && python s4b_report.py
for s in 0 600 1200; do python s3_features.py test $s $((s+600)); done
for s in $(seq 0 600 7800); do python s3_features.py train $s $((s+600)); done
python s5_xgb.py
```

Grad-CAM++ feature extraction runs on a CPU (about 0.4 s per image); a GPU is not required for the revision scripts.

## Software

The original study used Python 3.9.21, PyTorch 1.12.1+cu116, timm 1.0.14, grad-cam 1.5.5, Captum 0.8.0,
LIME 0.2.0.1, XGBoost 2.1.4, scikit-learn 1.6.1, scikit-image 0.24.0, OpenCV 4.11.0.86 and Albumentations 2.0.5.
The revision analyses were run on CPU with Python 3.13, PyTorch 2.x, timm 1.0.14, grad-cam 1.5.5 and XGBoost 2.1.4;
they reproduce the original test predictions exactly and the original Table 3 exactly from the original feature exports.

## Licence

MIT (code). The dataset is subject to its own licence.
