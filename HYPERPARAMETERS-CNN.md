# Hyperparameter Search Results

This report summarizes all hyperparameter runs.

## Summary for CNN

- [Hyperparameter Search Results](#hyperparameter-search-results)
  - [Summary for CNN](#summary-for-cnn)
    - [Archtecture classique](#archtecture-classique)
    - [Résultats](#résultats)
    - [Architecture amélioré](#architecture-amélioré)
    - [Résultats](#résultats-1)
    - [Architecture avec dropout et FC intermédiaire](#architecture-avec-dropout-et-fc-intermédiaire)
    - [Résultats](#résultats-2)
    - [Architecture avec downsampling par stride et gros classifieur](#architecture-avec-downsampling-par-stride-et-gros-classifieur)
    - [Résultats](#résultats-3)
    - [Architecture légère et rapide](#architecture-légère-et-rapide)
    - [Résultats](#résultats-4)

### Archtecture classique

| Type de couche | Détails                                   |
| -------------- | ----------------------------------------- |
| Entrée         | Image 1 canal (28×28 pixels)              |
| Convolution 1  | 32 filtres de taille 5×5, activation SiLU |
| Convolution 2  | 32 filtres de taille 5×5, activation SiLU |
| Normalisation  | Normalisation par lots (32 canaux)        |
| Pooling        | Max-pooling (réduction de taille)         |
| Convolution 3  | 64 filtres de taille 3×3, activation SiLU |
| Convolution 4  | 64 filtres de taille 3×3, activation SiLU |
| Normalisation  | Normalisation par lots (64 canaux)        |
| Pooling        | Max-pooling (réduction de taille)         |
| Aplatissement  | Conversion en vecteur 1D                  |
| Couche dense   | 576 neurones vers 10 neurones             |

### Résultats

| Batch |    LR | Steps | LR Decay | Patience | Time  |   Accuracy | Precision | Recall |     F1 |
| ----: | ----: | ----: | -------: | -------: | :---: | ---------: | --------: | -----: | -----: |
|   512 |  0.01 |   100 |      0.9 |       25 | 00:10 | **98.05%** |    0.9806 | 0.9802 | 0.9803 |
|   512 |  0.02 |   150 |     0.95 |       10 | 00:14 | **99.05%** |    0.9905 | 0.9905 | 0.9905 |
|   256 | 0.005 |   150 |      0.9 |       50 | 00:09 | **98.19%** |    0.9820 | 0.9817 | 0.9818 |
|   512 |  0.02 |    50 |      0.9 |       10 | 00:07 | **94.35%** |    0.9479 | 0.9428 | 0.9424 |
|  2048 |  0.02 |   200 |     0.85 |       25 | 01:05 | **99.39%** |    0.9939 | 0.9939 | 0.9939 |
|   512 |  0.02 |    50 |     0.85 |       50 | 00:07 | **97.14%** |    0.9717 | 0.9712 | 0.9712 |
|  1024 |  0.01 |   500 |     0.85 |       25 | 01:19 | **99.50%** |    0.9950 | 0.9949 | 0.9950 |
|  1024 |  0.02 |   150 |     0.95 |       25 | 00:26 | **99.06%** |    0.9905 | 0.9906 | 0.9905 |
|   512 |  0.01 |   500 |     0.95 |       10 | 00:41 | **99.38%** |    0.9938 | 0.9937 | 0.9938 |
|   256 | 0.005 |   500 |      0.9 |       50 | 00:23 | **99.20%** |    0.9920 | 0.9919 | 0.9919 |
|   512 |  0.05 |    50 |     0.95 |       25 | 00:07 | **96.37%** |    0.9645 | 0.9650 | 0.9641 |
|   256 | 0.005 |    50 |     0.85 |       50 | 00:05 | **92.23%** |    0.9259 | 0.9215 | 0.9220 |
|  1024 |  0.05 |   100 |     0.95 |       10 | 00:19 | **98.46%** |    0.9847 | 0.9845 | 0.9844 |
|   512 |  0.02 |   150 |     0.85 |       50 | 00:14 | **98.46%** |    0.9846 | 0.9847 | 0.9846 |
|  1024 |  0.05 |   150 |     0.95 |       50 | 00:26 | **98.85%** |    0.9886 | 0.9884 | 0.9884 |

---

### Architecture amélioré

| Type de couche     | Détails                                         |
| ------------------ | ----------------------------------------------- |
| Entrée             | Image 1 canal (28×28 pixels)                    |
| Convolution 1      | 32 filtres, noyau 3×3, padding=1                |
| Activation         | SiLU                                            |
| Convolution 2      | 32 filtres, noyau 3×3, padding=1                |
| Activation         | SiLU                                            |
| Normalisation      | BatchNorm (32 canaux)                           |
| Pooling            | Max-pooling (2×2)                               |
| Convolution 3      | 64 filtres, noyau 3×3, padding=1                |
| Activation         | SiLU                                            |
| Convolution 4      | 64 filtres, noyau 3×3, padding=1                |
| Activation         | SiLU                                            |
| Normalisation      | BatchNorm (64 canaux)                           |
| Pooling            | Max-pooling (2×2)                               |
| Global Avg Pooling | Moyenne spatiale (convertit en vecteur 64 dim.) |
| Couche dense       | 64 → 10 neurones (classification)               |

### Résultats

| Batch |    LR | Steps | LR Decay | Patience | Time  |   Accuracy | Precision | Recall |     F1 |
| ----: | ----: | ----: | -------: | -------: | :---: | ---------: | --------: | -----: | -----: |
|  1024 |  0.02 |   100 |      0.9 |       25 | 00:17 | **98.14%** |    0.9820 | 0.9815 | 0.9815 |
|   512 |  0.01 |   100 |      0.9 |       25 | 00:12 | **82.68%** |    0.8960 | 0.8239 | 0.8311 |
|   256 | 0.005 |    50 |     0.85 |       50 | 00:08 | **36.86%** |    0.4555 | 0.3589 | 0.2907 |
|   256 |  0.02 |   100 |      0.9 |       25 | 00:09 | **95.39%** |    0.9605 | 0.9535 | 0.9545 |
|  1024 |  0.05 |    50 |     0.95 |       25 | 00:10 | **96.72%** |    0.9708 | 0.9670 | 0.9678 |
|  1024 | 0.005 |   150 |     0.95 |       50 | 00:22 | **90.33%** |    0.9175 | 0.9023 | 0.9050 |
|   512 |  0.01 |    50 |     0.95 |       25 | 00:08 | **41.54%** |    0.6429 | 0.4101 | 0.3841 |

---

### Architecture avec dropout et FC intermédiaire

| Type de couche      | Détails                            |
| ------------------- | ---------------------------------- |
| Entrée              | Image 1 canal (28×28 pixels)       |
| Convolution 1       | 32 filtres, noyau 5×5              |
| Activation          | SiLU                               |
| Convolution 2       | 32 filtres, noyau 5×5              |
| Activation          | SiLU                               |
| Normalisation       | BatchNorm (32 canaux)              |
| Pooling             | Max-pooling (2×2)                  |
| Dropout             | Dropout 25%                        |
| Convolution 3       | 64 filtres, noyau 3×3              |
| Activation          | SiLU                               |
| Convolution 4       | 64 filtres, noyau 3×3              |
| Activation          | SiLU                               |
| Normalisation       | BatchNorm (64 canaux)              |
| Pooling             | Max-pooling (2×2)                  |
| Dropout             | Dropout 25%                        |
| Aplatissement       | Passage en vecteur 1D              |
| Couche dense cachée | 576 → 128 neurones                 |
| Activation          | SiLU                               |
| Dropout             | Dropout 50%                        |
| Couche dense finale | 128 → 10 neurones (classification) |

### Résultats

| Batch |    LR | Steps | LR Decay | Patience | Time  |   Accuracy | Precision | Recall |     F1 |
| ----: | ----: | ----: | -------: | -------: | :---: | ---------: | --------: | -----: | -----: |
|   512 |  0.02 |   100 |     0.95 |       50 | 00:12 | **98.50%** |    0.9849 | 0.9850 | 0.9849 |
|  1024 | 0.005 |    50 |     0.85 |       25 | 00:12 | **89.69%** |    0.9008 | 0.8955 | 0.8952 |
|   512 |  0.01 |   100 |      0.9 |       25 | 00:11 | **98.25%** |    0.9823 | 0.9824 | 0.9823 |
|   512 |  0.02 |   200 |     0.95 |       25 | 00:19 | **98.94%** |    0.9894 | 0.9893 | 0.9893 |
|   256 |  0.05 |   150 |     0.85 |       50 | 00:10 | **97.89%** |    0.9791 | 0.9787 | 0.9788 |
|  1024 |  0.05 |    50 |     0.95 |       25 | 00:11 | **94.49%** |    0.9484 | 0.9465 | 0.9451 |
|  2048 |  0.02 |   200 |     0.85 |       25 | 01:07 | **99.40%** |    0.9940 | 0.9939 | 0.9940 |
|   256 |  0.01 |   100 |     0.85 |       10 | 00:08 | **96.28%** |    0.9628 | 0.9626 | 0.9626 |
|   512 |  0.01 |    50 |      0.9 |       25 | 00:07 | **92.78%** |    0.9306 | 0.9273 | 0.9272 |

---

### Architecture avec downsampling par stride et gros classifieur

| Type de couche      | Détails                                                        |
| ------------------- | -------------------------------------------------------------- |
| Entrée              | Image 1 canal (28×28 pixels)                                   |
| Convolution 1       | 32 filtres, 3×3, padding=1                                     |
| Activation          | SiLU                                                           |
| Normalisation       | BatchNorm (32 canaux)                                          |
| Convolution 2       | 32 filtres, 3×3, padding=1                                     |
| Activation          | SiLU                                                           |
| Normalisation       | BatchNorm (32 canaux)                                          |
| Convolution 3       | 64 filtres, 3×3, stride=2, padding=1 _(downsampling spatial)_  |
| Activation          | SiLU                                                           |
| Normalisation       | BatchNorm (64 canaux)                                          |
| Convolution 4       | 64 filtres, 3×3, padding=1                                     |
| Activation          | SiLU                                                           |
| Normalisation       | BatchNorm (64 canaux)                                          |
| Convolution 5       | 128 filtres, 3×3, stride=2, padding=1 _(downsampling spatial)_ |
| Activation          | SiLU                                                           |
| Normalisation       | BatchNorm (128 canaux)                                         |
| Aplatissement       | Flatten 3D → vecteur                                           |
| Couche dense cachée | (128 × 7 × 7) → 256                                            |
| Activation          | SiLU                                                           |
| Dropout             | 50%                                                            |
| Couche dense finale | 256 → 10 neuronnes (classification)                            |

### Résultats

| Batch |   LR | Steps | LR Decay | Patience | Time  |   Accuracy | Precision | Recall |     F1 |
| ----: | ---: | ----: | -------: | -------: | :---: | ---------: | --------: | -----: | -----: |
|   512 | 0.01 |   200 |     0.95 |       10 | 00:32 | **99.29%** |    0.9928 | 0.9929 | 0.9929 |
|   256 | 0.01 |   500 |      0.9 |       10 | 00:52 | **99.37%** |    0.9936 | 0.9937 | 0.9936 |
|   512 | 0.01 |    50 |      0.9 |       10 | 00:11 | **97.27%** |    0.9727 | 0.9726 | 0.9726 |
|   512 | 0.01 |   100 |      0.9 |       25 | 00:17 | **98.76%** |    0.9876 | 0.9875 | 0.9875 |
|   256 | 0.05 |   500 |      0.9 |       50 | 00:51 | **99.18%** |    0.9917 | 0.9918 | 0.9918 |

---

### Architecture légère et rapide

| Type de couche      | Détails                                   |
| ------------------- | ----------------------------------------- |
| Entrée              | Image 1 canal (28×28 pixels)              |
| Convolution 1       | 16 filtres, noyau 3×3                     |
| Activation          | SiLU                                      |
| Pooling             | Max-pooling (2×2)                         |
| Convolution 2       | 32 filtres, noyau 3×3                     |
| Activation          | SiLU                                      |
| Pooling             | Max-pooling (2×2)                         |
| Aplatissement       | Conversion en vecteur 1D (≈ 800 neurones) |
| Couche dense finale | 800 → 10 neurones (classification)        |

### Résultats

| Batch |    LR | Steps | LR Decay | Patience | Time  |   Accuracy | Precision | Recall |     F1 |
| ----: | ----: | ----: | -------: | -------: | :---: | ---------: | --------: | -----: | -----: |
|   512 |  0.01 |   100 |      0.9 |       25 | 00:03 | **92.68%** |    0.9278 | 0.9265 | 0.9265 |
|   256 | 0.005 |   500 |     0.95 |       25 | 00:05 | **97.66%** |    0.9764 | 0.9764 | 0.9764 |
|   256 |  0.01 |    50 |      0.9 |       50 | 00:01 | **82.40%** |    0.8395 | 0.8203 | 0.8216 |
|  1024 |  0.02 |   150 |     0.95 |       50 | 00:03 | **97.55%** |    0.9756 | 0.9754 | 0.9753 |
|   256 |  0.05 |   150 |      0.9 |       50 | 00:02 | **96.97%** |    0.9706 | 0.9693 | 0.9697 |
|   512 |  0.01 |   100 |     0.85 |       50 | 00:02 | **92.36%** |    0.9253 | 0.9233 | 0.9234 |
|  2048 |  0.02 |   200 |     0.85 |       25 | 00:06 | **98.54%** |    0.9855 | 0.9854 | 0.9854 |
|   256 |  0.01 |    50 |     0.95 |       10 | 00:01 | **79.82%** |    0.8033 | 0.7950 | 0.7954 |
|   256 | 0.005 |   500 |     0.85 |       50 | 00:04 | **97.59%** |    0.9757 | 0.9758 | 0.9757 |
|   256 |  0.01 |   150 |      0.9 |       25 | 00:02 | **95.82%** |    0.9580 | 0.9582 | 0.9579 |
|  1024 |  0.01 |   100 |      0.9 |       50 | 00:02 | **93.43%** |    0.9345 | 0.9342 | 0.9341 |
|   256 | 0.005 |   500 |      0.9 |       50 | 00:04 | **97.78%** |    0.9777 | 0.9777 | 0.9777 |
|  1024 |  0.01 |   150 |      0.9 |       50 | 00:03 | **95.99%** |    0.9598 | 0.9596 | 0.9596 |
|  1024 |  0.02 |   150 |     0.85 |       25 | 00:03 | **97.61%** |    0.9761 | 0.9759 | 0.9760 |
|   512 |  0.01 |   150 |     0.95 |       10 | 00:02 | **94.39%** |    0.9439 | 0.9437 | 0.9435 |
