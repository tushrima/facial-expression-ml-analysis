# Facial Expression and Multimodal Classification — CMPT 419

A collection of Python notebooks exploring facial feature analysis, emotion recognition, and classification using face, body, and audio features.

## Notebooks

### `A2419.ipynb` — Facial feature analysis and clustering

Explores dimensionality reduction and unsupervised clustering of facial features. The notebook loads `./faces/faces_AU.csv`, applies principal component analysis (PCA), uses K-means clustering, compares silhouette scores across cluster counts, and explores UMAP-based visualization. Some cells also use images from the `./faces` directory.

### `A3419.ipynb` — Facial emotion classification

Experiments with classification using support vector machines (SVMs) and convolutional neural networks (CNNs). The notebook references the FER2013 dataset (`fer2013.csv`) and a separate facial action unit dataset (`phoebe_AU.csv`). It includes classification metrics and code for processing additional facial images.

### `Fear_Classification.ipynb` — Multimodal classification

Explores classification using audio, body annotation, and facial feature datasets. The notebook includes separate sections for each modality, analysis of classification accuracy, and a workflow for classifying new processed inputs. It expects several local CSV files, including `audio_features.csv`, `Body_Annotations.csv`, and `Face_features.csv`.

## Environment

The notebooks use Python and libraries including:

- NumPy, pandas, and Matplotlib
- scikit-learn
- UMAP (`umap-learn`)
- TensorFlow/Keras for the CNN experiments

A starting point for installing common dependencies is:

```bash
python -m pip install numpy pandas matplotlib scikit-learn umap-learn tensorflow jupyter
```

This is not a tested or pinned environment. Individual notebooks may require additional packages or adjustments for current library versions.

## Running the notebooks

1. Obtain the datasets referenced by the notebook you want to explore. **The repository contains the notebooks but not the required CSV datasets or image directories.**
2. Update local paths inside the notebook to point to those datasets. Some cells use Google Colab-specific paths.
3. Start Jupyter from the repository directory:

   ```bash
   jupyter notebook
   ```

4. Open the desired notebook and run its cells in order, checking any notebook-specific preprocessing steps.

The notebooks include experimental cells, saved outputs, and environment-specific assumptions; they have not been verified as reproducible end-to-end from a fresh clone.

## Data and interpretation notes

- The FER2013 dataset is referenced in `A3419.ipynb`; see the dataset citation in that notebook for its source.
- Facial action unit, audio, and body annotation files are external inputs and are not redistributed here.
- Classification results depend on the training data, preprocessing choices, and evaluation setup. These experiments should not be interpreted as reliable inference of a person's internal emotional state in real-world settings.

## Scope

This repository preserves exploratory coursework notebooks. It is not a packaged application or production-ready emotion recognition system.
