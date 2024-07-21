# Machine Learning Uncovers miRNA Biomarkers for Dementia Enriched in Eye Disease Pathways

## Overview

This repository contains the code and data used for identifying miRNA biomarkers for dementia. The datasets used for the analysis are publicly available from the NCBI Gene Expression Omnibus (GEO).

### Datasets

The datasets used in this study were obtained from the following GEO accession numbers:

- **[GSE120584](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120584)**: This dataset contains miRNA expression profiles.
- **[GSE167559](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167559)**: This dataset includes additional miRNA expression data for further analysis.

## Repository Structure

The repository is organized into several directories, each containing scripts and data for different stages of the analysis:

### 1. `Zero-R-Algorithm`

- This directory contains the implementation of the Zero-R algorithm, which is used as a baseline classifier for comparison purposes.

### 2. `data_extraction_and_processing`

- **Purpose**: Contains the main scripts used to preprocess the data and create the initial datasets for analysis.
- **Files**:
  - `data_extraction.py`: Script for extracting and preprocessing data from the GEO datasets.
  - `data_processing.py`: Additional processing steps to clean and format the data for analysis.

### 3. `deep_learning`

- **Purpose**: Contains the deep learning algorithms used for each dataset condition.
- **Files**:
  - `deep_learning_model.py`: Implementation of deep learning models for miRNA classification.
  - `model_training.py`: Script for training the deep learning models on the processed datasets.

### 4. `feature_selection`

- **Purpose**: Contains the feature selection methods used to identify significant miRNA features.
- **Files**:
  - `welch_anova.py`: Script for performing Welch ANOVA to identify significant features.
  - `relieff.py`: Script for applying the ReliefF algorithm for feature selection.
  - `combine_features.py`: Script for combining features selected by different methods.

### 5. `ml_classifiers`

- **Purpose**: Contains the machine learning classifier algorithms for each dataset condition.
- **Files**:
  - `svm_classifier.py`: Support Vector Machine classifier implementation.
  - `rf_classifier.py`: Random Forest classifier implementation.
  - `mlp_classifier.py`: Multilayer Perceptron classifier implementation.
  - `classifier_training.py`: Script for training and evaluating the classifiers.

### 6. `visualizations`

- **Purpose**: Contains the scripts for creating visualizations to compare the accuracy of different models and feature sets.
- **Files**:
  - `heatmap.py`: Script for generating heatmaps of model performance.
  - `roc_curve.py`: Script for plotting ROC curves for different classifiers.

## Steps for Analysis

1. **Data Extraction and Processing**:
   - Use the scripts in the `data_extraction_and_processing` directory to extract and preprocess the miRNA expression data from the GEO datasets.

2. **Feature Selection**:
   - Apply Welch ANOVA and ReliefF algorithms using the scripts in the `feature_selection` directory to identify significant miRNA features.
   - Combine the selected features from different methods.

3. **Model Training**:
   - Train deep learning models using the scripts in the `deep_learning` directory.
   - Train machine learning classifiers using the scripts in the `ml_classifiers` directory.

4. **Evaluation and Visualization**:
   - Evaluate the performance of the models and visualize the results using the scripts in the `visualizations` directory.

## Contact

For any questions or issues, please contact the repository owners

---

This README provides an overview of the repository structure and the general steps taken to analyze the miRNA expression data for identifying biomarkers associated with dementia. Each directory contains scripts specific to different stages of the analysis, from data extraction and preprocessing to feature selection, model training, and result visualization.
