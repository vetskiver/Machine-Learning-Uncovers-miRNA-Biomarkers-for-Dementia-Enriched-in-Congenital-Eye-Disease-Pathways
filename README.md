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

- **Files**:
  - `Zero_R_(Zero_Rule)_Algorithm.ipynb`: Implementation of the Zero-R algorithm, used as a baseline classifier for comparison purposes.

### 2. `data_extraction_and_processing`

- **Purpose**: Contains the main scripts used to preprocess the data and create the initial datasets for analysis.
- **Files**:
  - `clinical_factors_processing_and_combination.ipynb`: Script for processing and combining clinical factors with the miRNA data.
  - `data_preprocessing_steps_for_clinical_analysis.ipynb`: Additional preprocessing steps to clean and format the data for analysis.

### 3. `deep_learning`

- **Purpose**: Contains the deep learning algorithms used for each dataset condition.
- **Files**:
  - `AC_vs_NC_fastai.ipynb`: Deep learning model for distinguishing Alzheimer's Control (AC) from Normal Control (NC).
  - `DLB_vs_NC_fastai.ipynb`: Deep learning model for distinguishing Dementia with Lewy Bodies (DLB) from Normal Control (NC).
  - `MCI_vs_NC_fastai.ipynb`: Deep learning model for distinguishing Mild Cognitive Impairment (MCI) from Normal Control (NC).
  - `NPH_vs_NC_fastai.ipynb`: Deep learning model for distinguishing Normal Pressure Hydrocephalus (NPH) from Normal Control (NC).
  - `VAD_vs_NC_fastai.ipynb`: Deep learning model for distinguishing Vascular Dementia (VAD) from Normal Control (NC).

### 4. `feature_selection`

- **Purpose**: Contains the feature selection methods used to identify significant miRNA features.
- **Files**:
  - `ANOVA_assumptions.ipynb`: Assumptions and implementation of the ANOVA method.
  - `bootstrapping_analysis_b1_p1_final.ipynb`: Bootstrapping analysis for branch 1, part 1.
  - `bootstrapping_analysis_b1_p2_final.ipynb`: Bootstrapping analysis for branch 1, part 2.
  - `bootstrapping_analysis_b2_final.ipynb`: Bootstrapping analysis for branch 2.
  - `feature_selector_b1_p2.py`: Feature selection script for branch 1, part 2.
  - `feature_selector_b1.py`: Feature selection script for branch 1.
  - `feature_selector_b2.py`: Feature selection script for branch 2.

### 5. `ml_classifiers`

- **Purpose**: Contains the machine learning classifier algorithms for each dataset condition.
- **Files**:
  - `ML_classifiers_NC_vs_AD_parameter_tuning.ipynb`: Parameter tuning for Alzheimer's Disease (AD) vs. Normal Control (NC).
  - `ML_classifiers_NC_vs_DLB_parameter_tuning.ipynb`: Parameter tuning for Dementia with Lewy Bodies (DLB) vs. Normal Control (NC).
  - `ML_classifiers_NC_vs_MCI_parameter_tuning.ipynb`: Parameter tuning for Mild Cognitive Impairment (MCI) vs. Normal Control (NC).
  - `ML_classifiers_NC_vs_NPH_parameter_tuning.ipynb`: Parameter tuning for Normal Pressure Hydrocephalus (NPH) vs. Normal Control (NC).
  - `ML_classifiers_NC_vs_VaD_parameter_tuning.ipynb`: Parameter tuning for Vascular Dementia (VaD) vs. Normal Control (NC).

- **Explanation**: We utilized three different machine learning algorithms for classification:
  - **Logistic Regression**: A linear model for binary classification that estimates the probability of a binary outcome.
  - **Support Vector Machine (SVM)**: A non-linear classifier that finds the hyperplane that best separates different classes in the feature space.
  - **XGBoost**: An ensemble method based on gradient boosting, known for its high performance in structured data tasks.

### 6. `visualizations`

- **Purpose**: Contains the scripts for creating visualizations to compare the accuracy of different models and feature sets.
- **Files**:
  - `create_heatmaps.ipynb`: Script for generating heatmaps of model performance.

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

For any questions or issues, please contact the repository owner.

---

This README provides an overview of the repository structure and the general steps taken to analyze the miRNA expression data for identifying biomarkers associated with dementia. Each directory contains scripts specific to different stages of the analysis, from data extraction and preprocessing to feature selection, model training, and result visualization.

