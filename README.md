# Classification of Defective Photovoltaic Cells in Electroluminescence Imagery

This project utilizes various machine learning techniques in conjunction with the ELPV dataset to classify solar cell images based on their defectiveness levels. The goal is to develop and test computer vision methods for predicting the health of PV cells in EL images of solar modules.

> **Contributors**: Minhua Tang, Kai Chi Mok, Fei Hu, Jiale Wang, Xieyang Jiang

## Getting Started

The project is implemented in Jupyter Notebook. Ensure all dependencies are installed before running the program.

To load the dataset:
1. Download the dataset manually from the provided repository.
2. Save the dataset in the root directory of the Jupyter environment.
3. Use the `load_dataset` function to load the data.

## ELPV Dataset 

The [ELPV dataset](https://github.com/zae-bayern/elpv-dataset) is used as the training reference for this model. It contains 2,624 EL images of both functional and defective PV cells, with varying degrees of degradation extracted from 44 different solar modules.

All cell images are normalized for size and perspective, corrected for camera lens distortions, and manually annotated with the type of solar module and defectiveness level.

**Solar Module Type**:
- Monocrystalline
- Polycrystalline

**Defectiveness Level**:
- Fully functional (0% probability)
- Possibly defective (33% probability)
- Likely defective (67% probability)
- Certainly defective (100% probability)

## Methodologies

Our experiments are divided into two major pipelines, each employing different classification models with or without pre-processing techniques to investigate performance gains.

The two pipelines are:
- **Classification Using General-Purpose Classifiers**
- **Classification Using Deep Convolutional Networks**

For the classifier pipeline, we employed:
- Stochastic Gradient Descent (SGD)
- Decision Tree (DT)
- K-Nearest Neighbours (KNN)
- Gaussian Naïve Bayes (GNB)
- Multinomial Naïve Bayes (MNB)
- Bernoulli Naïve Bayes (BNB)
- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM)

For the ConvNet pipeline, we employed:
- Modified VGG16 with ImageNet Transfer Learning
- Simplified VGG (HC6) with Randomly Initialized Weights

For Pre-Processing, we employed:
- Classing (4-Class vs. 8-Class)
- Dataset Partitioning (75/25 split)
- Oversampling (SMOTE, SVM-SMOTE, Borderline-SMOTE)
- Class-Specific Thresholding (TOZERO, BINARY, OTSU)
- Data Augmentation (Flip, Rotation, Translation)
- Sharpening (Laplacian Image Sharpening)

Both setups used the same dataset split across all iterations to ensure fairness in evaluation. Test set predictions were evaluated based on accuracy, precision, recall, F1-scores, and 4x4 confusion matrices, first with the two solar module types combined and then separated.

## Findings Summary

- SVM was the best-performing classifier, achieving up to 70% accuracy without oversampling, thresholding, or edge detection techniques.
- VGG16 without any pre-processing stood out among ConvNets, reaching an accuracy as high as 76% in defect categorization.
- VGG16 outperforms SVM by 6%.
- Both models performed strongly for monocrystalline modules,  with accuracies over 70%.
- Only VGG16 sustained an accuracy of over 70% on polycrystalline cells, outshining SVM by a margin of 6%.
- Thresholding, oversampling, and edge detection slightly improved precision and recall for minority classes, particularly for defect categories 0.33 and 0.67, but did not significantly enhance overall accuracy,
- VGG16 offers slightly better performance, but demands higher computational resources.
- SVM is more suitable for training under small footprint constraints.
- HC6 exhibited similar performance traits to VGG16, making it another practical option for quick assessment.

## Other Remarks

This project was completed to fulfill the requirements of COMP9517 Computer Vision at UNSW.
