# Soil Image Classification Challenge Part-1

A machine learning competition organized by Annam.ai at IIT Ropar for classifying soil images into four distinct categories using deep learning techniques.

## 📋 Challenge Overview

**Task**: Classify soil images into one of four categories:
- Alluvial soil
- Black soil  
- Clay soil
- Red soil

**Deadline**: May 25, 2025, 11:59 PM IST

## 🗂️ Project Structure

```
Challenge 1/
├── Data/
│   └── download.sh
├── Docs/
│   ├── Architecture.png
│   └── ml-metrics.json
├── Notebooks/
│   └── training.ipynb
└── src/
    └── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

Ensure you have access to:
- Kaggle account with notebook environment
- GPU runtime (recommended for faster training)
- Dataset: `soil_classification-2025`

### Required Libraries

The notebook uses the following key libraries:
- `torch` - PyTorch for deep learning
- `timm` - Pre-trained vision models
- `torchvision` - Image transformations
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `matplotlib` & `seaborn` - Visualization
- `PIL` - Image processing
- `tqdm` - Progress bars

## 📊 Dataset Structure

The dataset should be organized as follows in Kaggle:
```
/kaggle/input/soil-classification/soil_classification-2025/
├── train/           # Training images
├── test/            # Test images  
├── train_labels.csv # Training labels
├── test_ids.csv     # Test image IDs
└── sample_submission.csv # Submission format
```

## 🔧 How to Run

### Step 1: Setup Environment
1. Open Kaggle and create a new notebook
2. Add the soil classification dataset to your notebook
3. Ensure GPU accelerator is enabled (Settings → Accelerator → GPU)

### Step 2: Run the Notebook
1. Copy the provided code into your Kaggle notebook cells
2. Execute cells sequentially from top to bottom
3. The notebook will automatically:
   - Load and explore the dataset
   - Create train/validation splits
   - Build and train an EfficientNet-B0 model
   - Generate predictions on test data
   - Create submission file

### Step 3: Key Components

**Data Exploration**:
- Visualizes soil type distribution
- Shows sample images for each category
- Analyzes class balance

**Model Architecture**:
- Uses EfficientNet-B0 pre-trained on ImageNet
- Fine-tuned for 4-class soil classification
- Input size: 224x224 RGB images

**Training Configuration**:
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Epochs: 10 (adjustable)

**Data Augmentation**:
- Resize to 224x224
- Normalize with mean=0.5, std=0.5
- Additional augmentations can be added for improved performance

## 📈 Evaluation Metrics

The competition uses **Minimum F1-Score** as the primary evaluation metric:
- Calculates F1-score for each class
- Takes the minimum value across all classes
- Ensures balanced performance across soil types

Additional metrics tracked:
- Overall accuracy
- Per-class F1-scores
- Top-2 accuracy
- Confusion matrix

## 📁 Output Files

The notebook generates:
- `submission.csv` - Final predictions for test set
- Training/validation loss plots
- Confusion matrix visualization
- Classification report with per-class metrics

## 🔍 Troubleshooting

**Common Issues**:

1. **Memory Error**: Reduce batch size to 16 or 8
2. **CUDA Out of Memory**: Restart kernel and reduce batch size
3. **File Not Found**: Check dataset paths match your Kaggle input structure
4. **Poor Performance**: 
   - Increase training epochs
   - Add data augmentation
   - Try different learning rates

**Dataset Path Issues**:
If you encounter path errors, verify the dataset structure matches:
```python
BASE_PATH = "/kaggle/input/your-dataset-name/soil_classification-2025"
```

## 📋 Submission Guidelines

1. Run the complete notebook to generate `submission.csv`
2. Download the submission file
3. Upload to the competition submission page
4. Verify the format matches the sample submission

# Soil Image Classification Challenge - Part 2

A machine learning competition organized by Annam.ai at IIT Ropar for binary classification to determine whether images contain soil or not, serving as an initial task for shortlisted hackathon participants.

## 📋 Challenge Overview

**Task**: Binary classification to determine if images are soil images or not
- **Label 1**: Soil image
- **Label 0**: Non-soil image

**Deadline**: May 25, 2025, 11:59 PM IST

⚠️ **Important**: Submit well before the deadline to avoid server overload or last-minute issues. Late traffic may lead to upload failures.

## 🗂️ Project Structure

```
Challenge 2/
├── Data/
│   └── download.sh
├── Docs/
│   ├── Architecture.png
│   └── ml-metrics.json
├── Notebooks/
│   └── training.ipynb
├── src/
│   └── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Ensure you have access to:
- Kaggle account with notebook environment
- GPU runtime (recommended for faster feature extraction)
- Dataset: `soil-classification-part-2`

### Required Libraries

The notebook uses the following key libraries:
- `tensorflow` - Deep learning framework
- `keras` - High-level neural networks API
- `scikit-learn` - Machine learning utilities
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` & `seaborn` - Visualization
- `PIL` - Image processing
- `tqdm` - Progress bars

## 📊 Dataset Structure

The dataset should be organized as follows in Kaggle:
```
/kaggle/input/soil-classification-part-2/soil_competition-2025/
├── train/              # Training images
├── test/               # Test images  
├── train_labels.csv    # Training labels (0 or 1)
|── test_ids.csv        # Test image IDs
└── sample_submission.csv # Submission format
```

## 🔧 How to Run

### Step 1: Setup Environment
1. Open Kaggle and create a new notebook
2. Add the soil classification part 2 dataset to your notebook
3. Ensure GPU accelerator is enabled (Settings → Accelerator → GPU)

### Step 2: Run the Notebook
1. Copy the provided code into your Kaggle notebook cells
2. Execute cells sequentially from top to bottom
3. The notebook will automatically:
   - Load and explore the dataset
   - Extract features using EfficientNetB0
   - Perform similarity-based classification
   - Generate predictions on test data
   - Create submission file

### Step 3: Key Components

**Data Exploration**:
- Visualizes label distribution (soil vs non-soil)
- Shows sample images from both training and test sets
- Analyzes class balance

**Feature Extraction Approach**:
- Uses pre-trained EfficientNetB0 (ImageNet weights)
- Extracts deep features from global average pooling layer
- Normalizes feature vectors for similarity computation

**Classification Strategy**:
- **Similarity-based approach** using cosine similarity
- Compares test images with training images
- Uses threshold-based decision making (default: 0.8)
- Predicts based on maximum similarity to training samples

**Image Processing**:
- Input size: 224x224 RGB images
- EfficientNet preprocessing (ImageNet normalization)
- Feature vector normalization for robust similarity computation

## 📈 Evaluation Metrics

The competition uses **Minimum F1-Score** as the primary evaluation metric:
- Calculates F1-score for each class (0 and 1)
- Takes the minimum value across both classes
- Ensures balanced performance for both soil and non-soil images

Additional metrics tracked:
- Overall accuracy
- Per-class precision and recall
- Confusion matrix visualization


## 🔍 Algorithm Explanation

**Feature-Based Similarity Approach**:

1. **Feature Extraction**: 
   - Use pre-trained EfficientNetB0 to extract 1280-dimensional features
   - Apply global average pooling to get compact representations

2. **Similarity Computation**:
   - Calculate cosine similarity between test and training features
   - Find maximum similarity for each test image

3. **Classification Decision**:
   - If max_similarity > threshold → Soil image (1)
   - If max_similarity ≤ threshold → Non-soil image (0)

4. **Evaluation**:
   - Sanity check using nearest neighbor on training set
   - Generate confusion matrix and classification report

## 📁 Output Files

The notebook generates:
- `submission.csv` - Final binary predictions for test set
- Label distribution visualization
- Sample image displays
- Confusion matrix for training set evaluation
- Classification report with precision/recall metrics



## 🎯 Key Differences from Part 1

**Part 1 vs Part 2**:
- **Part 1**: Multi-class classification (4 soil types)
- **Part 2**: Binary classification (soil vs non-soil)
- **Part 1**: Uses PyTorch + fine-tuning
- **Part 2**: Uses TensorFlow + similarity matching
- **Part 1**: End-to-end training approach
- **Part 2**: Feature extraction + threshold-based classification

Good luck with the binary classification challenge! 🍀