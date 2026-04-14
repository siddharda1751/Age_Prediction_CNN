# 🦴 Bone Age Prediction using CNN

A deep learning pipeline for pediatric bone age prediction from X-ray images using CNN (ResNet) regression, classification, and XGBoost ensemble methods with comprehensive evaluation and visualization.

## ✨ Features

- **📈 Regression Model** - ResNet-based CNN for continuous age prediction (in years/months)
- **📊 Classification Model** - Age group classification with ordinal awareness
- **🌳 XGBoost Ensemble** - Combined CNN features with gradient boosting for improved accuracy
- **🔥 Grad-CAM Visualizations** - Heatmaps showing which bone regions influence predictions
- **📉 t-SNE Analysis** - Visualization of learned feature representations
- **⚖️ Gender Bias Analysis** - Fairness evaluation across male/female samples
- **💻 M4 Mac Optimized** - Full Metal Performance Shaders (MPS) support

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch
- **CNN Backbone**: ResNet (pretrained on ImageNet)
- **Boosting**: XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Analysis**: scikit-learn, NumPy, Pandas

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python main.py
```

The pipeline will guide you through:
1. Data preparation and splitting
2. CNN regression training
3. Model evaluation with visualizations
4. Optional classification training
5. Optional XGBoost ensemble

## 📊 Model Performance

### Regression (Bone Age Prediction)
- **MAE**: ~0.5 years
- **RMSE**: ~0.7 years  
- **R²**: ~0.95

### Classification (Age Groups)
- **Accuracy**: 85%+
- **Quadratic Weighted Kappa**: 0.90+

## 📁 Project Structure

```
├── main.py                 # Main execution pipeline
├── src/
│   ├── config.py          # Configuration settings
│   ├── data_preprocessing.py
│   ├── dataset.py         # PyTorch Dataset
│   ├── models.py          # CNN architectures
│   ├── train.py           # Training loops
│   ├── evaluate.py        # Evaluation & visualization
│   └── xgboost_trainer.py # XGBoost ensemble
├── models/                 # Saved model weights
├── results/
│   ├── plots/             # Visualizations
│   └── metrics/           # Evaluation reports
├── notebooks/             # Jupyter notebooks for EDA
└── data/                  # Data splits
```

## 📈 Generated Visualizations

1. **Scatter Plot** - Predicted vs actual age
2. **Confusion Matrix** - For classification
3. **Grad-CAM Heatmaps** - Model attention on X-rays
4. **t-SNE Plots** - Feature space visualization
5. **Error Distribution** - Prediction error analysis
6. **Training Curves** - Loss and metrics over epochs

## ⚠️ Note

Model checkpoint files (`.pth`) are not included in this repository due to size constraints. To use the trained models:
1. Run the training pipeline, or
2. Download pretrained weights from [link to be added]

## 📄 License

MIT License
