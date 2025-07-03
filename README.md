## 🌿 Plant Disease Prediction using CNN
A deep learning-based system for detecting plant leaf diseases using Convolutional Neural Networks (CNN). This project helps identify 38 different plant disease categories including healthy leaves, empowering early diagnosis and treatment in agriculture.

----
## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify leaf images into 38 categories (diseased/healthy). The user-friendly Streamlit web interface allows real-time image upload and disease prediction using a pretrained model.

## 📁 Project Structure
```
Plant-Disease-Prediction/
├── PlantDisease.ipynb   # Jupyter notebook for training and experimentation
├── run.py               # Python script for making predictions
└── README.md            # Project documentation
```
Download Model --> Download the pretrained model (model.h5) from the Releases section
[**Model**](https://github.com/Pranav-Uniyal/Plant-Disease-Prediction/releases/tag/CNN-model)

 ---
## Installation
1. Clone the repository
```
git clone https://github.com/Pranav-Uniyal/Plant-Disease-Prediction.git
cd Plant-Disease-Prediction
```
2. Install dependencies
   ```
   pip install streamlit tensorflow pillow
   ```
3. Run the Streamlit App
   ```
   streamlit run run.py
   ```
---
## Dataset
Source: **[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)**

Classes: 38 total (diseased and healthy plants)

Examples: Apple Scab, Tomato Yellow Leaf Curl Virus, Potato Late Blight, etc.

---
## Author
**Pranav Uniyal**


