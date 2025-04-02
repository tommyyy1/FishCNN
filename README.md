# FishCNN

# Fish Image Classification with Deep Learning

Problem Statement

This project focuses on classifying fish images into multiple categories using deep learning models. The task involves training a CNN from scratch and leveraging transfer learning with pre-trained models to enhance performance. The project also includes saving models for later use and deploying a Streamlit application to predict fish categories from user-uploaded images.

🎯 Business Use Cases

Enhanced Accuracy: Determine the best model architecture for fish image classification.

Deployment Ready: Create a user-friendly web application for real-time predictions.

Model Comparison: Evaluate and compare metrics across models to select the most suitable approach.

🔬 Approach

1️⃣ Data Preprocessing and Augmentation
Rescale images to the [0,1] range.
Apply data augmentation techniques like rotation, zoom, and flipping to enhance model robustness.

2️⃣ Model Training

Train a CNN model from scratch.
Experiment with five pre-trained models:

VGG16
ResNet50
MobileNet
InceptionV3
EfficientNetB0

Fine-tune the pre-trained models on the fish dataset.
Save the trained model (best accuracy) in .h5 or .pkl format for future use.

3️⃣ Model Evaluation

Compare metrics such as accuracy, precision, recall, F1-score, and confusion matrix across all models.
Visualize training history (accuracy and loss) for each model.

4️⃣ Deployment

Build a Streamlit application to:
Allow users to upload fish images.
Predict and display the fish category.
Provide model confidence scores.

🚀 Installation & Usage

1️⃣ Clone the Repository
git clone https://github.com/your-username/fish-classification.git
cd fish-classification

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train.py

4️⃣ Run the Streamlit App
streamlit run app.py


📊 Results & Findings

The best model achieved an accuracy of 98.74% (MobileNet) on the test dataset.
Transfer learning models outperformed the CNN trained from scratch.
The final model was deployed as a Streamlit web app for real-time predictions.
