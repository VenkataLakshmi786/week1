To develop a CNN model to classify images of plastic waste

Guideline:

Overview

Dataset

Weekly progess

Technologies used

📖 Overview PlasticWasteClassifier is a Convolutional Neural Network (CNN)-based project designed to classify images of plastic waste into predefined categories. The goal is to assist in waste management and recycling by automatically identifying plastic types from images.

Key Objectives: Develop an efficient deep learning model to classify plastic waste. Contribute to environmental sustainability by facilitating plastic recycling.

📊 Dataset The dataset used for this project consists of images of plastic waste, categorized into different types for classification. The dataset is curated to train and evaluate the Convolutional Neural Network (CNN) model effectively.

Dataset Details:

📂Source: download the dataset from here https://www.kaggle.com/datasets/techsash/waste-classification-data/data

📅 Weekly Progress Week 1: Data Import, Setup, and Learning the Basics of CNN This week, we focused on laying the groundwork for the project and building a strong understanding of CNNs. Activities:

Imported the required libraries and frameworks.
Set up the project environment.
Explored the dataset structure.

Week1- Wasteclassification.ipynb

Kaggle Notebook

🌐Technologies Used

Python

TensorFlow/Keras

OpenCV

NumPy

Pandas

Matplotlib# week1
Waste management using CNN model

Week 2: Model Training, Evaluation, and Predictions
Date: 28th January 2025 - 31st January 2025

Activities:

Trained the CNN model on the dataset.
Optimized hyperparameters to improve accuracy.
Evaluated model performance using accuracy and loss metrics.
Performed predictions on test images.
Visualized classification results with a confusion matrix.
Notebooks:

Week2-Model-Training-Evaluation-Predictions.ipynb


Kaggle Notebook
📌 Conclusion & Summary of Model Performance


1️⃣ Overview of the Model
The trained Convolutional Neural Network (CNN) model was designed to classify waste into two categories:

O (Organic Waste)
R (Recyclable Waste)
It was trained on a dataset of training images using convolutional layers, max-pooling, batch normalization, and fully connected dense layers. The model was optimized using categorical cross-entropy loss and evaluated based on accuracy.

2️⃣ Model Evaluation on Test Data
After training, the model was evaluated on a separate test dataset. The results are as follows:

✅ Test Accuracy: 85.32%
✅ Test Loss: 0.3997

🔹 This means the model correctly classifies waste in 85 out of 100 cases on unseen data. The low test loss indicates that the model has learned meaningful patterns and is not overfitting significantly.

3️⃣ Predictions and Sample Results
The model made predictions on test images, converting probability outputs into class labels. Here’s a sample of predicted vs. actual results:

Predicted Classes: ['O', 'O', 'O', 'O', 'R', 'O', 'O', 'O', 'O', 'O']
Actual Classes: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
🔹 The model mostly predicted correctly, but one case (index 5) was classified as "R" instead of "O", indicating a potential misclassification.

4️⃣ Classification Report Analysis
The classification report provides deeper insights into model performance for each class:

Class	Precision	Recall	F1-Score	Support
O (Organic)	0.89	0.84	0.86	1401
R (Recyclable)	0.81	0.87	0.84	1112
Overall Accuracy	85%			
Macro Avg (Balanced Score for Classes)	85%			
Weighted Avg (Adjusted for Imbalance)	85%			
📌 Key Observations:
🔹 Precision for Organic (O) is higher: The model is more confident when predicting Organic waste than Recyclable waste.
🔹 Recall for Recyclable (R) is higher: The model captures more actual Recyclable waste but sometimes mislabels Organic waste as Recyclable.
🔹 Balanced F1-score: The model performs well for both classes, with a small bias towards predicting Organic waste correctly.

5️⃣ Confusion Matrix Insights
The confusion matrix helps visualize the model’s errors:

1401 Organic waste samples:

1180 were correctly classified as Organic (True Positives)
221 were misclassified as Recyclable (False Negatives)
1112 Recyclable waste samples:

968 were correctly classified as Recyclable (True Positives)
144 were misclassified as Organic (False Positives)
📌 Key Takeaways from Confusion Matrix:
✅ The model performs well overall but struggles slightly more with distinguishing Recyclable waste from Organic waste.
✅ 221 Organic samples were wrongly classified as Recyclable waste, which might be due to overlapping features (e.g., food-contaminated paper/cardboard).

🚀 Final Conclusion:
The CNN model has achieved 85.32% accuracy, which is a strong performance for waste classification.
The model performs slightly better for Organic waste but can sometimes misclassify Recyclable waste.
Possible Improvements:
Adding more diverse training data to reduce misclassification.
Applying data augmentation to expose the model to more variations.
Experimenting with fine-tuning on pre-trained CNN models (like ResNet, VGG16) to further boost accuracy.
Tweaking hyperparameters (learning rate, dropout rate) to optimize performance.
✅ Overall, the model is highly effective and can be used for real-world waste classification tasks with further refinements! 🔥♻️
