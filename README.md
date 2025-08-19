# ML_Project2-KNN-
Dataset link : https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset
Heart Disease Prediction using K-Nearest Neighbors (KNN)
This project implements a K-Nearest Neighbors (KNN) classifier to predict the presence of heart disease in a patient based on various medical attributes. The focus is on a thorough data preprocessing pipeline and meticulously tuning the model to achieve the best possible accuracy.

Project Overview
The goal is to build a predictive model that can assist in the early diagnosis of heart disease. We use a classic machine learning approach, K-Nearest Neighbors, and demonstrate how strategic data cleaning and hyperparameter tuning can significantly improve the model's performance.

Key Achievements:

Initial Accuracy: 78% with basic KNN.

Final Accuracy: 85% after feature selection and hyperparameter tuning.

Methodology
1. Data Preparation & Exploration
Data Loading & Inspection: The dataset was loaded and immediately inspected for quality.

Handling Data Issues: Checked for and handled missing values, incorrect data types, and duplicate entries to ensure a clean dataset.

Correlation Analysis: Plotted a heatmap to visualize the correlation of all features with the target variable. This step was crucial for identifying the most influential factors for prediction.

2. Initial Modeling
The data was split into features (X) and the target variable (y).

An 80:20 train-test split was applied to create validation sets.

Used StandardScaler to normalize the feature values. This is critical for distance-based algorithms like KNN to ensure no single feature dominates the distance calculation.

An initial KNN model was trained, yielding a baseline accuracy of 78%.

3. Finding the Optimal K
Plotted a graph of model accuracy against different values of K (number of neighbors).

Identified that K=8 provided the best performance at 81% accuracy. However, an even K value can sometimes lead to ties in voting; therefore, K=7 was selected as a more robust choice to avoid potential tie-breaker scenarios.

4. Feature Selection for Improvement
To improve the model further, features with very low correlation to the target variable were dropped from the dataset.

This process of feature selection simplifies the model and can reduce noise, leading to better performance.

5. Final Model & Hyperparameter Tuning
The model was retrained on the refined dataset with the most relevant features.

The default K=5 now yielded an improved accuracy of 83%.

A final round of hyperparameter tuning was conducted, testing different K values and distance metrics (p).

The best result was achieved with K=7 and using the Manhattan distance (p=1), pushing the final accuracy to an impressive 85%.

Results
Model Version	Key Parameters	Accuracy
Baseline KNN	Default K=5, All Features	78%
Tuned KNN	K=7, All Features	81%
Tuned KNN + Feature Selection	K=5 (default), Selected Features	83%
Final Model	K=7, p=1 (Manhattan), Selected Features	85%
Key Takeaways
Data Preprocessing is Crucial: Properly handling data quality and scaling features had an immediate positive impact.

Feature Selection Matters: Removing irrelevant features not only made the model simpler but also more accurate.

Hyperparameter Tuning is Key: The choice of K and the distance metric (p) are not just minor details; they are fundamental to the model's performance. systematically finding the right combination led to a significant 7% boost from the initial baseline.

Technologies Used
Python

Libraries:

pandas, numpy (Data manipulation)

matplotlib, seaborn (Data visualization & heatmaps)

scikit-learn (train_test_split, StandardScaler, KNeighborsClassifier, metrics)
