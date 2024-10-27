# Used-Car-Price-Prediction-Competition-
This project involves predicting used car prices using an ensemble model that achieved an R² accuracy of 0.965 and secured 2nd place in a class competition. Utilizing Random Forest, XGBoost, and Gradient Boosting, the model combines cross-validation and regularization to enhance predictive accuracy and minimize overfitting.

## **Project Overview**

This project showcases a machine learning model designed to predict the prices of used cars. Created as part of a prediction competition for the *ECON 424: Machine Learning for Economists* course, this project achieved **second place**, attaining an R² accuracy of **0.965**. The model uses a blend of ensemble techniques—*Random Forest*, *XGBoost*, and *Gradient Boosting*—to enhance predictive accuracy and model robustness.

## **Dataset**

The dataset used in this project consists of historical used car prices with multiple features provided through a class competition, initially sourced from Kaggle. The project includes the following components:

1. **CSV File**: Contains predictions on the test set.
2. **Python Script**: Implements the ensemble modeling approach.
3. **PDF Report**: Provides insights into model performance and accuracy distribution.

The dataset comprises **100,000 observations** in the test set, with some missing values across feature variables. The model is structured to handle these cases to ensure a complete set of predictions.

## **Model Development**

The model combines three ensemble algorithms:
- **Random Forest**
- **XGBoost**
- **Gradient Boosting**

These models were chosen for their ability to handle complex relationships within data while minimizing overfitting. Techniques such as **cross-validation** and **regularization** were used to further enhance performance and stability.

### **Key Results**
- **Prediction Accuracy (R²)**: 0.965
- **Rank**: 2nd in class competition

## **Usage**

To replicate the results or test the model on your data:

1. Download the repository.
2. Run the provided `.py` script to train the model.
3. Use the CSV output as your prediction file for further analysis.

---

Feel free to explore the code, try out different parameters, or reach out with any questions. Your feedback is valuable for improving this project!
