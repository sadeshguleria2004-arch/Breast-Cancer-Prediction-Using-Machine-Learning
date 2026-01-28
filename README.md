🩺 Breast Cancer Prediction Using Machine Learning

Breast cancer is one of the most common and life-threatening diseases worldwide, and early diagnosis plays a vital role in improving patient survival rates. This project focuses on predicting whether a breast tumor is Malignant or Benign using machine learning classification techniques based on medical diagnostic features.

The objective of this project is to build and compare multiple machine learning models that can accurately classify breast cancer cases and demonstrate the complete machine learning workflow, including data preprocessing, training, evaluation, and model comparison.

The dataset used in this project is the Breast Cancer Wisconsin Dataset, sourced from Scikit-learn. It contains 30 numerical medical features extracted from breast mass images. The target variable consists of two classes: 0 representing Malignant tumors and 1 representing Benign tumors. The features include measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry, along with their mean, worst, and standard error values.

This project is implemented using Python and popular data science libraries including NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. The code is written and executed in a Jupyter Notebook, making it compatible with both local environments and Google Colab.

The machine learning workflow followed in this project includes loading and exploring the dataset, preprocessing the data, scaling features, splitting the data into training and testing sets, training multiple machine learning models, evaluating their performance using relevant metrics, and comparing results to identify reliable models for medical diagnosis.

Four machine learning classification models are implemented and evaluated in this project: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and Gradient Boosting Classifier. Each model is trained on the same dataset to ensure fair comparison.

Model performance is evaluated using medically relevant and statistical metrics such as Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC Score, and Matthews Correlation Coefficient (MCC). These metrics are especially important in healthcare applications, where minimizing false negatives is critical.

The results show that all implemented models perform well on the dataset, with ensemble models demonstrating strong and reliable predictive performance. This project highlights how machine learning can effectively assist in medical decision-making and early disease detection.

The project repository consists of a Jupyter Notebook file (TASK_2_.ipynb) containing the complete implementation, along with a README file and optional requirements file.

To run the project, clone the repository, open the Jupyter Notebook, and execute all cells, or upload the notebook directly to Google Colab and run it there.

Future improvements for this project include hyperparameter tuning, feature selection techniques, experimenting with deep learning models, and deploying the model using web frameworks such as Flask or Streamlit.
