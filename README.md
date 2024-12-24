# Predicting-In-Hospital-Mortality-Unveiling-Insights-with-Machine-Learning
This project addresses this issue by leveraging machine learning techniques to predict patient survival rates within hospital settings

DATA UNDERSTANDING

We extracted patient survival prediction data (‘[Accuracy 92.5% | Prediction of Patient Survival](https://www.kaggle.com/code/shahzaibmalik44/accuracy-92-34-prediction-of-patient-survival/input)’) from Kaggle to conduct our analysis.

The dataset comprises 85 variables, offering insights into factors typically recorded during a patient’s hospitalization. These factors play a role in determining whether the patient will survive.

1. Dependent Variable: Patient survival. The dependent variable, labeled 'hospital_death,' indicates the patient's survival outcome. A value of '0' indicates survival, while '1' indicates non-survival (death).
 
2. Independent Variables: 85 variables in total, including age, gender, ethnicity, height, BMI, whether or not the surgery was elective, diagnosis, weight, and many others.

  
a. It’s important to note that the variables containing ‘APACHE’ refer to the [Acute Physiology and Chronic Health Evaluation](https://reference.medscape.com/calculator/12/apache-ii), which is a severity-of-disease classification system commonly used in intensive care units to assess the severity of illness in critically ill patients. It provides a numerical score based on several physiological parameters including age and chronic health conditions.
