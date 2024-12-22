# Resume Screening
The project leverages machine learning algorithms and TF-IDF vectorization to extract features from resumes and classify them into predefined categories. The system aims to reduce manual effort, save time, and improve recruitment efficiency.
#Features : 
1. Job Role Prediction: Classifies resumes into relevant job roles.
2. TF-IDF Feature Extraction: Processes textual data to extract meaningful features.
3. Random Forest Classifier: Provides the highest accuracy model for predictions.
4. Preprocessing Pipeline: Cleans and preprocesses resume text data.
# Technologies Used
-Python 3 <br>
-Natural Language Processing (NLP)<br>
-Scikit-learn for machine learning models<br>
-Pandas and NumPy for data manipulation<br>
-Joblib for model saving and loading<br>
-Jupyter Notebook/Google Colab for development and testing<br>
# Dataset
-The dataset consists of resumes labeled with job roles.<br>
-Each resume is represented as a text string.<br>
-The labels indicate the relevant job roles for classification.<br>
# Model Overview
-Data Preprocessing:<br>
Cleaned text by removing special characters, numbers, and stopwords.
Converted text to lowercase.
Tokenized and vectorized using TF-IDF.
-Model Training:<br>
Tested multiple algorithms (Logistic Regression, Naive Bayes, Random Forest).
Selected Random Forest Classifier for best accuracy (64%).
-Model Evaluation:<br>
Evaluated using metrics like accuracy and confusion matrix.
-Prediction Pipeline:<br>
Input resumes are preprocessed and vectorized before classification.
# Comparitive Analysis
Models Tested:
1. Logistic Regression
2. Naive Bayes
3. Random Forest Classifier
Model - Accuracy (%) - Observations
Logistic Regression - 64 - Moderate performance, provided meaningful predictions.
Naive Bayes - 57 - Lower accuracy due to independence assumptions.
Random Forest - 72 - Higher accuracy but gave the same output for all inputs. <br>
Logistic Regression was chosen as the final model.<br>
Reason: It provided consistent and correct predictions compared to Random Forest, which struggled with variability and produced the same output for all inputs.
Logistic Regression also demonstrated better interpretability and faster processing times.
# Limitations
-Limited Context Understanding - The model relies on TF-IDF, which does not capture semantic meaning or context, making it less effective for complex sentences.<br>
-Generalization Issues - The model may struggle with resumes containing uncommon terminology or unique formatting.<br>
-Overfitting Risks - Logistic Regression can overfit the training data, especially when features are sparse or redundant.<br>
-Scalability - The current implementation may face performance bottlenecks with very large datasets.
-Bias in Data - Predictions heavily depend on the training data, so biases in the dataset can lead to biased results.v
-Limited Categories - The model is restricted to predefined categories and may not handle new job roles effectively without retraining.<br>
# Future Enhancements :
-Incorporate deep learning models like LSTMs or BERT for improved context understanding.<br>
-Add more categories and handle imbalanced datasets effectively.<br>
-Deploy the model as a web application or integrate it with ATS systems<br>


















