- Make sure to adjust the paths because I created the folder structure after completing the class. However, the app should run fine.

# ML Absenteeism Prediction Model

This repository contains the machine learning model developed to predict absenteeism using various features. The model was built, trained, and tested during the Innoquest Cohort-1 Class 6. This repository includes the preprocessing steps, feature encoding, and the final model used for prediction.

## How to Run the Application

For inference, we're using the model trained without feature scaling. To perform inference with the model through a Streamlit application, follow these steps:

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install the Requirements**  
   Install the necessary dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**  
   To launch the application, run the following command:
   ```bash
   streamlit run app.py
   ```

4. **Access the Application**  
   Once the app is running, open your browser and go to the URL provided by Streamlit (typically `http://localhost:8501`).

## Notebooks Used for Model Development

The following Jupyter notebooks were used for developing and analyzing the machine learning model:

1. **Preprocessing Notebook**  
   This notebook covers the data preprocessing steps, including handling missing values, encoding categorical variables, and scaling numerical features. It also includes the analysis and visualization of the data.

2. **Model Building Notebook**  
   This notebook details the steps for building the model. It covers the Logistic Regression model (for categorical targets), including training, testing, and evaluating the models.


## Features
- **Data Preprocessing**: Includes encoding techniques and handling of missing data.
- **Model Building**: The model is built using both encoded and scaled data.
- **Prediction**: The app allows you to make predictions based on user input.

## Accuracy
- The model achieved an accuracy of **75%** on the dataset, with a similar accuracy of **73%** from the instructor.

## Suggestions for Improvements
- Add additional feature engineering techniques to optimize model performance.
- Experiment with other encoding methods to improve results.
- Perform hyperparameter tuning for model optimization.

## License
MIT License