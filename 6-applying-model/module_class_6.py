import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression


class Validator:
    @staticmethod
    def validate_inputs(inputs: dict) -> bool:
        """Validates if all required numerical input fields are filled."""
        required_fields = [
            'education', 'body_mass_index', 'age',
            'transportation_expense', 'distance_to_work',
            'daily_work_load_average'
        ]
        if any(inputs.get(field) in [0, None] for field in required_fields):
            st.error("Please fill all the required numerical input fields.")
            return False
        return True


class Transformer:
    # Mapping column names for consistency
    COLUMN_MAP = {
        'transportation_expense': 'Transportation Expense',
        'distance_to_work': 'Distance to Work',
        'age': 'Age',
        'daily_work_load_average': 'Daily Work Load Average',
        'body_mass_index': 'Body Mass Index',
        'education': 'Education',
        'children': 'Children',
        'pets': 'Pets',
        'reason_1': 'Reason_1',
        'reason_2': 'Reason_2',
        'reason_3': 'Reason_3',
        'reason_4': 'Reason_4',
        'day': 'Day',
        'month': 'Month',
        'weekday': 'Weekday',
    }

    @staticmethod
    def transform_inputs(inputs: dict) -> pd.DataFrame:
        """Transforms inputs into a format suitable for machine learning models."""
        inputs['education'] = Transformer._transform_education(inputs['education'])
        inputs['children'] = Transformer._transform_count(inputs['children'])
        inputs['pets'] = Transformer._transform_count(inputs['pets'])
        inputs['reason'] = Transformer._transform_reason(inputs['reason'])
        inputs['day'], inputs['month'], inputs['weekday'] = Transformer._transform_date(inputs['date'])

        return Transformer._prepare_df(inputs)

    @staticmethod
    def _transform_education(education: int) -> int:
        """Maps education levels into binary categories."""
        return 1 if education in [2, 3, 4] else 0

    @staticmethod
    def _transform_count(count: int | str) -> int:
        """Handles transformation of count fields like children or pets."""
        return 3 if count == 'More than 2' else int(count)

    @staticmethod
    def _transform_reason(reason: str) -> list:
        """Encodes the reason into a one-hot vector."""
        reason_map = {
            'Category A': [1, 0, 0, 0],
            'Category B': [0, 1, 0, 0],
            'Category C': [0, 0, 1, 0],
            'Category D': [0, 0, 0, 1],
        }
        return reason_map.get(reason, [0, 0, 0, 0])

    @staticmethod
    def _transform_date(date) -> tuple:
        """Extracts day, month, and weekday from a date."""
        return date.day, date.month, date.weekday()

    @staticmethod
    def _prepare_df(inputs: dict) -> pd.DataFrame:
        """Prepares the final DataFrame for model input."""
        row = {
            Transformer.COLUMN_MAP['transportation_expense']: inputs['transportation_expense'],
            Transformer.COLUMN_MAP['distance_to_work']: inputs['distance_to_work'],
            Transformer.COLUMN_MAP['age']: inputs['age'],
            Transformer.COLUMN_MAP['daily_work_load_average']: inputs['daily_work_load_average'],
            Transformer.COLUMN_MAP['body_mass_index']: inputs['body_mass_index'],
            Transformer.COLUMN_MAP['education']: inputs['education'],
            Transformer.COLUMN_MAP['children']: inputs['children'],
            Transformer.COLUMN_MAP['pets']: inputs['pets'],
            Transformer.COLUMN_MAP['reason_1']: inputs['reason'][0],
            Transformer.COLUMN_MAP['reason_2']: inputs['reason'][1],
            Transformer.COLUMN_MAP['reason_3']: inputs['reason'][2],
            Transformer.COLUMN_MAP['reason_4']: inputs['reason'][3],
            Transformer.COLUMN_MAP['day']: inputs['day'],
            Transformer.COLUMN_MAP['month']: inputs['month'],
            Transformer.COLUMN_MAP['weekday']: inputs['weekday'],
        }

        return pd.DataFrame([row])


class AbsentHoursPredictor:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path

    def display_inputs(self) -> None:
        """Displays the input fields for the prediction form."""
        # Numerical input fields
        num_fields = {
            'Transportation Expense': (0, 1000),
            'Distance to Work': (0, 100),
            'Age': (18, 65),
            'Daily Work Load Average': (0, 300),
            'Body Mass Index': (10, 50),
        }

        for field, (min_val, max_val) in num_fields.items():
            st.number_input(field, min_value=min_val, max_value=max_val, key=field.lower().replace(' ', '_'))

        # Categorical fields
        st.selectbox('Education', options=[1, 2, 3, 4], key='education')
        st.selectbox('Children', options=[0, 1, 2, 'More than 2'], key='children')
        st.selectbox('Pets', options=[0, 1, 2, 'More than 2'], key='pets')
        st.selectbox('Reason', options=['Category A', 'Category B', 'Category C', 'Category D'], key='reason')

        # Date input
        st.date_input('Date', key='date')

        # Submit button
        st.button('Predict', type='primary', key='submit_btn')

    def process_inputs(self) -> int:
        """Processes user inputs, validates them, and returns the prediction."""
        inputs = dict(st.session_state).copy()

        if Validator.validate_inputs(inputs):
            df = Transformer.transform_inputs(inputs)
            prediction = self.get_prediction(df)
            return prediction
        else:
            st.error("Invalid inputs. Please check the fields again.")
            st.stop()

    def get_prediction(self, inputs: pd.DataFrame) -> int:
        """Returns the prediction based on the model and transformed inputs."""
        # Use the standalone cached function to load the model
        model = load_model(self.model_path)
        return model.predict(inputs)[0]
    

# Standalone model loading function
@st.cache_data
def load_model(model_path: str) -> LogisticRegression:
    """Loads the trained logistic regression model from the file system."""
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
        return model
    else:
        st.error("Model file not found.")
        st.stop()
