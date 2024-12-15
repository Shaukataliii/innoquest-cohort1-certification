import os

import streamlit as st

from module_class_6 import Validator, Transformer, AbsentHoursPredictor

# Constants
MODEL_PATH = f'{os.getcwd()}/pkls/model.pkl'

# Streamlit App Configuration
st.title("Absent Hours Predictor")
st.divider()

# Main Function
if __name__ == "__main__":
    predictor = AbsentHoursPredictor(MODEL_PATH)
    predictor.display_inputs()

    if st.session_state.get('submit_btn'):
        with st.spinner("Predicting..."):
            prediction = predictor.process_inputs()

            if prediction == 0:
                st.write(f":blue[The employee is not gonna be off.]")
            else:
                st.write(f":red[The employee is gonna be off.]")
