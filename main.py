import streamlit as st
from salary_prediction_page import salary_prediction_page
from salary_analysis_page import salary_analysis_page

# call the function
#salary_prediction_page()
#salary_analysis_page()

# Creating the multiple pages for the exploratory analysis and visualizations
pages = st.sidebar.selectbox('salary_prediction or salary_analysis', ('salary_prediction', 'salary_analysis'))

if pages == 'salary_prediction':
    salary_prediction_page()
else:
    salary_analysis_page()