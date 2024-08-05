# import all necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import pickle

# load the paths to the project folders
main_path = 'C:\\Users\\Olanrewaju Adegoke\\Desktop\\TechTern\\mywork\\Stackoverflow_Salary_Project'
data_path = '../data'
model_path = '../models'
note_path = '../notebooks'
out_path = '../outputs'
vis_path = '../visuals'
res_path = '../resources'

# define a function that load the choice saved model and the preprocessing steps
def load_saved_model_and_artifact():
    os.chdir(model_path)
    with open('model_preprocessing_steps_stackoverflow.pkl', 'rb') as file:
        inference = pickle.load(file)
    return inference

# instantiate the instances of the saved model and preprocessing steps
inference = load_saved_model_and_artifact()
os.chdir(out_path)
inference_model = inference['model']
inference_scaler = inference['scaler']
inference_minmax = inference['min_max']
inference_onehot = inference['onehot']
model = inference_model

# Build the streamlit function to show or displays the predicted salary
def salary_prediction_page():
    st.title(''' Salary Prediction Model for Developers and Tech Professional by Olanrewaju Adegoke ''')
    st.write('''###### You are required to select all relevant information to predict the salary''')

    # Define all the categorical columns you cleaned
    highest_education = (
        'Bachelor',
        'below bachelor',
        'Master',
        'PhD'
    )

    certifications = (
        'Others',
        'Udemy',
        'Codecademy',
        'edX',
        'Pluralsight',
        'Coursera'
    )

    job_type = (
        'Others',
        'Developer_full-stack',
        'Developer_back-end',
        'Developer_front-end',
        'Developer_desktop'
    )

    country = (
        'Others',
        'USA',
        'Germany',
        'India',
        'UK',
        'Canada'
    )

    age_range = (
        '25-34',
        '35-44',
        '18-24',
        '45-54',
        'Under_18',
        '55-64',
        'Above_65'
    )

    prog_lang = (
        'Bash/Shell',
        'C#',
        'HTML/CSS',
        'C',
        'Others',
        'Java',
        'Assembly',
        'Go',
        'Python',
        'Dart',
        'Delphi'
    )

    database = (
        'Others',
        'Microsoft SQL Server',
        'PostgreSQL',
        'MySQL',
        'MariaDB',
        'MongoDB',
        'Elasticsearch',
        'Dynamodb',
        'Cloud Firestore',
        'BigQuery',
        'SQLite',
        'Cosmos DB',
        'Firebase Realtime Database',
        'Cassandra',
        'Microsoft Access',
        'H2',
        'Oracle'
    )

    cloud_platform = (
        'Others',
        'Amazon Web Services (AWS)',
        'Microsoft Azure',
        'Google Cloud',
        'Firebase'
    )

    webframe = (
        'Others',
        'Angular',
        'ASP.NET',
        'Express',
        'Django'
    )

    tech_tool = (
        'Others',
        'Docker',
        'Cargo',
        'Ansible',
        'npm'
    )

    collab_tool = (
        'Others',
        'Android Studio',
        'IntelliJ IDEA',
        'Visual Studio Code',
        'Notepad++',
        'Visual Studio'
    )

    aisearchtool = (
        'ChatGPT',
        'Others',
        'Bing_AI',
        'Wolfram_Alpha',
        'Google_Bard_AI'
    )

    aidevtool = (
        'Others',
        'GitHub Copilot',
        'Tabnine',
        'AWS CodeWhisperer'
    )

    employment_status = (
        'Employed, full-time',
        'Student, full-time',
        'Independent',
        'Others',
        'Employed, part-time',
        'Student, part-time'
    )

    work_option = (
        'Hybrid',
        'Remote',
        'In-person'
    )

    org_size = (
        '20 to 99 employees',
        '100 to 499 employees',
        '10,000 or more employees',
        '1,000 to 4,999 employees',
        '2 to 9 employees',
        '10 to 19 employees',
        '500 to 999 employees',
        'Independent',
        '5,000 to 9,999 employees'
    )

    industry = (
        'Other',
        'Information Services, IT, Software Development, or other Technology',
        'Financial Services',
        'Manufacturing, Transportation, or Supply Chain',
        'Healthcare',
        'Retail and Consumer Services',
        'Higher Education',
        'Advertising Services',
        'Insurance',
        'Oil & Gas',
        'Legal Services',
        'Wholesale'
    )
    # Instantiate all the categorical columns cleaned in streamlit
    highest_education = st.selectbox('Highest_education', highest_education)
    certifications = st.selectbox('Certifications', certifications)
    job_type = st.selectbox('Job_type', job_type)
    country = st.selectbox('Country', country)
    age_range = st.selectbox('Age_range', age_range)
    prog_lang = st.selectbox('Programming_lang', prog_lang)
    database = st.selectbox('Database', database)
    cloud_platform = st.selectbox('Cloud_platform', cloud_platform)
    webframe = st.selectbox('Webframe', webframe)
    tech_tool = st.selectbox('Technical_tool', tech_tool)
    collab_tool = st.selectbox('Collab_tool', collab_tool)
    aisearchtool = st.selectbox('AIsearchtool', aisearchtool)
    aidevtool = st.selectbox('AIdevtool', aidevtool)
    employment_status = st.selectbox('Employment_type', employment_status)
    work_option = st.selectbox('Work_option', work_option)
    org_size = st.selectbox('Organization_size', org_size)
    industry = st.selectbox('Industry', industry)

    # instantiate all the numerical columns cleaned in streamlit using sliderbar
    years_of_coding = st.slider('Years of coding', min_value=0, max_value=55, value=2)
    years_of_pro_coding = st.slider('Years of Professional Coding', min_value=0, max_value=55, value=5)
    years_of_work_exp = st.slider('Work_Experience', min_value=0, max_value=50, value=3)

    predict_button = st.button('Compute Salary')
    if predict_button:

        def prep_prediction_inference(
            highest_education, certifications, job_type, country,
            age_range, prog_lang, database, cloud_platform, webframe,
            tech_tool, collab_tool, aisearchtool, aidevtool,
            employment_status, work_option, org_size, industry,
            years_of_coding, years_of_pro_coding, years_of_work_exp,
            inference_scaler, inference_minmax, inference_onehot, model
        ):
            cols = [
                'highest_education', 'certifications', 'job_type', 'country',
            'age_range', 'prog_lang', 'database', 'cloud_platform', 'webframe',
            'tech_tool', 'collab_tool', 'aisearchtool', 'aidevtool',
            'employment_status', 'work_option', 'org_size', 'industry',
            'years_of_coding', 'years_of_pro_coding', 'years_of_work_exp'
            ]
            input_data = np.array([[
                highest_education, certifications, job_type, country,
                age_range, prog_lang, database, cloud_platform, webframe,
                tech_tool, collab_tool, aisearchtool, aidevtool,
                employment_status, work_option, org_size, industry,
                years_of_coding, years_of_pro_coding, years_of_work_exp
            ]])
            input_df = pd.DataFrame(input_data, columns=cols)

            # Convert the string columns to numeric
            input_df['years_of_coding'] = pd.to_numeric(input_df['years_of_coding'], downcast='float')
            input_df['years_of_pro_coding'] = pd.to_numeric(input_df['years_of_pro_coding'], downcast='float')
            input_df['years_of_work_exp'] = pd.to_numeric(input_df['years_of_work_exp'], downcast='float')
            
            
            num_data = input_df.select_dtypes(include=['int', 'float'])
            cat_data = input_df.select_dtypes(include=['object'])
            
            num = inference_scaler.transform(num_data)
            num = inference_minmax.transform(num)
            num_df = pd.DataFrame(num, columns=list(num_data.columns))
            
            cat = inference_onehot.transform(cat_data)
            cat_df = cat.reset_index(drop=True) 

            features = pd.concat([num_df, cat_df], axis=1)
            predictions = model.predict(features)
            salary = predictions[0]

            return salary

        salary = prep_prediction_inference(
            highest_education, certifications, job_type, country,
            age_range, prog_lang, database, cloud_platform, webframe,
            tech_tool, collab_tool, aisearchtool, aidevtool,
            employment_status, work_option, org_size, industry,
            years_of_coding, years_of_pro_coding, years_of_work_exp,
            inference_scaler=inference_scaler, inference_minmax=inference_minmax, inference_onehot=inference_onehot, model=inference_model
        )
        st.subheader(f'The estimated salary is: ${salary:,.2f}')