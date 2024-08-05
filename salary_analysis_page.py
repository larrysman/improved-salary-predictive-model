# import all necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os
import pickle

# copy all the functions used to clean the dataset
def education_level_classifier(edu):
    if 'Bachelor’s degree' in edu:
        return 'Bachelor'
    if 'Master’s degree' in edu:
        return 'Master'
    if 'Professional degree' in edu:
        return 'PhD'
    else:
        return 'below bachelor'


def certification_classifier(cert):
    if 'Udemy' in cert:
        return 'Udemy'
    if 'Codecademy' in cert:
        return 'Codecademy'
    if 'edX' in cert:
        return 'edX'
    if 'Pluralsight' in cert:
        return 'Pluralsight'
    if 'Coursera' in cert:
        return 'Coursera'
    else:
        return 'Others'


def collapse_value_count(value_count_df, cutoff):
    entries_map = {}
    for i in range(len(value_count_df)):
        if value_count_df.values[i] >= cutoff:
            entries_map[value_count_df.index[i]] = value_count_df.index[i]
        else:
            entries_map[value_count_df.index[i]] = 'Others'
    return entries_map


def job_type_classifier(job):
    if 'Developer, full-stack' in job:
        return 'Developer_full-stack'
    if 'Developer, back-end' in job:
        return 'Developer_back-end'
    if 'Developer, front-end' in job:
        return 'Developer_front-end'
    if 'Developer, desktop or enterprise applications' in job:
        return 'Developer_desktop'
    if 'unverified' in job:
        return 'Others'
    else:
        return job


def collapse_value_count(value_count_df, cutoff):
    entries_map = {}
    for i in range(len(value_count_df)):
        if value_count_df.values[i] >= cutoff:
            entries_map[value_count_df.index[i]] = value_count_df.index[i]
        else:
            entries_map[value_count_df.index[i]] = 'Others'
    return entries_map


def rename_row_entry(name):
    if 'United States' in name:
        return 'USA'
    if 'United Kingdom' in name:
        return 'UK'
    else:
        return name


def age_classifier(age):
    if '25-34 years old' in age:
        return '25-34'
    if '35-44 years old' in age:
        return '35-44'
    if '18-24 years old' in age:
        return '18-24'
    if '45-54 years old' in age:
        return '45-54'
    if 'Under 18 years old' in age:
        return 'Under_18'
    if '55-64 years old' in age:
        return '55-64'
    if '65 years or older' in age:
        return 'Above_65'
    if 'Prefer not to say' in age:
        return 'Under_18'
    else:
        return age


def collapse_value_count(value_count_df, cutoff):
    entries_map = {}
    for i in range(len(value_count_df)):
        if value_count_df.values[i] >= cutoff:
            entries_map[value_count_df.index[i]] = value_count_df.index[i]
        else:
            entries_map[value_count_df.index[i]] = 'Others'
    return entries_map


def prog_lang_classifier(lang):
    if 'Bash/Shell (all shells)' in lang:
        return 'Bash/Shell'
    if 'C#' in lang:
        return 'C#'
    if 'HTML/CSS' in lang:
        return 'HTML/CSS'
    if 'C' in lang:
        return 'C'
    if 'Assembly' in lang:
        return 'Assembly'
    if 'C++' in lang:
        return 'C++'
    if 'Java' in lang:
        return 'Java'
    if 'JavaScript' in lang:
        return 'JavaScript'
    if 'Go' in lang:
        return 'Go'
    if 'Python' in lang:
        return 'Python'
    if 'Dart' in lang:
        return 'Dart'
    if 'Delphi' in lang:
        return 'Delphi'
    if 'Others' in lang:
        return 'Others'
    if 'unverified' in lang:
        return 'Others'
    else:
        return lang


def collapse_value_count(value_count_df, cutoff):
    entries_map = {}
    for i in range(len(value_count_df)):
        if value_count_df.values[i] >= cutoff:
            entries_map[value_count_df.index[i]] = value_count_df.index[i]
        else:
            entries_map[value_count_df.index[i]] = 'Others'
    return entries_map


def database_classifier(database):
    if 'unverified' in database:
        return 'Others'
    if 'Others' in database:
        return 'Others'
    else:
        return database


def cloud_platform_classifier(cloud):
    if 'Amazon Web Services (AWS)' in cloud:
        return 'Amazon Web Services (AWS)'
    if 'Microsoft Azure' in cloud:
        return 'Microsoft Azure'
    if 'Google Cloud' in cloud:
        return 'Google Cloud'
    if 'Firebase' in cloud:
        return 'Firebase'
    else:
        return 'Others'


def webframe_classifier(webframe):
    if 'Angular' in webframe:
        return 'Angular'
    if 'Express' in webframe:
        return 'Express'
    if 'Django' in webframe:
        return 'Django'
    if 'ASP.NET' in webframe:
        return 'ASP.NET'
    else:
        return 'Others'


def tech_tool_classifier(tool):
    if 'Docker' in tool:
        return 'Docker'
    if 'Cargo' in tool:
        return 'Cargo'
    if 'Ansible' in tool:
        return 'Ansible'
    if 'npm' in tool:
        return 'npm'
    else:
        return 'Others'


def collab_tool_classifier(collab):
    if 'Android Studio' in collab:
        return 'Android Studio'
    if 'IntelliJ IDEA' in collab:
        return 'IntelliJ IDEA'
    if 'Visual Studio Code' in collab:
        return 'Visual Studio Code'
    if 'Notepad++' in collab:
        return 'Notepad++'
    if 'Visual Studio' in collab:
        return 'Visual Studio'
    else:
        return 'Others'


def aisearch_tool_classifier(ai):
    if 'ChatGPT' in ai:
        return 'ChatGPT'
    if 'Bing AI' in ai:
        return 'Bing_AI'
    if 'WolframAlpha' in ai:
        return 'Wolfram_Alpha'
    if 'Google Bard AI' in ai:
        return 'Google_Bard_AI'
    else:
        return 'Others'


def aidev_tool_classifier(dev):
    if 'GitHub Copilot' in dev:
        return 'GitHub Copilot'
    if 'Tabnine' in dev:
        return 'Tabnine'
    if 'AWS CodeWhisperer' in dev:
        return 'AWS CodeWhisperer'
    else:
        return 'Others'


def employment_classifier(emp):
    if 'Employed, full-time' in emp:
        return 'Employed, full-time'
    if 'Employed, part-time' in emp:
        return 'Employed, part-time'
    if 'Student, full-time' in emp:
        return 'Student, full-time'
    if 'Student, part-time' in emp:
        return 'Student, part-time'
    if 'Independent contractor, freelancer, or self-employed' in emp:
        return 'Independent'
    else:
        return 'Others'


def work_option_classifier(work):
    if 'Hybrid (some remote, some in-person)' in work:
        return 'Hybrid'
    if 'unverified' in work:
        return 'Hybrid'
    else:
        return work


def company_size_classifier(size):
    if 'unverified' in size:
        return '20 to 99 employees'
    if 'I don’t know' in size:
        return '20 to 99 employees'
    if 'Just me - I am a freelancer, sole proprietor, etc.' in size:
        return 'Independent'
    else:
        return size


def industry_classifier(ind):
    if 'unverified' in ind:
        return 'Other'
    else:
        return ind


def yearsofcoding(year):
    if year == 'Less than 1 year':
        return float(0.5)
    if year == 'More than 50 years':
        return float(51)
    else:
        return float(year)


def years_of_professional_coding(year):
    if year == 'Less than 1 year':
        return float(0.5)
    if year == 'More than 50 years':
        return float(51)
    else:
        return float(year)


# The paths to the project folders
main_path = 'C:\\Users\\Olanrewaju Adegoke\\Desktop\\TechTern\\mywork\\Stackoverflow_Salary_Project'
data_path = '../data'
model_path = '../models'
note_path = '../notebooks'
out_path = '../outputs'
vis_path = '../visuals'
res_path = '../resources'


# Function that load data and cleanse data for further analysis
@st.cache
def load_and_clean_data():
    os.chdir(data_path)
    df = pd.read_csv('stackoverflow.csv')
    os.chdir(out_path)
    data = df.copy()
    # select columns of interest for the analysis
    cols = ['EdLevel', 'LearnCodeCoursesCert', 'YearsCode', 'YearsCodePro', 'DevType', 'Country', 'Age',
        'LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith',
       'ToolsTechHaveWorkedWith', 'NEWCollabToolsHaveWorkedWith', 'AISearchHaveWorkedWith', 'AIDevHaveWorkedWith',
       'Employment', 'RemoteWork', 'OrgSize', 'WorkExp', 'Industry', 'ConvertedCompYearly']
    
    useful_df = data[cols]
    # selecting numerical, categorical and target columns from the selected data for the analysis
    cat_cols = [
    'EdLevel', 'LearnCodeCoursesCert', 'DevType', 'Country', 'Age', 'LanguageHaveWorkedWith',
    'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith', 'ToolsTechHaveWorkedWith',
    'NEWCollabToolsHaveWorkedWith', 'AISearchHaveWorkedWith', 'AIDevHaveWorkedWith', 'Employment', 'RemoteWork', 'OrgSize',
    'Industry']

    num_cols = [
    'YearsCode', 'YearsCodePro', 'WorkExp']

    target_col = ['ConvertedCompYearly']
    # cleaning the categorical columns
    missing_value = 'unverified'
    useful_df[cat_cols] = useful_df[cat_cols].fillna(missing_value)

    useful_df['highest_education'] = useful_df['EdLevel'].apply(education_level_classifier)
    useful_df['certifications'] = useful_df['LearnCodeCoursesCert'].apply(lambda x: x.split(';')[0])
    useful_df['certifications'] = useful_df.certifications.apply(certification_classifier)
    value_count_df = useful_df.DevType.value_counts()
    cutoff = 3900
    entries_map = collapse_value_count(value_count_df, cutoff)
    useful_df['job_type'] = useful_df['DevType'].map(entries_map)
    useful_df['job_type'] = useful_df['job_type'].apply(job_type_classifier)
    value_count_df = useful_df.Country.value_counts()
    cutoff = 3500
    entries_map = collapse_value_count(value_count_df, cutoff)
    useful_df['country'] = useful_df['Country'].map(entries_map)
    useful_df['country'] = useful_df['country'].apply(rename_row_entry)
    useful_df['age_range'] = useful_df['Age'].apply(age_classifier)
    useful_df['prog_lang'] = useful_df['LanguageHaveWorkedWith'].apply(lambda lang: lang.split(';')[0])
    value_count_df = useful_df.prog_lang.value_counts()
    cutoff = 1000
    entries_map = collapse_value_count(value_count_df, cutoff)
    useful_df['prog_lang'] = useful_df['prog_lang'].map(entries_map)
    useful_df['prog_lang'] = useful_df['prog_lang'].apply(prog_lang_classifier)
    useful_df['database'] = useful_df['DatabaseHaveWorkedWith'].apply(lambda database: database.split(';')[0])
    value_count_df = useful_df.database.value_counts()
    cutoff = 1000
    entries_map = collapse_value_count(value_count_df, cutoff)
    useful_df['database'] = useful_df['database'].map(entries_map)
    useful_df['database'] = useful_df['database'].apply(database_classifier)
    useful_df['cloud_platform'] = useful_df['PlatformHaveWorkedWith'].apply(lambda platform: platform.split(';')[0])
    useful_df['cloud_platform'] = useful_df['cloud_platform'].apply(cloud_platform_classifier)
    useful_df['webframe'] = useful_df['WebframeHaveWorkedWith'].apply(lambda webframe: webframe.split(';')[0])
    useful_df['webframe'] = useful_df['webframe'].apply(webframe_classifier)
    useful_df['tech_tool'] = useful_df['ToolsTechHaveWorkedWith'].apply(lambda tech_tool: tech_tool.split(';')[0])
    useful_df['tech_tool'] = useful_df['tech_tool'].apply(tech_tool_classifier)
    useful_df['collab_tool'] = useful_df['NEWCollabToolsHaveWorkedWith'].apply(lambda collab: collab.split(';')[0])
    useful_df['collab_tool'] = useful_df['collab_tool'].apply(collab_tool_classifier)
    useful_df['aisearchtool'] = useful_df['AISearchHaveWorkedWith'].apply(lambda aisearch: aisearch.split(';')[0])
    useful_df['aisearchtool'] = useful_df['aisearchtool'].apply(aisearch_tool_classifier)
    useful_df['aidevtool'] = useful_df['AIDevHaveWorkedWith'].apply(lambda aidev: aidev.split(';')[0])
    useful_df['aidevtool'] = useful_df['aidevtool'].apply(aidev_tool_classifier)
    useful_df['employment_status'] = useful_df['Employment'].apply(lambda emp: emp.split(';')[0])
    useful_df['employment_status'] = useful_df['employment_status'].apply(employment_classifier)
    useful_df['work_option'] = useful_df['RemoteWork'].apply(work_option_classifier)
    useful_df['org_size'] = useful_df['OrgSize'].apply(company_size_classifier)
    useful_df['industry'] = useful_df['Industry'].apply(industry_classifier)
    # cleaning numerical columns
    useful_df['years_of_coding'] = useful_df['YearsCode'].apply(yearsofcoding)
    avg_year_of_coding = round(useful_df['years_of_coding'].astype('float').mean(axis=0), 0)
    useful_df['years_of_coding'] = useful_df['years_of_coding'].replace(np.nan, avg_year_of_coding)
    useful_df['years_of_pro_coding'] = useful_df['YearsCodePro'].apply(years_of_professional_coding)
    avg_year_of_pro_coding = round(useful_df['years_of_pro_coding'].astype('float').mean(axis=0), 0)
    useful_df['years_of_pro_coding'] = useful_df['years_of_pro_coding'].replace(np.nan, avg_year_of_pro_coding)
    avg_work_exp = round(useful_df['WorkExp'].astype('float').mean(axis=0), 0)
    useful_df['years_of_work_exp'] = useful_df['WorkExp'].replace(np.nan, avg_work_exp)
    # dropping off unwanted columns
    useful_df = useful_df.drop(columns=cat_cols)
    useful_df = useful_df.drop(columns=num_cols)
    # renaming the ConvertedCompYearly to salary
    useful_df = useful_df.rename(columns={'ConvertedCompYearly': 'salary'})

    return useful_df

# instantiate the function
useful_df = load_and_clean_data()

# Function that load the missing values model and preprocessing steps
def load_missing_values_prep():
    os.chdir(model_path)
    with open('missing_preprocess_step.pkl', 'rb') as file:
        missing_prep = pickle.load(file)
    return missing_prep

missing_prep = load_missing_values_prep()
os.chdir(out_path)
# instantiate the instances of the model and preprocessing artifacts for predicting the missing values
missing_model = missing_prep['model_missing']
missing_scaler = missing_prep['scaler_missing']
missing_minmax = missing_prep['min_max_missing']
missing_onehot = missing_prep['onehot_missing']

# Function for performing the correctness of the missing values
def correct_missing_values_and_returned_cleaned_df():
    os.chdir(out_path)
    df = useful_df.copy()
    # select dataframe with and without missing values
    df_noNAN = df[df['salary'].notnull()]
    df_NAN = df[df['salary'].isnull()]
    nan_df = df_NAN.copy()
    df_NAN = df_NAN.drop(columns=['salary'])
    # preprocess the dataframe containing the missing values
    num_data = df_NAN.select_dtypes(include=['int', 'float'])
    cat_data = df_NAN.select_dtypes(include=['object'])
    num = missing_scaler.transform(num_data)
    num = missing_minmax.transform(num)
    num = pd.DataFrame(num, columns=num_data.columns)
    cat = missing_onehot.transform(cat_data)
    cat = cat.reset_index(drop=True)
    features = pd.concat([num, cat], axis=1)
    feat_arr = features.values
    y_pred_missing = missing_model.predict(feat_arr)
    nan_df.loc[nan_df['salary'].isnull(), 'salary'] = y_pred_missing
    cleaned_df = pd.concat([nan_df, df_noNAN], axis=0)
    # correcting the outlier from the target
    col = 'salary'
    q1 = cleaned_df[col].quantile(0.25)
    q3 = cleaned_df[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    cleaned_df[col] = np.where(cleaned_df[col] < lb, lb, cleaned_df[col])
    cleaned_df[col] = np.where(cleaned_df[col] > ub, ub, cleaned_df[col])
    # selecting salary greater than or equals to 5000
    cleaned_df = cleaned_df[cleaned_df[col] >= 5000]
    return cleaned_df

# instantiate the function
cleaned_df = correct_missing_values_and_returned_cleaned_df()


# Building the function that display the analysis on streamlit
def salary_analysis_page():
    st.title(''' Exploratory Analysis and Visualizations of features that drives salary ''')
    st.write('''###### The StackOverflow Developer and Tech Professional Salary Survey for 2023''')

    # The first chart
    data = cleaned_df['country'].value_counts()
    plt.figure(figsize=(20, 8))
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct='%1.1f%%', shadow=True, startangle=45)
    ax.axis('equal')
    ax.legend(title='countries', bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, labels=[f'{label} ({count})' for label, count in data.items()])
    st.write('''###### Number of Developers and Tech Professionals by countries ''')
    st.pyplot(fig)

    # The second charts
    st.write('''###### Mean Salary based on Country''') 
    data = cleaned_df.groupby(cleaned_df['country'])['salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    # The third charts
    st.write('''###### Average Salary based on Level of Education''') 
    data = cleaned_df.groupby(cleaned_df['highest_education'])['salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    # The fourth charts
    st.write('''###### Average Salary based on Certifications''') 
    data = cleaned_df.groupby(cleaned_df['certifications'])['salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    # The fifth charts
    st.write('''###### Average Salary based on Age Range''') 
    data = cleaned_df.groupby(cleaned_df['age_range'])['salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    # The sixth charts
    st.write('''###### Average Salary based on Work Experience''') 
    data = cleaned_df.groupby(cleaned_df['years_of_work_exp'])['salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    # The seventh charts
    st.write('''##### The count of country ''')
    fig, ax = plt.subplots(figsize=(10, 4))
    sb.countplot(x='country', data=cleaned_df, ax=ax, palette='pastel')
    ax.set_title('The count of country')
    ax.set_xlabel('countries')
    ax.set_ylabel('The number of countries')
    st.pyplot(fig)

    # The eight charts
    st.write('''##### The count of aidevtool per country ''')
    fig, ax = plt.subplots(figsize=(10, 4))
    sb.countplot(x='country', data=cleaned_df, hue='aidevtool', ax=ax, palette='coolwarm')
    ax.set_title('The count of aidevtool per country')
    ax.set_xlabel('countries')
    ax.set_ylabel('The number of aidevtool')
    st.pyplot(fig)

    # The nineth charts
    st.write('''##### The count of aisearchtool per country ''')
    fig, ax = plt.subplots(figsize=(10, 4))
    sb.countplot(x='country', data=cleaned_df, hue='aisearchtool', ax=ax, palette='viridis')
    ax.set_title('The count of aisearchtool per country')
    ax.set_xlabel('countries')
    ax.set_ylabel('The number of aisearchtool')
    st.pyplot(fig)

    # The tenth charts
    st.write('''##### The employment status per country ''')
    fig, ax = plt.subplots(figsize=(10, 4))
    sb.countplot(x='country', data=cleaned_df, hue='employment_status', ax=ax, palette='pastel')
    ax.set_title('The employment status per country')
    ax.set_xlabel('countries')
    ax.set_ylabel('The number of employment')
    st.pyplot(fig)

    # The eleventh charts
    st.write('''##### The work_option per country ''')
    fig, ax = plt.subplots(figsize=(10, 4))
    sb.countplot(x='country', data=cleaned_df, hue='work_option', ax=ax, palette='pastel')
    ax.set_title('The work_option per country')
    ax.set_xlabel('countries')
    ax.set_ylabel('The number of work_option')
    st.pyplot(fig)

    # The twelveth charts
    st.write('''##### Average Salary vs Years of Experience''')
    data = cleaned_df.groupby(cleaned_df['years_of_work_exp'])['salary'].mean().sort_values(ascending=True)
    st.line_chart(data)
    st.markdown('**X_axis:** Years of Experience')
    st.markdown('**Y_axis:** Average Salary')

    # The thirteenth charts
    st.write('''##### Average Salary vs Organization size ''')
    data = cleaned_df.groupby(cleaned_df['org_size'])['salary'].mean().sort_values(ascending=True)
    st.line_chart(data)
    st.markdown('**x_axis:** Organization size')
    st.markdown('**y_axis:** Average salary')

    # The fourteenth charts
    st.write('''##### Average Salary vs Industry ''')
    data = cleaned_df.groupby(cleaned_df['industry'])['salary'].mean().sort_values(ascending=True)
    st.line_chart(data)
    st.markdown('**x_axis:** industries')
    st.markdown('**y_axis:** Average salary')

    # The fourteenth charts
    st.write('''##### Average Salary vs Job type ''')
    data = cleaned_df.groupby(cleaned_df['job_type'])['salary'].mean().sort_values(ascending=True)
    st.line_chart(data)
    st.markdown('**x_axis:** job_type')
    st.markdown('**y_axis:** Average salary')








