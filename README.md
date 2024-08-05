# Stochastic Model to Predict the Salaries of Tech Professionals and Developers using the StackOverFlow Salary Survey Reports for 2023.
### Project Description:
![stackoverflowimage](https://github.com/user-attachments/assets/c2ffefab-c0c6-4c24-a15d-d7721e9c6bc9)
In this project, I explored the stackoverflow developers salary survey datasets for 2023 and leveraged various techniques to cleanse, preprocessed and engineered features that allowed me cover great insights that are contributive features to determining the salaries of developers and tech professionals.

The stackoverflow survey for 2023 contains the responses of developers, analysts, IT leaders, data scientists, machine learning engineers etc evolving around their experiences, technologies and the new trends and how these factors influenced their earning capacity which is salary - a key target in this project.

Thanks to stackoverflow who deep dived into AI/ML to capture the views of developers and tech professionals and how these concepts impacted their efficiency and earning potentials for diversed organizations. I extensively leveraged on the impact of AI/ML as contributive factors to the earning potentials for any developers in this project.

The focus of this project is to build a robust stochastic model that will capture all varying attributes and capable of predicting the salary of any developers and tech professionals across organizations, country, experiences, technical expertise etc and productionalize this model using streamlit and gradio. The project will also performed indepth exploratory data analsis to cover broader scope and contexts in the data that drives decision-making.

### Project Folders structures:
![2](https://github.com/user-attachments/assets/4f5803d3-c7e5-4e55-8e50-9e73915220ea)

### Modules and Packages:
`pip install python 3.12.1`

`pip install gradio`

`pip install streamlit`

![3](https://github.com/user-attachments/assets/16a663c1-f714-4a04-a188-6113c42258f9)

### Columns of Interest:
![4](https://github.com/user-attachments/assets/836533cc-e197-4449-b387-3b4482288fed)

### Engineered columns of interest:
![5](https://github.com/user-attachments/assets/3417e9e4-ac84-45a9-aae8-87ec721125c4)

### Strategies and Techniques:

- Built functions to automate preprocessing steps.

- Corrected missing values using the mean strategy for numerical columns.

- Corrected missing values for categorical using imputed categorical - fillna().

- Applied the Regression Imputation Techniques for computing the missing values in the target column with over 46% missing values.

- Corrected outliers in the target variable using the `interquartile range`.

- For realistic analysis, I pecked salary to be 5000 and above.

- Preprocessed the numerical using `standardscaler and minmaxscaler` and encoded the categorical using `onehotencoder`.

- Two algorithms were trained on the data - `randomforestregressor` and `xgbregressor`.

- Tunned the `XBGRegressor` model using `cross validation` and `GridSearchCV` to improve the performance.

### Model Evaluation:

`RandomForestRegressor`
![6](https://github.com/user-attachments/assets/e48ef539-cbae-4654-bc51-6a1a5d99d9f7)

![7](https://github.com/user-attachments/assets/bd8ad0fd-6d3e-4f55-8fea-b5070f58b02d)
This implies that, 90.0% of the variances are explainable by the `RandomForestRegressor` model every time it makes prediction with an error of `$26,973.22` in salaries.

`XGBRegressor`
![8](https://github.com/user-attachments/assets/df70428e-07c7-4548-9b4d-fd8e58bff1e6)

![9](https://github.com/user-attachments/assets/c08c0da8-79db-4325-92be-9a2e9591de41)
This implies that, 91.0% of the variances are explainable by the `Hyperparametrized XGBRegressor` model every time it makes prediction with an error of `$26,887.48` in salaries.

#### Performance Summary:
`mean_absolute_error = $26,887.48`

`r2_score = 91.0%`

### Model Productionalization using gradio:

`import gradio as gr`

![20](https://github.com/user-attachments/assets/5ed2fe9d-43bd-4e3c-9ded-fbe3ecabd7b4)

`sample datasets`
![17](https://github.com/user-attachments/assets/b74b7d31-886e-4871-b8d1-15892c30ef85)
![18](https://github.com/user-attachments/assets/fc9246e8-be53-439d-b296-5db252e9d57f)
![19](https://github.com/user-attachments/assets/00e00b37-f5ba-4b72-971d-d9b6a32d9c2d)

`predictions`

![21](https://github.com/user-attachments/assets/9bd25a53-c05f-471d-bcfa-30e04fc22a8b)

### Model Productionalization with Streamlit:
#### Web App - Streamlit
- There are two (2) pages:
  - `Salary_Prediction_Page` - Where users are allowed to select their choices based on different criteria.
  - `Salary_Analysis_Page` - Where I provided detailed exploratory data analysis and visualizations and generate insights that drives and contributed to earning potentials across various spectrum and specializations.

### Salary_Prediction_Page
  
`selected input paramters`

![13](https://github.com/user-attachments/assets/d38bf911-84f8-4040-9447-b6f949e175d1)
![14](https://github.com/user-attachments/assets/56bff69c-4a2d-4c37-b242-1595e4ccb191)
![15](https://github.com/user-attachments/assets/63ce6472-c5ae-4651-b844-247ecd3e8128)

`predictions`

![16](https://github.com/user-attachments/assets/1065922b-279a-4382-8b92-901b23a4a431)

### Salary_Analysis_Page

![23](https://github.com/user-attachments/assets/b01fd1fa-7349-402d-bada-7c7567d787e3)
The pie-chart shows the distribution of developers and tech professionals by country and it shows that conutries combined as others as a proportion of about 56.0% effect on overall but this comprises of more countries and by counts, they tends to be more but `USA` can be said to have more developers and tech professionals as per country distribution with a proportion of about 21.7%.


![24](https://github.com/user-attachments/assets/e632df8d-766c-4f03-ae18-552563c88c73)
The `USA` pays more to developers and tech professionals with an average yearly salary above $200K and developers and tech professionals with level of education below the bachelor's degree earned more compared to other educational milestone.


![25](https://github.com/user-attachments/assets/49496bc5-c09d-40e5-8b54-a99b64daceaf)
Developers and tech professionals with 11 years of experience earned more with an average salary of about $220K yearly.

![26](https://github.com/user-attachments/assets/b6dd87cb-8ea5-4f97-b954-a6c55e351be2)
All across the countries, developers and tech professionals leveraged more on `chatGPT` as their choice of `AI search tools` to perform their task.

![27](https://github.com/user-attachments/assets/23e2e3ea-eda4-4655-901f-5cdf6eaba01c)
All across the countries, developers and tech professionals leveraged more on `HYBRID` as their choice of `working condition` than the `Remote and In-Person` work option.

![28](https://github.com/user-attachments/assets/bcf7b372-07d9-4a76-9a16-b613ccaf3416)
The earning potentials for developers and tech professionals increases as they increase their years of experience with a sharp rise in earnings at `11 years` and constantly increasing at a constant rate after `12 years` of experience until declination at `38 years` of gathered working experience. There is increase in experience at above 50 years of experience and this may require further studies to penetrate factors responsible for earning high at that instance.

![29](https://github.com/user-attachments/assets/ec7a300f-f6db-4ac8-936e-0a2ed4b49296)
It is not news that the larger the size of organization, the higher the expected salaries. In this project, we can observed that, organization having number of employees between `20 to 99` on average pays more to developers and tech professionals that worked in such organization and this validates why companies like `GOOGLE, APPLE, MICROSOFT` etc pays their developers and tech professionals more.

### Running Steps:
![31](https://github.com/user-attachments/assets/e5ec6308-3725-416a-a0eb-9654c5f811d6)

### Contribution
I welcome contributions to this project! Here's how you can contribute:

  - Fork the Repository
  - Clone the Repository
  - Create a New Branch
  - Make Your Changes
  - Commit Your Changes
  - Push Your Changes
  - Submit a Pull Request
  - Remember, contributing to open source projects is about more than just a code. You can also contribute by reporting bugs, suggesting new features, improving documentation, and more.

Thank you for considering contributing to this project! 😊

### Author:
`Name` - `Olanrewaju Adegoke`

`Email` - `Larrysman2004@yahoo.com`






