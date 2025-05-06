# potential-customer-prediction
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Used-green.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Used-orange.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Used-purple.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project tackles a common challenge in the rapidly growing EdTech industry: identifying which prospective leads are most likely to convert into paying customers. Using a dataset provided by the hypothetical startup "ExtraaLearn", this analysis involves building and evaluating a machine learning pipeline to predict lead conversion based on user demographics, engagement metrics, and marketing channel interactions.

The primary goal is to demonstrate a practical application of data science techniques for lead scoring and prioritization, ultimately helping businesses optimize their sales and marketing efforts. This project serves as a portfolio piece showcasing data analysis, feature engineering, model building, and interpretation skills.

## Motivation & Problem Statement

In the competitive EdTech landscape, efficiently converting leads is crucial for growth. Sales and marketing teams often have limited resources and need to focus their efforts on prospects with the highest conversion potential.

This project aims to:
1.  Develop predictive models to estimate the probability of a lead converting.
2.  Identify key factors (demographics, behavior, acquisition source) that significantly influence conversion.
3.  Provide actionable insights to help ExtraaLearn allocate resources more effectively and tailor engagement strategies.

## Repository Contents

This repository includes the following files:

-   `01_model_training_notebook.ipynb`: The complete Jupyter Notebook containing all steps of the analysis, from data loading and EDA to model training and evaluation.
-   `02_model_output_summary.html`: An HTML export of the Jupyter Notebook for easy viewing in a web browser without needing to run the code.
-   `03_customer_data_raw.csv`: The raw input dataset containing information about leads and their interactions.
-   `README.md`: This file, providing an overview of the project.

## Methodology & Tools

This project follows a standard data science workflow:

1.  **Data Loading & Initial Exploration:**
    * Loaded the `03_customer_data_raw.csv` dataset using **Pandas**.
    * Performed initial checks for data dimensions, data types, missing values (none found), and duplicates (none found).
    * Dropped the non-predictive `ID` column.

2.  **Exploratory Data Analysis (EDA):**
    * **Univariate Analysis:** Examined distributions of individual features using histograms and box plots (**Plotly**) for numerical variables (`age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit`) and bar charts (**Plotly**) for categorical variables (`current_occupation`, `first_interaction`, `profile_completed`, `last_activity`, media/referral flags). Identified skewness in engagement metrics and key characteristics of the lead pool (e.g., dominant 'Professional' occupation, high profile completion rates).
    * **Bivariate Analysis:** Investigated relationships between features and the target variable (`status`) to uncover initial patterns related to conversion.

3.  **Data Preprocessing:**
    * **Encoding:**
        * Applied **One-Hot Encoding** (`pd.get_dummies`) to nominal categorical features.
        * Applied **Ordinal Encoding** to the `profile_completed` feature ('Low', 'Medium', 'High').
    * **Train-Test Split:** Divided the data into training and testing sets using **Scikit-learn**'s `train_test_split` with stratification to maintain class balance.
        *(Note: Feature scaling was not applied as tree-based models like Decision Trees and Random Forests are generally insensitive to the scale of numerical features.)*

4.  **Model Building & Evaluation:**
    * Implemented and evaluated the following classification models using **Scikit-learn**:
        * **Decision Tree Classifier:** Explored feature importance and model performance both with default parameters and after basic hyperparameter tuning (e.g., adjusting `max_depth`) to mitigate overfitting.
        * **Random Forest Classifier:** Evaluated this ensemble method using default parameters and explored potential improvements via basic hyperparameter tuning.
    * **Evaluation Metrics:** Assessed model performance using:
        * Accuracy
        * Precision
        * Recall
        * F1-Score
        * ROC AUC Score
        * Confusion Matrix
        * Classification Report

5.  **Tools & Libraries:**
    * **Core:** Python, Pandas, NumPy
    * **Visualization:** Matplotlib, Seaborn, Plotly
    * **Machine Learning:** Scikit-learn
    * **(Statistical Inference):** Statsmodels

## Key Findings & Results

* **Data Quality:** The raw file had no missing values or duplicates. However, conversions made up just 29.9 % of all leads, creating a 70/30 class imbalance that shaped metric selection (Recall > Accuracy) and the choice to use class‑balanced models.
* **Engagement Drives Conversion:** EDA revealed heavy right‑skew in behaviour features (e.g., page views per visit) and higher conversion odds for leads with High profile completion or repeated site visits—insights later confirmed by feature‑importance plots.
* **Baseline Models Flagged Overfitting:**
   * **Initial DT:** (max_depth = 5) reached 86 % accuracy / 0.77 Recall on converts.
   * **Initial RF:** hit perfect train scores but slid to 85 % accuracy / 0.71 Recall on the test set—clear overfitting.
* **Hyper‑tuning Lifted Recall by ~11 pp:** 
   * **Tuned DT:** 84 % accuracy, 0.87 Recall, 0.69 Precision on converts.
   * **Tuned RF:** 84 % accuracy, 0.88 Recall, 0.68 Precision, slashing false negatives from 121 → 48.
* **Chosen Model – Tuned Random Forest:** With the highest Recall (88 %) and fewest missed converters (48), the tuned RF best meets ExtraaLearn’s goal of catching nearly nine out of ten potential customers while keeping precision and overall accuracy steady. These probabilities can now feed a lead‑scoring dashboard to focus sales outreach where it matters most.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arifpazhwak/potential-customer-prediction.git
    cd potential-customer-prediction
    ```
2.  **Set up environment (Recommended):**
    Create a virtual environment and install dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels jupyterlab dash jupyter-dash
    ```
3.  **Explore the Analysis:**
    * Open the `01_model_training_notebook.ipynb` file in JupyterLab or VS Code to view and run the code step-by-step.
    * Alternatively, open the `02_model_output_summary.html` file directly in your web browser for a static view of the notebook and its outputs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

* **Arif Pazhwak**
    * [LinkedIn](https://www.linkedin.com/in/arifpazhwak/)
