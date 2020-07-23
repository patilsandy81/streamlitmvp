import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
import os
import featuretools as ft
import seaborn as sns
import re
import streamlit as st
from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
import sklearn.datasets
import sklearn.ensemble
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular



# Standard Library Imports
from pathlib import Path

# Installed packages
from ipywidgets import widgets

# Our package
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file

from streamlit import caching
caching.clear_cache()


    
def main():
    
    st.sidebar.title("Gearbox Fault Detector")
    
    st.sidebar.info("Select Below to Proceed")
    # InputData='data/'
    # Model_path='results/api_experiment_run/model/'
    
    
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Data_Engg", "Train Tagging Model", "Run Interpretor"])
    if app_mode == "Show instructions":
        st.sidebar.success('Select other options to continue')
        run_instructions()
    elif app_mode == "Data_Engg":
        st.subheader("Dataset Visualization")
        st.sidebar.subheader("Navigation")
        load_the_dataset()
    elif app_mode == "Train Tagging Model":
        st.subheader("Training ML Model")
        st.sidebar.subheader("Navigation")
        run_the_app()
    elif app_mode== "Run Interpretor":
        st.subheader("Interpreting a particular case")
        st.sidebar.subheader("Navigation")
        case_interpretor()


def run_instructions():
    st.write("Documentation will be added")


def load_the_dataset():
    if st.sidebar.checkbox('Load Dataset'):
        data = accept_user_data()
    # Insert Check-Box to show the snippet of the data.
    if st.sidebar.checkbox('Show Data'):
        st.subheader("Data Table")	
        values = st.slider('Select a range of values',1, 20, 5)
        st.write(data[:values]) # Displays the instance of our dataset on the web app
    
    if st.sidebar.checkbox('Visualize Data'):
        st.subheader("Visualize Table")
        data['target'] = data['target'].astype(int)
        data['load'] = data['load'].astype(str)
        # Generate the Profiling Report - Not working
        #profile = ProfileReport(data, title="Gearbox Dataset", html={'style': {'full_width': True}}, sort="None")
        #st.write(profile.to_notebook_iframe())
        #profile = data.profile_report()
        #st.write(profile.html, unsafe_allow_html = True)
        #Histogram
        plt.hist(data['load'].astype(int))
        st.subheader("Load Distribution")
        st.pyplot()

        plt.hist(data['target'].astype(int))
        st.subheader("target Distribution")
        st.pyplot()



def accept_user_data():
    # Select the file from the directory
    rawpath = os.getcwd()
    # print("raw path is",rawpath)
    filenames = os.listdir(rawpath)
    selected_filename = st.selectbox('Select the folder with training documents', filenames)
    st.write('You selected ', selected_filename)
    df = pd.read_csv(os.path.join(rawpath, selected_filename), sep=',')
    return df

def run_the_app():
    if st.sidebar.checkbox('Run Logistic Regression'):
        lin_reg()
    elif st.sidebar.checkbox('Run Random Forest'):
        rand_for()


# Linear Regression Function
def lin_reg():
    # Labels are the values we want to predict
    data = accept_user_data()
    st.subheader("Result")	
    labels = np.array(data['target'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features= data.drop('target', axis = 1)

    # Saving feature names for later use
    feature_list = list(data.columns)
    
    #Using Skicit-learn to split data into training and testing sets
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,random_state = 42)
    
    # Apply logistic regression
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Turn up tolerance for faster convergence
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)
    sparsity = np.mean(clf.coef_ == 0) * 100
    score = clf.score(test_features,test_labels)
    
    # print('Best C % .4f' % clf.C_)
    st.write("Sparsity with L1 penalty: ", sparsity)
    st.write("Test score with L1 penalty: ", score)

    # To calculate the P-Value
    if st.sidebar.checkbox('P-Value'):
        # Breaking the dataset into Target and Features for P- Vales
        st.subheader("P-value Statistics")
        y_target = data['target']
        x_input = data
        x_input = x_input.drop(['target'], axis=1)
        
        X2 = sm.add_constant(x_input)
        est = sm.OLS(y_target, X2)
        est2 = est.fit()
        # st.write(est2.summary())
        plt.rc('figure', figsize=(12, 7))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 0.05, str(est2.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('output.png')
        img = Image.open('output.png')
        st.image(img,caption = "P-value")

# Random Forest
def rand_for():

    data = accept_user_data()
    st.subheader("Result")	
    # Labels are the values we want to predict
    labels = np.array(data['target'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features= data.drop('target', axis = 1)

    # Saving feature names for later use
    feature_list = list(data.columns)
    
    #Using Skicit-learn to split data into training and testing sets
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,random_state = 42)
    

    # Make the random forest classifier
    random_forest = RandomForestClassifier(n_estimators = 100, min_samples_leaf = 3,  n_jobs=-1, max_features = 'sqrt')

    # Train on the training data
    random_forest.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    test_pred = random_forest.predict(test_features)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(test_labels, test_pred)
    st.write(accuracy)


def case_interpretor():
    if st.sidebar.checkbox('Run Random Forest'):
        interpretor()

# Single Case Interpretation
def interpretor():

    data = accept_user_data()
    st.subheader("Result")	
    # Labels are the values we want to predict
    labels = np.array(data['target'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features= data.drop('target', axis = 1)

    # Saving feature names for later use
    feature_list = list(data.columns)
    
    #Using Skicit-learn to split data into training and testing sets
    
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,random_state = 42)
    

    # Make the random forest classifier
    random_forest = RandomForestClassifier(n_estimators = 100, min_samples_leaf = 3,  n_jobs=-1, max_features = 'sqrt')

    # Train on the training data
    random_forest.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    test_pred = random_forest.predict(test_features)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    accuracy = metrics.accuracy_score(test_labels, test_pred)
    st.write(accuracy)
    # To Interpret a single instance
    if st.sidebar.checkbox("Run Explainer"):
        rand_number = st.text_input("Enter a number between 1 -100" )
        if st.button("Submit"):
            i = int(rand_number.title())
            #st.success(i)
            st.subheader("Single Instance")
            st.write(data.iloc[i])
            explainer = lime.lime_tabular.LimeTabularExplainer(train_features, feature_names=feature_list,  discretize_continuous=False)
            exp = explainer.explain_instance(features.iloc[i], random_forest.predict_proba, num_features=18)
            #st.write(exp.as_list())
            # %matplotlib inline
            st.subheader("Factors")
            fig = exp.as_pyplot_figure()
            st.pyplot()
            # Result
            st.subheader("Result")
            exp.save_to_file('oi.html')
            st.write(oi.html, unsafe_allow_html = True)
            # plt.rc('figure', figsize=(12, 7))
            # plt.text(0.01, 0.05, str(show_in_notebook(show_table=True, show_all=False)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
            # plt.axis('off')
            # plt.tight_layout()
            # plt.savefig('outputII.png')
            # img = Image.open('outputII.png')
            # st.image(img,caption = "P-value")


if __name__ == "__main__":
    main()