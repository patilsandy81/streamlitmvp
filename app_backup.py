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

from streamlit import caching
caching.clear_cache()


@st.cache
def run_instructions():
    st.write("Documentation will be added")

def accept_user_data():
    # Select the file from the directory
    rawpath = os.getcwd()
    # print("raw path is",rawpath)
    filenames = os.listdir(rawpath)
    
    #trainRawData = file_selector()
    st.write('You selected `%s`' % selected_filename)
    df = pd.read_csv(os.path.join(folder_path, selected_filename))
    return (df)


def load_the_dataset():
    #if st.sidebar.checkbox('Load Data File'):
    #    files_path = accept_user_data()




#def run_the_app(trainData_path,models_path):
    
    

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
        run_the_app(InputData,Model_path)
    elif app_mode== "Run Interpretor":
        st.subheader("Interpreting a particular case")
        st.sidebar.subheader("Navigation")
        case_interpretor(InputData)


if __name__ == "__main__":
    main()