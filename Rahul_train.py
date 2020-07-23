

import os, urllib
import pandas as pd
import argparse
from startprocess import OCR_Service
from startprocess1 import OCR_Service1
import yaml
from ludwig.api import LudwigModel
from ludwig import visualize
import pandas as pd
import shutil
import boto3
import json
from startprocess import OCR_Service
from jsontocsv import savetable
import math
from trp import Document
import re
import streamlit as st
from nltk.corpus import stopwords
from PyPDF2 import PdfFileWriter, PdfFileReader, PdfFileMerger
import glob2
import time
from sTables import stExtract
import subprocess
import csv

from streamlit import caching
caching.clear_cache()



from PIL import Image

# st.image(image, caption='', 
#          use_column_width=True, 
# #          width= 1800, 
#          clamp=False, 
#          channels='RGB')

# st.sidebar.image(image1, caption='', 
#          use_column_width=True, 
# #          width= 300, 
#          clamp=False, 
#          channels='BGR')


def split(inputpdf, path, rel):
    inputpdf = PdfFileReader(open(inputpdf, "rb"))

    for i in range(inputpdf.numPages):
        output = PdfFileWriter()
        for j in rel:
            if j==i:
                output.addPage(inputpdf.getPage(i))
                with open(path+"split/"+str(i)+'.pdf', "wb") as outputStream:
                    output.write(outputStream)
                    
def metadata(inputpdf, path, rel):
    inputpdf = PdfFileReader(open(inputpdf, "rb"))

    for i in range(inputpdf.numPages):
        output = PdfFileWriter()
        if rel==i:
            output.addPage(inputpdf.getPage(i))
            with open(path+"metadata/"+str(i)+'.pdf', "wb") as outputStream:
                output.write(outputStream)
        

def clean(text_out):
    result4 = re.sub("[^a-zA-Z]"," ", text_out)
#     print (result3)
    punctuations ="""#'''!()-[]{},'"\"<>?@#%^&_~|"""

    my_str = result4

    # To take input from the user
    # my_str = input("Enter a string: ")

    # remove punctuation from the string
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
        else: 
            no_punct=no_punct + " "
    result5=re.findall('\w+',no_punct)
    s = " ".join(result5)

    return (s)


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     stems = []
#     for item in tokens:
#         stems.append(PorterStemmer().stem(item))
#     return stems

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def folder_structure_to_csv(path: str, save_path: str):
    text_data = []
    labels = []
    names=[]
    for folder in os.listdir(path):
        for image in os.listdir(path+'/'+folder):
            file=(path+folder+'/'+image)
            text_json=OCR_Service(file)
#             st.write(image+'---> Converted to text')
            text_data.append(text_json['file'][0]['text'])
            labels.append(folder)
            names.append(image)

    df = pd.DataFrame({'name': names,'text': text_data,'class': labels})
    df.to_csv(save_path, index=None)
    return ('Done!')


@st.cache
def loadData(path):
    df = pd.read_csv(path)
    return (df)

def accept_user_data():

    trainRawData = file_selector()
    st.write('You selected `%s`' % trainRawData)
    return (trainRawData+'/')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select the folder with training documents="traindata"', filenames)
    return os.path.join(folder_path, selected_filename)



def main():
    
    st.sidebar.title("Patented AI Engine")
    
    st.sidebar.info("Select Below to Proceed")
    InputData='data/'
    Model_path='results/api_experiment_run/model/'
    
    
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Train Tagging Model", "Run Tagging Predictions", "Run Statement Processor"])
    if app_mode == "Show instructions":
        st.sidebar.success('Select other options to continue')
        run_instructions()
    elif app_mode == "Train Tagging Model":
        run_the_app(InputData,Model_path)
    elif app_mode == "Run Tagging Predictions":
        st.subheader("Run Predictions")
        st.sidebar.subheader("Navigation")
        run_the_predictions(InputData)
    elif app_mode== "Run Statement Processor":
        st.subheader("Statement Processor")
        st.sidebar.subheader("Navigation")
        statement_processor(InputData)
        
def run_the_app(trainData_path,models_path):
    
    
    st.subheader("Training ML Model")
    st.sidebar.subheader("Navigation")
        
    if st.sidebar.checkbox('Load Raw Files'):
        files_path = accept_user_data()
    
    if st.sidebar.button('Run OCR'):
        st.write('In Process')
        text_data = []
        labels = []
        names=[]
        for folder in os.listdir(files_path):
            for image in os.listdir(files_path+'/'+folder):
                file=(files_path+folder+'/'+image)
                text_json=OCR_Service(file)
                text_raw=[]
                text_raw= [clean((pages['text'])) for pages in text_json['file']]
                comp=["".join(normalize(text_raw))]
                text_data.append(comp[0])
                labels.append(folder)
                names.append(image)
            df = pd.DataFrame({'name': names,'text': text_data,'class': labels})
            df.to_csv(trainData_path+'train.csv', index=None)
        st.success('Done!')

    # Function call to Load the dataset
    
    
    if st.sidebar.checkbox('Load Processed Data'):
        data = loadData(trainData_path+'train.csv')
        st.write('Loading Done')
      
        
        
        # Insert Check-Box to show the snippet of the data.
        if st.sidebar.checkbox('Show Data'):
            st.subheader("Data Table")	
            values = st.slider('Select a range of values',1, 20, 5)
            st.write(data[:values]) # Displays the instance of our dataset on the web app
            df=data
        



        if st.sidebar.checkbox('Show Parameters'):
            config  = yaml.load(open(trainData_path+'model_definition-rnn.yaml', 'r'))
            model_definition=config
            st.subheader("Hyperparameters")
            st.write(model_definition)



        if st.sidebar.button('Start Training'):

            print('creating model')
            model = LudwigModel(model_definition)
            print('training model')
            shutil.rmtree('results', ignore_errors=True)
            train_stats = model.train(
                                      data_df=df,
                                      experiment_name='api_experiment',
                                      model_name='run',
                                      output_directory='results',
                                        )
            st.subheader('Training Done')
        if st.sidebar.checkbox('Show Training Stats'):
            training_statistics = 'results/api_experiment_run/training_statistics.json'
            training_metadata = 'results/api_experiment_run/model/train_set_metadata.json'
            subprocess.run('ludwig visualize --visualization learning_curves --training_statistics '+training_statistics+' --output_directory '+trainData_path+' --file_format png', 
                           shell=True, 
                           capture_output=True, 
                           check=True)
            image = Image.open(trainData_path+'learning_curves_combined_loss.png')
            st.image(image, caption='',use_column_width=True)
#             image1 = Image.open(trainData_path+'learning_curves_combined_accuracy.png')
#             st.image(image1, caption='',use_column_width=True)
            st.subheader('')


def run_the_predictions(InputData):
    ludwig_model = LudwigModel.load('results/api_experiment_run/model/')
    
    if st.sidebar.checkbox('Load Test Documents'):
        trainRawData = file_selector()
        st.write('You selected `%s`' % trainRawData)
        testData_path=InputData
        
    if st.sidebar.button('Run OCR'):
        st.write('In Process')
        text_data = []
        labels = []
        names=[]
        for folder in os.listdir(trainRawData):
            for image in os.listdir(files_path+'/'+folder):
                file=(files_path+folder+'/'+image)
                text_json=OCR_Service(file)
                text_raw=[]
                text_raw= [clean((pages['text'])) for pages in text_json['file']]
                comp=["".join(normalize(text_raw))]
                text_data.append(comp[0])
                labels.append(folder)
                names.append(image)
            df = pd.DataFrame({'name': names,'text': text_data,'class': labels})
            df.to_csv(testData_path+'train.csv', index=None)
        st.success('Done!')
    
    uploaded_file = st.file_uploader('Choose a "test.csv" file to load processed data", type="csv"')
    if uploaded_file is not None:
        if st.sidebar.checkbox("Load processed Data"):
            data = pd.read_csv(uploaded_file)
            st.subheader('Loading Done')
        
            if st.sidebar.checkbox("Show Processed Data"):
                st.subheader('Processed Test Data')
                values1 = st.slider('Select a range of values',1, 20, 18)
                st.write(data[:values1]) #'data/train.csv'
                st.subheader('')

            if st.sidebar.button('Predict'):
                predictions = ludwig_model.predict(data_df=data)
                pred=pd.DataFrame({'name': data['name'], 
                                    'document': predictions['class_predictions'], 
                                    'probabilties': predictions['class_probability']})
                st.subheader('Prediction Results')
                st.write(pred.loc[:values1].style.background_gradient(cmap='viridis'))
        #         st.write(pred)

def statement_processor(InputData):
    
    files_path = accept_user_data()
    
    if st.sidebar.button('Run OCR'):
        st.write('In Process')
        page_no=[]
        clean_train_Docs_Match=[]
        names=[]
        for folder in os.listdir(files_path):
            for image in os.listdir(files_path+'/'+folder):
                file=(files_path+folder+'/'+image)
                text_json =OCR_Service(file)
                for pages in text_json['file']:
                    text_data=clean(pages['text'])
                    norm_corpus=normalize([text_data])
                    clean_train_Docs_Match.append(norm_corpus[0])
                    page_no.append(pages['page'])
                    names.append(image)
            df = pd.DataFrame({'text': clean_train_Docs_Match, 'page': page_no, 'name': image})
            df.to_csv(InputData+'statement.csv', index=None)
            st.success('Done!')
        
    if st.sidebar.checkbox('Load statement'):      
        data = pd.read_csv(InputData+'statement.csv')
        st.subheader('Loading Done')
        
        if st.sidebar.checkbox("Show Processed Data"):
            st.subheader('Processed Test Data')
            values2 = st.slider('Select a range of values',1, 20, 18)
            st.write(data[:values2]) #'data/train.csv'
            st.subheader('')
        if st.sidebar.checkbox("Find relevant pages"):
            search_string = ["Holdings","Quantity","Price","Total Cost Basis"]
            rel=[]
            for i in range(0,int(len(data.index)/2)):
                count=0
                
                for string in search_string:
#                     st.write(string)
                    if re.search(string,data['text'][i],re.IGNORECASE):
#                         st.write(data['text'][i])
                        count=count+1
#                         st.write(count)
                        if count==4:
                            rel.append(i+1)
            st.write(rel)
                        
            if st.sidebar.checkbox("Split to Relevant Pages"):
                pdf_path=glob2.glob(files_path+'/statements/'+'*.pdf')
                st.write('Required pages saved at "data/split/"' )
                st.subheader('' )
                split(pdf_path[0],InputData,rel)
                
                if st.sidebar.checkbox("Display input page"):
                    image = Image.open(InputData+'split/'+'input.png')
                    st.image(image, caption='',use_column_width=True)
    
                if st.sidebar.checkbox("Extract Holdings"):
                    files=glob2.glob(InputData+'split/'+'*.pdf')
#                     stExtract(files[2])
                    
                if st.sidebar.checkbox('Show extracted holdings'):
                    dft=pd.read_csv('input-png-page-1Table0.csv')
                    st.write(dft)
                    
    if st.sidebar.checkbox('Extract metadata'):
        pdf_path=glob2.glob(files_path+'/statements/'+'*.pdf')
        st.write('Required pages saved at "documents/metadata/"' )
        st.subheader('' )
        rel=1
        metadata(pdf_path[0],InputData,rel)
        files=glob2.glob(InputData+'metadata/'+'*.pdf')
        data_json, img = OCR_Service1(files[0])
        image = img.save('input.png')
        documentName='input.png'
        
        if st.sidebar.checkbox("Display metadata page"):
            image = Image.open('input.png')
            st.image(image, caption='',use_column_width=True)
#         with open(documentName, 'rb') as document:
#             imageBytes = bytearray(document.read())


#         # Call Amazon Textract
#         # Amazon Textract client
#         textract = boto3.client('textract')
#         response = textract.analyze_document(Document={'Bytes': imageBytes}, 
#                                          FeatureTypes=["TABLES"])
        
#         print("\nText\n========")

#         text = ""
#         for item in response["Blocks"]:
#             if item["BlockType"] == "LINE":
#                 text = text + " " + item["Text"]
       


#         # Amazon Comprehend client
#         comprehend = boto3.client('comprehend')
        
#         # Detect entities
#         entities =  comprehend.detect_entities(LanguageCode="en", Text=text)
#         print("\nEntities\n========")
#         with open(InputData+'metadata/'+'entities.csv', "w") as csvFile:
#             fieldnames = ['Entity Name','Entity Type']
#             writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
#             writer.writeheader()
#             for entity in entities["Entities"]:
#                 print ("{}\t=>\t{}".format(entity["Type"], entity["Text"]))
#                 writer.writerow({'Entity Type': entity["Type"],'Entity Name': entity["Text"] })

    if st.sidebar.checkbox("Show extracted metadata"):
        dft=pd.read_csv(InputData+'metadata/'+'entities.csv')
        org= (dft[dft['Entity Type']=='ORGANIZATION'].iloc[0])
        per= (dft[dft['Entity Type']=='PERSON'].iloc[0])
        sdate= (dft[dft['Entity Type']=='DATE'].iloc[0])
        edate= (dft[dft['Entity Type']=='DATE'].iloc[1])
        df_entities = {'Bank Name': org['Entity Name'],
                   'Account Owner': per['Entity Name'],
                   'Statement Start Date': sdate['Entity Name'],
                   'Statement End Date': edate['Entity Name'],
                  }
        st.write(df_entities)
     
    
def run_instructions():
    st.write("Documentation will be added")

if __name__ == "__main__":
    main()

