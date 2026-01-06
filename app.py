import pandas as pd
import numpy as np
import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score

#logger 

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{(timestamp)} {message}")

# Session state Initialization 

if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False
    
# Folder Setup
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR,"data","raw")
CLEANED_DIR = os.path.join(BASE_DIR,"data","cleaned")

os.makedirs(RAW_DIR,exist_ok=True)
os.makedirs(CLEANED_DIR,exist_ok=True)

log("Application Started")
log(f"RAW_DIR = {RAW_DIR}")
log(f"CLEANED_DIR = {CLEANED_DIR}")

#Page config
st.set_page_config("End-to-End SVM", layout = "wide")
st.title("End-to-End SVM platform")

#Sidebar: Model Settings
st.sidebar.header("SVM settings")
#kernels
kernel= st.sidebar.selectbox("Kernel",["linear","rbf","poly","sigmoid"])
c = st.sidebar.slider("c (Regularization Factor)" , 0.01,10.0,1.0)
gamma = st.sidebar.selectbox("Gamma",["scale","auto"])

log(f"SVM settings ----> kernel = {kernel}, c= {c}, Gamma ={gamma}")

# Step 1 : Data Ingestion
st.header("Step 1 : Data Ingestion")
log("Step 1 started : Data Ingestion")
option = st.radio("Choose Data Source ",[ "Download Dataset","Upload CSV"])

df = None
raw_path = None
if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris Dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)
        
        raw_path = os.path.join(RAW_DIR,"iris.csv")
        with open(raw_path,"wb") as f:
            f.write(response.content)
            
            
        df=pd.read_csv(raw_path)
        st.success("Dataset Downloaded successfully")
        log(f"iris dataset saved at {raw_path}")
        
        
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV File",type=["csv"])
    if uploaded_file:
        raw_path=os.path.join(RAW_DIR,uploaded_file)
        with open(raw_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        df=pd.read_csv(raw_path)
        st.success("File Uploaded successfully")
        log(f"Uploaded data saved at {raw_path}")
        
        
# Step 2 : EDA
if df is not None:
    st.header("Step 2 : Exploratory Data Analysis")
    log("Step 2 started : EDA")
    
    st.dataframe(df.head())
    st.write("Shape",df.shape)
    st.write("Mising Values:",df.isnull().sum())
    
    fig,ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only = True),annot = True,cmap = "coolwarm",ax=ax)
    st.pyplot(fig)
    
    log("EDA completed")
    
# Step 3: Data Cleaning
if df is not None:
    st.header("Step 3 : Data Cleaning")
    
    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean","Median","Drop Rows"]
    )
    
    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
        
    else:
        for col in df_clean.select_dtypes(include = np.number):
            if strategy == "Mean":
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    st.session_state.df_clean = df_clean
    st.success("Data cleaning completed")
    
else:
    st.info("please complete step 1 (Data Ingestion) first...")
    
# step 4 : Save the Cleaned data

if st.button("Save cleaned Dataset"):
        if st.session_state.df_clean is None:
            st.error("No cleaned data found . Please complete step 3 first....")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_filename = f"cleaned_dataset_{timestamp}.csv"
            clean_path = os.path.join(CLEANED_DIR,clean_filename)
            
            st.session_state.df_clean.to_csv(clean_path,index=False)
            st.success("cleaned dataset saved")
            st.info(f"Saved at : {clean_path}")
            log(f"Cleaned data saved at {clean_path}")
            
#step 5: Load cleaned dataset
st.header("Step 5 : Load cleaned dataset")
clean_files = os.listdir(CLEANED_DIR)   
if not clean_files:
    st.warning("No cleaned datasets found. Please save one in step 4")
    log("No cleaned dataset available")
else:
    selected = st.selectbox("Select Cleaned DataSet ",clean_files)
    df_model = pd.read_csv(os.path.join(CLEANED_DIR,selected))
    st.success(f"Loaded Dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")
    st.dataframe(df_model.head())
    
# step 6 :Train SVM

st.header("Step 6 : Train SVM")
log("Step 6 started : SVM Training")
target = st.selectbox("Select Target Column",df_model.columns)

y=df_model[target]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)
    log("Target Column encded")
    
# Select numerical features only

x = df_model.drop(columns = [target])
x = x.select_dtypes(include = np.number)

if x.empty:
    st.error("No numeric features available for training..")
    st.stop()
    
#Scale features

scaler = StandardScaler()
x=scaler.fit_transform(x)

#Train-test-split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
model=SVC(kernel=kernel,C=c,gamma=gamma)
model.fit(x_train,y_train)
#Evaluation metrics
y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)
st.success(f"Accuracy: {acc:.2f}")
log(f"SVM Trained Succesfully | Accuracy = {acc:.2f}")

cm=confusion_matrix(y_test,y_pred)
fig,ax=plt.subplots()
sns.heatmap(cm,annot=True, fmt="d" , cmap="coolwarm",ax=ax)
st.pyplot(fig)