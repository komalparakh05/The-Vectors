import streamlit as st
import pandas as pd

def FileTypeEntry():
    st.header("Loan Prediction")
    file=st.file_uploader("uploas file", type=["csv"])
    show_file=st.empty()
    if not file:
        show_file.info("Please upload a file".format(''.join(["csv"])))
        return
    data=pd.read_csv(file)
    st.dataframe(data.head())
    file.close()
    if st.button("Predict"):
        st.text("predicted")#put you fast api response
def Indiviual_entry():
    product_list=[]
    product_list.append(st.selectbox("Gender",["Male","Female"]))
    product_list.append(st.selectbox("Credit_History",["1","0"]))
    product_list.append(st.selectbox("Education",["Graduate","Non-Graduate"]))
    product_list.append(st.selectbox("Married",["Yes","No"]))
    product_list.append(st.selectbox("Self_Employed",["Yes","No"]))
    product_list.append(st.selectbox("Property_Area",["Urban","Semi-Urban","Rural"]))
    df = pd.DataFrame([product_list], columns=['product_name', 'price','fr','frf','rv','rf'])
    if st.button("Predict"):
        st.text("predicted")
if __name__=="__main__":
    #main()
    Select_type_entry=st.sidebar.selectbox("Select the type of input",["File Type","Indiviual Entry"])
    if(Select_type_entry=="File Type"):
        FileTypeEntry()
    else:
        Indiviual_entry()
