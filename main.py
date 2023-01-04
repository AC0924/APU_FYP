import pickle
import pandas as pd
import numpy as np
import streamlit as st 

# For aesthetic design
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
 
# loading the trained model
with open('final_model.pkl', 'rb') as train_model:
    classifier = pickle.load(train_model)


# loading the scaler
with open('scaler.pkl', 'rb') as pickle_in_s:
    scaler = pickle.load(pickle_in_s)
 
@st.cache()

def load_data(dataset):
    data = pd.read_csv(dataset)
    return data

def get_value(val,my_dict):
    for key ,value in my_dict.items():
        if val == key:
            return value

State_label = {'Alaska': 0.209, 'Alabama': 0.241,  'Arkansas': 0.235,  'Arizona': 0.3,  'California': 0.268,  'Colorado': 0.249,  
                'Connecticut': 0.191,  'District of Columbia': 0.278,  'Delaware': 0.25,  'Florida': 0.384,  'Georgia': 0.336,  'Hawaii': 0.193,  
                'Iowa': 0.16,   'Idaho': 0.209,  'Illinois': 0.307,  'Indiana': 0.224,  'Kansas': 0.188,  'Kentucky': 0.223,  
                'Louisiana': 0.255,  'Massachusetts': 0.179,  'Maryland': 0.263,  'Maine': 0.124,  'Michigan': 0.293,  'Minnesota': 0.153,  
                'Missouri': 0.21,   'Mississippi': 0.218,  'Montana': 0.107,  'North Carolina': 0.255,  'North Dakota': 0.105,  'Nebraska': 0.139,  
                'New Hampshire': 0.155,  'New Jersey	': 0.281,  'New Mexico	': 0.17,  'Nevada': 0.312,  'New York': 0.261,  'Ohio': 0.208,  
                'Oklahoma': 0.203,  'Oregon': 0.216,  'Pennsylvania': 0.195,  'Rhode Island': 0.154,  'South Carolina': 0.295,  'South Dakota': 0.113, 
                'Tennessee': 0.308,  'Texas': 0.253,  'Utah': 0.243,  'Virginia': 0.256,  'Vermont': 0.11,  'Washington': 0.215, 
                'Wisconsin': 0.162,  'West Virginia': 0.199,  'Wyoming': 0.111}

Sector_label    = {'Agriculture, Forestry, Fishing & Hunting': 0.121, 'Mining, Quarying, Oil & Gas': 0.116,
                   'Utilities': 0.216, 'Constuction': 0.31, 'Manufacturing': 0.266, 'Manufacturing': 0.226, 'Manufacturing': 0.192,
                   'Wholesale Trade': 0.272, 'Retail Trade': 0.301, 'Retail Trade': 0.307, 'Transportation & Warehousing': 0.349,
                   'Transportation & Warehousing': 0.314, 'Information': 0.324, 'Finance & Insurance': 0.373, 
                   'Real Estate, Rental & Leasing': 0.391, 'Professional, Scientific & Technical Service': 0.267,
                   'Management of Companies & Enterprise': 0.159, 'Administrative, Support, Waste Management & Remediation Service': 0.316,
                   'Educational Service': 0.313, 'Health Care & Social Assistance': 0.144, 'Arts, Entertainment & Recreation': 0.277, 
                   'Accomodation & Food Service': 0.287, 'Other Services': 0.265, 'Public Administration':0.237}

Context = """This project is about the loan default prediction of the U.S. Small Business. The prediction of the loan default probability is based on several parameters that have much siginificant impact on the loan repayment status.
            The developer has build several models and chose the best performing model, then model tuning was conducted in order to further optimize the model to achieve better performance.
            In order to ensure the accuracy of this prediction system, the developer implemented the tuned XGBoost model as it is the best performing model among other."""

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Term, Portion, Retained, RevLineCr, UrbanRural, RealEstate, DisGross, GrAppv, Sector, State, CreateJob):   
    
    
    # Pre-processing user input    
    if Retained == "Yes":
        Retained = 1
    elif Retained == "No":
        Retained = 0
 
    if RevLineCr == "Yes":
        RevLineCr = 1
    elif RevLineCr == "No":
        RevLineCr = 0
 
    if UrbanRural == "Urban":
        UrbanRural = 1
    elif UrbanRural == "Rural":
        UrbanRural = 2
    elif UrbanRural == "Undefined":
        UrbanRural = 0
 
    if RealEstate == "Yes":
        RealEstate = 1
    elif RealEstate == "No":
        RealEstate = 0
 
    if CreateJob == "Yes":
        CreateJob = 1
    elif CreateJob == "No":
        CreateJob = 0

    
    input_data = [Term, Portion, Retained, RevLineCr, UrbanRural, RealEstate, DisGross, GrAppv, Sector, State, CreateJob]
    
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    prediction = classifier.predict(input_data)
    predict_proba = classifier.predict_proba(input_data)[:,1]
    result = (str((np.around(float(predict_proba),3) * 100)) + '%')
                
    st.subheader('The Probability This Loan Will Be Default is: ' + result)

    if prediction == 0:
        pred = 'Not Default'
    else:
        pred = 'Default'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # Front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:orange;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Loan Default Prediction</h1> 
    </div> 
    """
    # Display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    st.write('By: Adrian Chen Yi Jie')
    # Menu
    menu = ['Prediction', 'About']
    choice = st.sidebar.selectbox('Select Activities', menu)

    if choice == 'Prediction':
        st.write('')
        st.subheader('Basic Business Information')
        Sector    = st.selectbox('What is the business sector?', tuple(Sector_label.keys()))
        State     = st.selectbox('Where is the business located?', tuple(State_label.keys()))
        UrbanRural= st.radio('Where the business locate?', ['Urban', 'Rural', 'Undefined'])
        st.write('')
        st.subheader('Loan Info')
        GrAppv    = st.number_input("How much is the loan approved (USD)?")
        DisGross  = st.number_input("How much the amount disbursed (USD)?")
        Portion   = st.slider("Portion SBA Guarantee?", 0.0,1.0)
        Term      = st.slider("How Long Is The Loan? (Month)", 60, 300 )
        RevLineCr = st.radio('Is it revolving line of credit?', ['Yes', 'No'])
        Retained  = st.radio('Is the employee retained?', ['Yes', 'No'])
        CreateJob = st.radio('Is job created?', ['Yes', 'No'])
        RealEstate= st.radio('Is the loan backed by real estate?', ['Yes', 'No'])
        

        V_State     = get_value(State, State_label)
        V_Sector    = get_value(Sector, Sector_label)

        if st.button("Predict"): 
            result = prediction(Term, Portion, Retained, RevLineCr, UrbanRural, RealEstate, DisGross, GrAppv, V_Sector, V_State, CreateJob) 
            # re = format(result)
            # st.success('Prediction Result: {}'.format(re))
            if(result == "Not Default"):
                st.success('Prediction Result: {}'.format(result))
            elif(result == "Default"):
                st.warning("Prediction Result: {}".format(result))
            # st.success('Prediction Result:  {}'.format(result))
    
    if choice == 'About':
        st.header('About This Project')
        st.write(Context)
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write("""Disclaimer: The information provided by this prediction system is for project demonstration purposes only.
         All information on this site is provided in good faith, however we make no representation or warranty of any kind, express or implied,
         regarding the accuracy, adequacy, validity, reliability and avaialability, or completeness of any information on the site.
         """)

def layout(*args):
    style = """
    """

    style_div = styles(
        # position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        margin=px(8, 8, "auto", "auto"),
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        
        "\n© 2022 Adrian Chen Yi Jie. All rights reserved. ",
        
    ]
    layout(*myargs)


if __name__=='__main__': 
    main()
    footer()

