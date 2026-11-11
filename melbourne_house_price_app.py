import streamlit as st 
import pickle
import pandas as pd 
import os 

st.set_page_config(page_title = 'Melbourne House Price')

mel = pd.read_csv('clean_data.csv')

def load_model ():
    with open('melbourne_house_price_model.pkl', 'rb') as f:
        return pickle.load(f)


st.header("Melbourn house Predication App ")
with st.sidebar:
    st.image('PU.jpg')
    st.text("Student Name: Ngai Yaw")
    st.text("Student ID: PIUS20240087")
    st.text("Class of 2028")



def show_picture (file_name):
    if os.path.exists(file_name):
        st.image(file_name)

    else:
        st.info("Picture space: add" + file_name + "to the project folder")


st.header ("Melbourne House Price Prediction App")

show_picture ('banner.jpg')

st.write("Choose the details of the house and press the button to see the predicted price.")

col1, col2 = st.columns(2)

with col1:
    # all the suburbs of the dataset
    var1 = st.selectbox("Enter the suburb", sorted(mel['Suburb'].unique()))
    
    var2 = st.number_input("Enter number of rooms",
                          min_value = int(mel['Rooms'].min()),
                           max_value = int(mel['Rooms'].max()),
                         value = 3 )

    var3 = st.selectbox("Enter the house type (h = house, t = townhouse, u = unit)",
                        mel['Type'].unique())
    
    var4 = st.slider ("Enter distance from the city center (Km)",

                      min_value = float(mel['Distance'].min()),
                      max_value = float (mel['Distance'].max()),
                      value = float(mel['Distance'].min()), 
                      step = 0.5)
    
    var5 = st.number_input("Enter number of bedrooms",
                          min_value = int(mel['Bedroom'].min()),
                          max_value = int(mel['Bedroom'].max()),
                          value = 5)
with col2:
    
    var6 = st.number_input("Enter number of bathrooms",
                          min_value = int(mel['Bathroom'].min()),
                          max_value = int(mel['Bathroom'].max()),
                          value = 4)
    
    var7 = st.number_input("Enter number of cars",
                          min_value = int(mel['Car'].min()),
                          max_value = int(mel['Car'].max()),
                          value = 3)

    # biggest 1 % of the land sizes and building areas are cut off to use sliders  easily
    var8 = st.slider ("Enter land size (m2)",
                      min_value = 0,
                      max_value  = int(mel['Landsize'].quantile(0.99)),
                      value = int(mel['Landsize'].median()),
                      step = 10 )
                                               
                         
        
    var9 = st.slider("Enter the building area (m2)",
                     min_value = int(mel['BuildingArea'].min()),
                     max_value = int(mel['BuildingArea'].max()),
                     value = int(mel['BuildingArea'].median()),
                     step = 5)
                        
        
    var10 = st.slider ("Enter the year built",
                       min_value = int(mel['YearBuilt'].min()),
                       max_value = int(mel['YearBuilt'].max()),
                       value = int(mel['YearBuilt'].median()),
                       step = 1)
        
    # every suburb belongs to one region, so the region is taken from the data
    var11 = mel[mel['Suburb'] == var1]['Regionname'].mode()[0]
    st.info(f"Region (found from the suburb): {var11}")
        

predict_button  = st.button("Predict House Price")
        
if predict_button:
    input_data = pd.DataFrame(
        [
            {
                "Suburb": var1,
                "Rooms": var2,
                "Type": var3,
                "Distance": var4,
                "Bedroom": var5,
                "Bathroom": var6,
                "Car": var7,
                "Landsize": var8,
                "BuildingArea": var9,
                "YearBuilt": var10,
                "Regionname": var11
            }
        ]
    )

    model = load_model()       
    result = model.predict(input_data)[0]
    result = max(result, 0)
        
    st.write(f"Predicted House Price: ${result:,.0f}")
    st.caption("On the test data, the model is usually off by about $229,000 (MAE).")
    # pictures for a cheap, median, and high price 
    low_price = mel['Price'].quantile(0.25)
    high_price = mel['Price'].quantile(0.75)                    


    if result < low_price:
        st.write("This is a cheaper house.")
        show_picture('cheap.jpg')
    elif result < high_price:
        st.write("This is a house with a median price.")
        show_picture('median.jpg')
    else:
        st.write("This is an expensive house.")
        show_picture('expensive.jpg')

with st.expander("Dataset"):
    st.dataframe(mel.head(100))
