import streamlit as st
import pickle
import pandas as pd

import plotly.graph_objects as go
import numpy as np



# ******** Getting function from the model **********
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# **************** SIDEBAR CONTENT ***************
# In add_sidebar
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    # call the copied function:
    data = get_clean_data()

    # Define the labels
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    # Loop through all these labels and create a dictionary
    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
       )       
    return input_dict



# ********** Scaling the values **********
def get_scaled_values(input_dict):
    data = get_clean_data()

    # take the predictors only
    X = data.drop(['diagnosis'], axis=1)
  
    # Create the scaled dictionary
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val) # 0 and 1 for every single value
        scaled_dict[key] = scaled_value

    return scaled_dict


# ********* Displaying the radar plot **********
def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  # categories around the chart
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], 
          input_data['symmetry_mean'], input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], 
          input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
          input_data['concavity_se'], input_data['concave points_se'], 
          input_data['symmetry_se'], input_data['fractal_dimension_se']        
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], 
          input_data['symmetry_worst'], input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1] # for scaled data the range is between 0 and 1
      )),
    showlegend=True
  )
  
  return fig


# ******************** PREDICTIONS *********************
def add_predictions(input_data):
    # Load and open the model and scaler files
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    # UPDATED
    # Make sure that input_data includes feature names as keys
    #input_data = get_scaled_values(input_data)

    # converting input_data values into an array and reshpe into 1 rows with all cols
    input_array = np.array(list(input_data.values())).reshape(1,-1)

    # scaling the array data:
    input_array_scaled = scaler.transform(input_array)

    # predicting the values
    prediction = model.predict(input_array_scaled)

    # making it more userfriendly
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is: ")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    # Probability
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionnals in making a diagnosis, but should not be used as a substitue for a professional diagnosis.")



# ************* SETTING PAGE CONFIGURATION **************
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
        )
    

    # css style: import the file into the app
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

   
    input_data = add_sidebar()

    # ****** Set up the structure ********
    if input_data is not None:
        with st.container():
            st.title("Breast Cancer Predictor")
            st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

        # Creating 2 columns: chart and prediction box
            col1, col2 = st.columns([4,1])
            with col1:
                radar_chart = get_radar_chart(input_data) 
                st.plotly_chart(radar_chart)

            with col2:
                add_predictions(input_data)
    else:
        st.write("Input data is None. There may be an issue with add_sidebar.")

if __name__ == '__main__':
    main()
