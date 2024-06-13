import pandas as pd
import streamlit as st
from pathlib import Path
import io
from scipy import stats
import plotly.figure_factory as ff
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


st.set_page_config(layout="wide")


# Title of the app
st.title("E-Learning Student Reactions")

# Subtitle of the app
st.subheader("Hardware & Software for Big Data Mod B")

# Introduction text
st.write(
    """
    This app provides insights and analysis on hardware and software considerations for big data. The data and code used are from the Jupyter notebook file.
    """
)


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data



def data_review(df):
  '''
  Extract information about the dataframe.

  Args:
    df (A Panda's data frame): It contains the data.

  Returns:
    info (A Panda's dictionary): It contains information such as the number of columns, number of rows, and counts of missing and duplicate values in the data.
    datatypes (A list): It contains columns and their respective data types.
  '''

  info = {
      'Number of Rows': df.shape[0],
      'Number of Columns': df.shape[1],
      'Missing Values': df.isnull().sum().sum(),
      'Duplicate Values': df.duplicated().sum(),
  }

  # Create a in-memory buffer to capture the output.
  buffer = io.StringIO()
  df.info(buf=buffer)

  # Get the content of the buffer.
  datatypes = buffer.getvalue()

  return info, datatypes


# data_path = Path('online_classroom_data.csv')
data = load_data('https://raw.githubusercontent.com/iampujan/e-learning-streamlit/main/online_classroom_data.csv')
info, datatypes = data_review(data)

# Create a button to choose between head and tail
choose = st.selectbox('Choose:', ['Head', 'Tail'])

# Create a slider to select the number of rows
num_rows = st.slider('Select number of rows:', min_value=1, max_value=len(data), value=5, step=1)

# Use the button and slider to decide whether to display head or tail
if choose == 'Head':
    if len(data) >= num_rows:
        st.write("### Head Data")
        st.dataframe(data.head(num_rows))
    else:
        st.write(f"The data has fewer than {num_rows} rows.")
else:
    if len(data) >= num_rows:
        st.write("### Tail Data")
        st.dataframe(data.tail(num_rows))
    else:
        st.write(f"The data has fewer than {num_rows} rows.")



# Data Preprocessing
def data_preprocessing(data):
    data.drop(columns=['Unnamed: 0'], inplace=True)
    cols_to_preprocess = ['sk1_classroom', 'sk2_classroom', 'sk3_classroom', 'sk4_classroom', 'sk5_classroom']
    data[cols_to_preprocess] = data[cols_to_preprocess].replace(',', '.', regex=True).astype(float)
    return data

# Perform data preprocessing
preprocessed_data = data_preprocessing(data)

# Display descriptive statistics
st.write("### Descriptive Statistics (after data preprocessing)")
st.write(preprocessed_data.describe())



# Univariate Analysis Histograms and QQ Plots
st.write("### Univariate Analysis")
cols = [
    'total_posts', 'helpful_post', 'nice_code_post', 'collaborative_post',
    'confused_post', 'creative_post', 'bad_post', 'amazing_post',
    'timeonline', 'sk1_classroom', 'sk2_classroom', 'sk5_classroom',
    'sk3_classroom', 'sk4_classroom', 'Approved'
]


hist_fig = sp.make_subplots(rows=3, cols=5, subplot_titles=cols)
qq_fig = sp.make_subplots(rows=3, cols=5, subplot_titles=cols)

for i, col in enumerate(cols):
    row = i // 5 + 1
    col_num = i % 5 + 1
    # Histogram
    hist_fig.add_trace(
        go.Histogram(x=preprocessed_data[col], nbinsx=30, name=col, histnorm='probability density'),
        row=row, col=col_num
    )

    # QQ plot
    qq_data = stats.probplot(preprocessed_data[col].dropna(), dist="norm")
    qq_fig.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name=f'QQ plot of {col}'),
        row=row, col=col_num
    )
    qq_fig.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[0][0]*qq_data[1][0] + qq_data[1][1], mode='lines', name='QQ line'),
        row=row, col=col_num
    )

hist_fig.update_layout(
    height=800, width=1000,
    title_text="Univariate Analysis - Histograms",
    showlegend=False,
)

qq_fig.update_layout(
    height=800, width=1000,
    title_text="Univariate Analysis - QQ Plots",
    showlegend=False,
)

st.plotly_chart(hist_fig, use_container_width=True)
st.plotly_chart(qq_fig, use_container_width=True)


# Count the values in the 'Approved' column
counts = preprocessed_data['Approved'].value_counts()

# Create a pie chart using Plotly
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title='Approved Status Distribution')

# Display the pie chart in Streamlit
st.write("### Approved Status Distribution")
st.plotly_chart(fig, use_container_width=True)


# Bivariate Analysis
# Define the columns representing the features
response = ['total_posts', 'helpful_post', 'nice_code_post', 'collaborative_post',
       'confused_post', 'creative_post', 'bad_post', 'amazing_post',
       'timeonline', 'sk1_classroom', 'sk2_classroom', 'sk5_classroom',
       'sk3_classroom', 'sk4_classroom']

# Extract the target variable into the target DataFrame
target = preprocessed_data['Approved']

# Create a dropdown menu for the target variable
target_var = st.selectbox('Select a target variable:', ['Approved'])

# Create a dropdown menu for the response variable
response_var = st.selectbox('Select a response variable:', response)

# Create a subplot for the BoxPlot
fig = go.Figure(data=[go.Box(x=target, y=preprocessed_data[response_var], boxmean=True)])

# Set the title and layout
fig.update_layout(title=f'{target_var} vs. {response_var}', xaxis_title=target_var, yaxis_title=response_var)

# Display the BoxPlot in Streamlit
st.write("### Bivariate Analysis")
st.plotly_chart(fig, use_container_width=True)

# Pairplot
st.write("### Pairplot")
# Function to create the pairplot
def create_pairplot(data, height, width):
    fig = px.scatter_matrix(
        data,
        dimensions=list(data.columns),
        color_discrete_sequence=[
            '#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#1abc9c', '#f1c40f',
            '#e67e73', '#2c3e50', '#8e44ad', '#1a1d23'
        ]
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins as needed
        width=width
    )
    return fig

# Set default values for height and width in session state
if 'height' not in st.session_state:
    st.session_state.height = 2000
if 'width' not in st.session_state:
    st.session_state.width = 2000

# Create sliders for height and width
st.session_state.height = st.slider('Height:', 600, 2000, st.session_state.height)
st.session_state.width = st.slider('Width:', 600, 2000, st.session_state.width)

# Create the initial pairplot
fig = create_pairplot(preprocessed_data, st.session_state.height, st.session_state.width)

# Display the pairplot
st.plotly_chart(fig, use_container_width=True)



# Pearson Correlation
st.write("### Select the desired height and width for the heatmap")

# Function to create the correlation heatmap
def create_heatmap(data, response, height, width):
    # Calculate the correlation matrix between response variables
    corr_matrix = data[response].corr()

    # Round the correlation matrix to one decimal place (Seaborn's number system)
    corr_matrix_rounded = corr_matrix.round(1)

    # Convert the correlation matrix to a list for Plotly heatmap
    corr_matrix_list = corr_matrix_rounded.values.tolist()
    x_labels = corr_matrix_rounded.index.tolist()
    y_labels = corr_matrix_rounded.columns.tolist()

    # Create Plotly figure for heatmap
    fig = ff.create_annotated_heatmap(
        z=corr_matrix_list,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',  # Choose your color scale here
        showscale=True,
    )

    fig.update_layout(
        title='Correlation Matrix Heatmap (Seaborn Number System)',
        xaxis=dict(side='bottom'),  # Position x-axis at the bottom
        width=width,  # Adjust as needed
        height=height,  # Adjust as needed
        margin=dict(t=100),  # Increase top margin for title spacing and left margin for title placement

    )
    return fig

# Set default values for height and width
default_height = 600
default_width = 800

# Create sliders for height and width
height = st.slider('Height:', 400, 1000, default_height)
width = st.slider('Width:', 400, 1200, default_width)

# Button to generate the heatmap
if st.button('Generate Pearson Correlation Heatmap'):
    # Create the heatmap
    fig = create_heatmap(preprocessed_data, response, height, width)
    
    # Display the heatmap
    st.plotly_chart(fig)

