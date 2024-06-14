import pandas as pd
import streamlit as st
from pathlib import Path
import io
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.figure_factory as ff
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")

# Title of the app
st.title("E-Learning Student Reactions")

# Subtitle of the app
st.subheader("Hardware & Software for Big Data Mod B")

# Introduction text
st.write(
    """
    The primary research question aims to explore the relationship between the types of reactions students received and their skill levels. Specifically, the goal is to determine if there is any significant correlation between the reactions (e.g., 'Amazing post', 'Confusing post') and the skills demonstrated by the students.
    """
)


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def data_review(df):
    """
    Extract information about the dataframe.

    Args:
      df (A Panda's data frame): It contains the data.

    Returns:
      info (A Panda's dictionary): It contains information such as the number of columns, number of rows, and counts of missing and duplicate values in the data.
      datatypes (A list): It contains columns and their respective data types.
    """

    info = {
        "Number of Rows": df.shape[0],
        "Number of Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicate Values": df.duplicated().sum(),
    }

    # Create a in-memory buffer to capture the output.
    buffer = io.StringIO()
    df.info(buf=buffer)

    # Get the content of the buffer.
    datatypes = buffer.getvalue()

    return info, datatypes


# data_path = Path("online_classroom_data.csv")
data = load_data('https://raw.githubusercontent.com/iampujan/e-learning-streamlit/main/online_classroom_data.csv')
info, datatypes = data_review(data)

# Create a button to choose between head and tail
choose = st.selectbox("Choose:", ["Head", "Tail"])

# Create a slider to select the number of rows
num_rows = st.slider(
    "Select number of rows:", min_value=1, max_value=len(data), value=5, step=1
)

# Use the button and slider to decide whether to display head or tail
if choose == "Head":
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
    data.drop(columns=["Unnamed: 0"], inplace=True)
    cols_to_preprocess = [
        "sk1_classroom",
        "sk2_classroom",
        "sk3_classroom",
        "sk4_classroom",
        "sk5_classroom",
    ]
    data[cols_to_preprocess] = (
        data[cols_to_preprocess].replace(",", ".", regex=True).astype(float)
    )
    return data


# Perform data preprocessing
preprocessed_data = data_preprocessing(data)

# Display descriptive statistics
st.write("### Descriptive Statistics (after data preprocessing)")
st.write(preprocessed_data.describe())


# Univariate Analysis Histograms and QQ Plots
st.write("### Univariate Analysis")
cols = [
    "total_posts",
    "helpful_post",
    "nice_code_post",
    "collaborative_post",
    "confused_post",
    "creative_post",
    "bad_post",
    "amazing_post",
    "timeonline",
    "sk1_classroom",
    "sk2_classroom",
    "sk5_classroom",
    "sk3_classroom",
    "sk4_classroom",
    "Approved",
]


def generate_histogram(data, cols):
    hist_fig = sp.make_subplots(rows=3, cols=5, subplot_titles=cols)

    for i, col in enumerate(cols):
        row = i // 5 + 1
        col_num = i % 5 + 1
        # Histogram
        hist_fig.add_trace(
            go.Histogram(
                x=data[col],
                nbinsx=30,
                name=col,
                histnorm="probability density",
            ),
            row=row,
            col=col_num,
        )

    hist_fig.update_layout(
        height=800,
        width=1000,
        title_text="Univariate Analysis - Histograms",
        showlegend=False,
    )
    return hist_fig


hist_fig = generate_histogram(preprocessed_data, cols)
st.plotly_chart(hist_fig, use_container_width=True)


# QQ Plots
def generate_qq_plots(data, cols):
    # Create a figure with subplots for QQ plots
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))

    # Generate QQ plots for each column
    for i, ax in enumerate(axes.flat):
        sm.qqplot(data=data[cols[i]], ax=ax, line="45")

        # Set title for each subplot
        ax.set_title(f"{cols[i]}", pad=10, fontsize=8)

        # Remove labels and ticks for clarity
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
    return fig


# Display the QQ plot in Streamlit
fig = generate_qq_plots(preprocessed_data, cols)
# Adjust layout
plt.tight_layout()
st.pyplot(fig)


# Bivariate Analysis
# Define the columns representing the features
response = [
    "total_posts",
    "helpful_post",
    "nice_code_post",
    "collaborative_post",
    "confused_post",
    "creative_post",
    "bad_post",
    "amazing_post",
    "timeonline",
    "sk1_classroom",
    "sk2_classroom",
    "sk5_classroom",
    "sk3_classroom",
    "sk4_classroom",
]

# Extract the target variable into the target DataFrame
target = preprocessed_data["Approved"]


# Create a function to generate a box plot for a given response variable
def create_boxplot(target, response_var):
    fig = go.Figure(
        data=[go.Box(x=target, y=preprocessed_data[response_var], boxmean=True)]
    )
    fig.update_layout(
        title=f"Approved vs. {response_var}",
        xaxis_title="Approved",
        yaxis_title=response_var,
    )
    return fig


# Display the BoxPlots in a grid
st.write("### Bivariate Analysis")

# Create a 3-column layout
cols = st.columns(3)

# Iterate over the response variables and plot them in the columns
for idx, response_var in enumerate(response):
    with cols[idx % 3]:
        st.plotly_chart(create_boxplot(target, response_var), use_container_width=True)


# Pairplot
st.write("### Pairplot")


# Function to create the pairplot
def create_pairplot(data, height, width):
    fig = px.scatter_matrix(
        data,
        dimensions=list(data.columns),
        color_discrete_sequence=[
            "#3498db",
            "#e74c3c",
            "#2ecc71",
            "#9b59b6",
            "#1abc9c",
            "#f1c40f",
            "#e67e73",
            "#2c3e50",
            "#8e44ad",
            "#1a1d23",
        ],
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins as needed
        width=width,
    )
    return fig


# Set default values for height and width in session state
if "height" not in st.session_state:
    st.session_state.height = 2000
if "width" not in st.session_state:
    st.session_state.width = 2000

# Create sliders for height and width
st.session_state.height = st.slider("Height:", 600, 2000, st.session_state.height)
st.session_state.width = st.slider("Width:", 600, 2000, st.session_state.width)

# Create the initial pairplot
fig = create_pairplot(
    preprocessed_data, st.session_state.height, st.session_state.width
)

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
        colorscale="Viridis",  # Choose your color scale here
        showscale=True,
    )

    fig.update_layout(
        title="Correlation Matrix Heatmap (Seaborn Number System)",
        xaxis=dict(side="bottom"),  # Position x-axis at the bottom
        width=width,  # Adjust as needed
        height=height,  # Adjust as needed
        margin=dict(
            t=100
        ),  # Increase top margin for title spacing and left margin for title placement
    )
    return fig


# Set default values for height and width
default_height = 600
default_width = 800

# Create sliders for height and width
height = st.slider("Height:", 400, 1000, default_height)
width = st.slider("Width:", 400, 1200, default_width)

# Button to generate the heatmap
if st.button("Generate Pearson Correlation Heatmap"):
    # Create the heatmap
    fig = create_heatmap(preprocessed_data, response, height, width)

    # Display the heatmap
    st.plotly_chart(fig)


# Elbow Curve

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        """
        Initialize the DataFrameSelector with the specified attribute names.

        Args:
            attribute_names (list): List of column names to be selected from the dataframe.
        """
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        """
        Fit the transformer. This method doesn't do anything as this transformer
        doesn't need to learn anything from the data.

        Args:
            X (pandas.DataFrame): Input dataframe.
            y (array-like, optional): Target values (ignored).

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Transform the dataframe by selecting the specified columns.

        Args:
            X (pandas.DataFrame): Input dataframe.

        Returns:
            pandas.DataFrame: A dataframe containing only the selected columns.
        """
        return X[self.attribute_names]


# Split the dataset into training and testing sets.
data = preprocessed_data.copy()

# Calculate the average skills score
data["avg_skills_score"] = (
    data["sk1_classroom"]
    + data["sk2_classroom"]
    + data["sk3_classroom"]
    + data["sk4_classroom"]
    + data["sk5_classroom"]
) / 5

# Define the response variables
response = [
    "total_posts",
    "helpful_post",
    "nice_code_post",
    "collaborative_post",
    "confused_post",
    "creative_post",
    "bad_post",
    "amazing_post",
    "timeonline",
    "sk1_classroom",
    "sk2_classroom",
    "sk5_classroom",
    "sk3_classroom",
    "sk4_classroom",
    "avg_skills_score",
]

# Initialize SMOTE with random state for reproducibility.
smote = SMOTE(random_state=42)

# Apply SMOTE to resample the data.
X, y = smote.fit_resample(data[response], target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=42
)

# Define the pipeline.
pipeline = Pipeline(
    [
        ("select_feature", DataFrameSelector(response)),
        ("imputer", SimpleImputer(strategy="median")),
        ("standardizer", StandardScaler()),
    ]
)

X_train = pipeline.fit_transform(X_train)

# Create instances of the classifiers.
rf = RandomForestClassifier()
ab = AdaBoostClassifier()
dt = DecisionTreeClassifier()
xg = XGBClassifier()


# Initialize StratifiedKFold with 10 splits, shuffling, and random state.
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def build_model(model_init, kf, X_train, y_train):
    """
    Build and evaluate a machine learning model using cross-validated predictions.

    Args:
      model_init (estimator object): The machine learning model to be trained and evaluated.
      kf (KFold or StratifiedKFold object): Cross-validation strategy for splitting the data.
      X_train (array-like): Feature matrix of the training data.
      y_train (array-like): Target labels for the training data.

    Returns:
      dict: A dictionary containing the classification report and accuracy score.
    """

    # Generate cross-validated predictions.
    y_pred = cross_val_predict(model_init, X_train, y_train, cv=kf)

    # Compute accuracy.
    accuracy = accuracy_score(y_train, y_pred)

    # Compute classification report.
    report = classification_report(y_train, y_pred)

    # Return the report and accuracy score.
    return report, accuracy


# Function to generate classification metrics data
def generate_metrics_df(models):
    # Iterate over each model and build the model, then print evaluation metrics.
    classification_df = []
    for model_name, model in models.items():
        # Build the model and obtain the classification report and accuracy.
        report, acc = build_model(model, kf, X_train, y_train)

        # Extract the precision, fscore, and recall for both approved values 0 and 1.
        precision_0, fscore_0, recall_0 = report.split("\n")[2].split()[1:4]
        precision_1, fscore_1, recall_1 = report.split("\n")[3].split()[1:4]

        # Append the data to the list.
        classification_df.append(
            {
                "Classifier": model_name,
                "Precision (Approved 0)": float(precision_0),
                "Fscore (Approved 0)": fscore_0,
                "Recall (Approved 0)": recall_0,
                "Precision (Approved 1)": precision_1,
                "Fscore (Approved 1)": fscore_1,
                "Recall (Approved 1)": recall_1,
                "Accuracy": acc,
            }
        )
    return classification_df


# Define a dictionary containing the classifiers.
models = {
    "RandomForestClassifier": rf,
    "AdaBoostClassifier": ab,
    "DecisionTreeClassifier": dt,
    "XGBClassifier": xg,
}

# Create a DataFrame from the data.
metrics_data = generate_metrics_df(models)
models_df = pd.DataFrame(metrics_data)

# Create a table in Streamlit.
st.write(
    "### Classification Metrics using stratified KFold Cross-Validation with 10 folds"
)
st.write(models_df)

# Transform the entire dataset using the pipeline.
X_t = pipeline.fit_transform(X)

# Convert the transformed data to a DataFrame with appropriate column names.
X_t = pd.DataFrame(X_t, columns=response)


# Create a function to generate the elbow curve
def generate_elbow_curve(n_clusters):
    wcss_values = []
    for k in list(n_clusters):
        kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
        kmeans.fit(X_t)
        wcss_values.append(kmeans.inertia_)

    # Create the elbow curve plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(n_clusters), y=wcss_values, mode="lines+markers"))
    fig.update_layout(
        xaxis_title="Number of Clusters(k)",
        yaxis_title="Within-Cluster Sum of Squares (WCSS)",
        shapes=[
            dict(
                type="line",
                x0=2,
                y0=0,
                x1=2,
                y1=max(wcss_values),
                line=dict(dash="dash", color="red"),
            )
        ],
    )
    return fig


# Generate and display the elbow curve
st.write("### Elbow Curve to find the optimal number of clusters")
n_clusters = range(1, 11)
fig = generate_elbow_curve(n_clusters)
st.plotly_chart(fig, use_container_width=True)


# Kmeans Clustering

# Apply KMeans clustering based on the optimal value of k = 2.
kmeans = KMeans(n_clusters=2, random_state=1, n_init=10)

# Assign cluster labels to each data point.
X_t["Cluster_KMeans"] = kmeans.fit_predict(X_t)


# Define the Streamlit app
def kmeans_cluster_viz():
    st.title("KMeans Clustering Visualization")

    # Print cluster counts
    st.subheader("Cluster Counts:")
    st.write(X_t["Cluster_KMeans"].value_counts())

    # Drop the 'Cluster_KMeans' column to revert to the original data frame without the clustering label
    X_t.drop(columns=["Cluster_KMeans"], inplace=True)

    # Apply PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_t)

    # Create a DataFrame for visualization
    df_vis = pd.DataFrame(
        {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": kmeans.labels_}
    )

    # Visualize with Plotly
    fig = px.scatter(
        df_vis,
        x="PC1",
        y="PC2",
        color="Cluster",
        title="KMeans Clustering with Centroids",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        color_continuous_scale="viridis",
    )

    # Add centroids to the plot
    centroids = pca.transform(kmeans.cluster_centers_)
    fig.add_scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode="markers",
        marker=dict(size=14, color="red", symbol="x"),
    )

    return fig


kmeans_viz = kmeans_cluster_viz()
st.plotly_chart(kmeans_viz)

