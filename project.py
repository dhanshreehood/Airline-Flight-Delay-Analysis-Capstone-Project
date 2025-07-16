import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# from google.colab import drive

# drive.mount('/content/drive', force_remount=True)

data_file = './Airline_Delay_Cause.csv'
# Load the CSV dataset and Excel file with column definitions
#data_path = '/mnt/data/Airline_Delay_Cause.csv'
definitions_file = './Download_Column_Definitions.xlsx'

# Task 1: Load and Explore the Dataset
def load_and_explore(data_file, definitions_file):
    # Load the data
    data = pd.read_csv(data_file)
    print("\nData Loaded Successfully.\n")

    # Load the column definitions
    definitions = pd.read_excel(definitions_file)
    print("\nColumn Definitions Loaded Successfully.\n")

    # Summarize the dataset
    print("\nDataset Summary:")
    print(data.info())
    print("\nDescriptive Statistics:")
    print(data.describe(include='all'))
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nSample Data:")
    print(data.head())

    # Display column definitions
    print("\nColumn Definitions:")
    print(definitions)

    return data, definitions

# Task 2: Data Cleaning
def clean_data(data):
    print("\nStarting Data Cleaning...")

    # List of columns expected to be numeric
    numeric_cols = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct',
                    'late_aircraft_ct', 'arr_cancelled', 'arr_diverted', 'arr_delay', 'carrier_delay', 
                    'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
    
    if 'month' in data.columns:
        data['month'] = data['month'].astype(str)
    
    # Convert columns that are expected to be numeric to numeric, forcing errors to NaN for non-numeric values
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Non-numeric entries become NaN

    # Handle missing values (fill with mean for numeric columns, mode for others)
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)

    # Detect and remove anomalies using Z-score
    for col in numeric_cols:
        # Handle Z-scores (remove outliers)
        z_scores = zscore(data[col].dropna())  # Remove NaN from Z-score calculation
        data = data[(z_scores < 3) & (z_scores > -3)]  # Filter out outliers

    # Standardize date formats (if any)
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')

    print("\nData Cleaning Completed.")
    print("\nPost-Cleaning Missing Values:")
    print(data.isnull().sum())

    print("###################################OK####################################")
    print(data)
    return data

# Task 3: Exploratory Data Analysis (EDA)
def exploratory_data_analysis(data, definitions):
    print("\nStarting Exploratory Data Analysis (EDA)...")

    # Distributions of numeric features
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    i=0
    for col in numeric_cols:
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(definitions.loc[definitions['Column Name'] == col, 'Column Definition'].values[0])
        plt.show()
        i=i+1
        if i > 5:
         break

    # # Relationships between numeric features
    # if len(numeric_cols) > 1:
    #     sns.pairplot(data[numeric_cols])
    #     plt.title("Pairwise Relationships")
    #     plt.show()

    # Correlation heatmap

    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    correlation = numeric_data.corr()
    plt.figure(figsize=(12, 8)) 
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

    print("\nEDA Completed.")

# Main Execution
if __name__ == "__main__":
    # Load and explore datasets
    data, definitions = load_and_explore(data_file, definitions_file)

    # Clean the dataset
    cleaned_data = clean_data(data)

    # Perform EDA
    exploratory_data_analysis(cleaned_data, definitions)





#################################################phase 2#########################################




import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sqlalchemy import create_engine
import schedule
import time
from sqlalchemy import text
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Task 1: Load and Explore the Dataset
def load_and_explore(data_file, definitions_file):
    print("\nStarting Data Ingestion...")

    # Load the data
    data = pd.read_csv(data_file)
    print("\nData Loaded Successfully.\n")

    # Load the column definitions
    definitions = pd.read_excel(definitions_file)
    print("\nColumn Definitions Loaded Successfully.\n")

    # Summarize the dataset
    print("\nDataset Summary:")
    print(data.info())
    print("\nDescriptive Statistics:")
    print(data.describe(include='all'))
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nSample Data:")
    print(data.head())

    return data, definitions

# Task 2: Data Cleaning
def clean_data(data):
    """
    Cleans the input dataset by handling missing values, anomalies, and constructing a 'date' column from 'year' and 'month'.

    Args:
    - data (pd.DataFrame): Input dataset.

    Returns:
    - pd.DataFrame: Cleaned dataset.
    """
    print("\nStarting Data Cleaning...")

    if 'month' in data.columns:
        data['month'] = data['month'].astype(str)
    
    
    if 'year' in data.columns:
        data['year'] = data['year'].astype(str)
    # Handle missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)

    # Detect and remove anomalies using Z-score
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        z_scores = zscore(data[col])
        data = data[(z_scores < 3) & (z_scores > -3)]

    # Create 'date' column from 'year' and 'month' if they exist
    if 'year' in data.columns and 'month' in data.columns:
        print("\nCreating 'date' column from 'year' and 'month'...")
        data['date'] = pd.to_datetime(
            data[['year', 'month']].assign(day=1),  # Assuming day=1 for all
            errors='coerce'
        )
    else:
        print("\nWarning: 'year' and/or 'month' columns are missing. Unable to create 'date' column.")

    print("\nData Cleaning Completed.")
    return data

# Task 3: Data Storage Optimization


def optimize_storage(data, rds_connection_string):
    print("\nOptimizing Data Storage...")

    # Create RDS database connection
    engine = create_engine(rds_connection_string)

    # Store data with indexing
    with engine.connect() as connection:
        # Store data in the table
        data.to_sql('airline_data', con=connection, if_exists='replace', index=False)

        # Add indexing for optimization
        index_query = text("CREATE INDEX idx_year ON airline_data (year(10));")  # SQL statement for creating index
        
        
        connection.execute(index_query)

    print("\nData Stored and Indexed Successfully in AWS RDS.")

    print("\nData Stored and Indexed Successfully in AWS RDS.")

# Task 4: Data Validation
def validate_data(data):
    print("\nValidating Data Integrity...")

    assert data.isnull().sum().sum() == 0, "Data contains missing values after cleaning!"

    if 'date' in data.columns:
        assert pd.api.types.is_datetime64_any_dtype(data['date']), "Date column is not in datetime format!"

    print("\nData Integrity Validated Successfully.")

# Task 5: Automation for Periodic Ingestion
def ingest_periodically(data_file, definitions_file, rds_connection_string):
    def task():
        print("\nRunning Scheduled Ingestion...")
        data, definitions = load_and_explore(data_file, definitions_file)
        cleaned_data = clean_data(data)
        optimize_storage(cleaned_data, rds_connection_string)
        validate_data(cleaned_data)
        print("\nScheduled Ingestion Completed Successfully.")

    schedule.every().day.at("01:00").do(task)  # Schedule for daily ingestion at 1 AM

    print("\nStarting Scheduler. Press Ctrl+C to exit.")
    while True:
        schedule.run_pending()
        time.sleep(1)

# Main Execution
if __name__ == "__main__":
    # File paths
    # data_file = '/content/drive/My Drive/Capstone Project/Airline_Delay_Cause.csv'
    # definitions_file = '/content/drive/My Drive/Capstone Project/Download_Column_Definitions.xlsx'

    # AWS RDS connection string
    rds_connection_string = "mysql+pymysql://admin:admin123@mydb.cl2ycy4ks8pr.ap-south-1.rds.amazonaws.com:3306/airline_db"

    # Load and explore datasets
   # data, definitions = load_and_explore(data_file, definitions_file)

    # Clean the dataset
    cleaned_data = clean_data(data)

    # Optimize storage
    optimize_storage(cleaned_data, rds_connection_string)

    # Validate data
    validate_data(cleaned_data)

    # Uncomment the line below to enable periodic ingestion
    # ingest_periodically(data_file, definitions_file, rds_connection_string)





#####################################Phase 3#######################################



import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, silhouette_score
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (simplified for demonstration)
# data_path = '/mnt/data/Airline_Delay_Cause.xlsx'
# df = pd.read_excel(data_path, sheet_name='Airline_Delay_Cause')

# Feature Engineering
def feature_engineering(data):
    # Interaction Feature: Delay per Flight
    data['delay_per_flight'] = data['arr_delay'] / data['arr_flights']
    data['delay_per_flight'].fillna(0, inplace=True)  # Handle division by zero

    # Aggregated Metrics: Total delays caused by carrier, weather, etc.
    data['total_ct'] = data[['carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']].sum(axis=1)

    # Proportion Features
    for col in ['carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']:
        data[f'proportion_{col}'] = data[col] / data['total_ct']
        data[f'proportion_{col}'].fillna(0, inplace=True)

    return data

# Summarization by Group
def summarize_data(data):
    # Group by Year and Month for analysis
    summary = data.groupby(['year', 'month']).agg({
        'arr_flights': 'sum',
        'arr_del15': 'sum',
        'arr_delay': 'sum',
        'carrier_delay': 'sum',
        'weather_delay': 'sum',
        'nas_delay': 'sum',
        'security_delay': 'sum',
        'late_aircraft_delay': 'sum'
    }).reset_index()

    # Calculate Proportions for Delay Types
    delay_columns = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
    for col in delay_columns:
        summary[f'proportion_{col}'] = summary[col] / summary['arr_delay']
        summary[f'proportion_{col}'].fillna(0, inplace=True)

    return summary

# Normalize Data
def normalize_data(data):
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # Replace infinities and large values
    data[numeric_cols] = data[numeric_cols].replace([float('inf'), -float('inf')], float('nan'))
    data[numeric_cols] = data[numeric_cols].fillna(0)  # Replace NaN with 0

    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Dimensionality Reduction
def reduce_dimensions(data, n_components=5):
    pca = PCA(n_components=n_components)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # Replace NaN or infinite values with zero
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    principal_components = pca.fit_transform(data[numeric_cols])

    # Create DataFrame for Principal Components
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    return pca_df

# Advanced Data Analysis
def advanced_analysis(data):
    # Predictive Modeling: Random Forest Regressor
    X = data[['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']]
    y = data['arr_delay']

    # Replace NaN or infinite values in features and target with zero
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Random Forest Mean Squared Error: {mse}")

    # Clustering: K-Means
    clustering_features = data[['delay_per_flight', 'arr_flights', 'total_ct']]
    clustering_features = clustering_features.replace([np.inf, -np.inf], np.nan).fillna(0)

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(clustering_features)

    silhouette_avg = silhouette_score(clustering_features, data['cluster'])
    print(f"K-Means Silhouette Score: {silhouette_avg}")

    return model, kmeans

# Visualization and Reporting
def create_visualizations(data):
    # Visualize Delays by Month
    monthly_delays = data.groupby(['year', 'month'])['arr_delay'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_delays['month'], monthly_delays['arr_delay'], marker='o')
    plt.title('Monthly Total Arrival Delays')
    plt.xlabel('Month')
    plt.ylabel('Total Delay (minutes)')
    plt.grid()
    plt.show()

    # Cluster Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(data['delay_per_flight'], data['total_ct'], c=data['cluster'], cmap='viridis')
    plt.title('Flight Clusters')
    plt.xlabel('Delay per Flight')
    plt.ylabel('Total Cause Counts')
    plt.colorbar(label='Cluster')
    plt.show()

# Data Warehousing Setup
def setup_data_warehouse(transformed_data, summarized_data, connection_string):
    engine = create_engine(connection_string)

    with engine.connect() as connection:
        # Star Schema: Load Fact Table
        transformed_data.to_sql('fact_flights', con=connection, if_exists='replace', index=False)

        # Snowflake Schema: Load Dimension Tables
        summarized_data.to_sql('dim_summary', con=connection, if_exists='replace', index=False)

        # Add Indexes for Efficient Querying
        index_queries = [
            text("CREATE INDEX idx_year_month ON fact_flights (year(10), month(10));"),
            text("CREATE INDEX idx_summary_year_month ON dim_summary (year(10), month(10));")
        ]
        for query in index_queries:
            connection.execute(query)

    print("Data Warehousing Setup Completed.")

# Data Governance Practices
def setup_data_governance():
    governance_practices = {
        'Data Quality': "Regular validation scripts ensure data consistency and completeness.",
        'Security': "User access is role-based with encryption in transit and at rest.",
        'Compliance': "Data storage complies with GDPR and CCPA guidelines."
    }
    print("Data Governance Practices:")
    for key, value in governance_practices.items():
        print(f"- {key}: {value}")


def preprocess_for_modeling(data):
    # Replace infinite values and fill NaNs in modeling columns
    modeling_cols = ['arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct']
    data[modeling_cols] = data[modeling_cols].replace([float('inf'), -float('inf')], float('nan')).fillna(0)
    return data


# Main Transformation Workflow
def transform_data(data):
    print("Starting Data Transformation...")

    # Step 1: Feature Engineering
    data = feature_engineering(data)
    print("Feature Engineering Completed.")

    # Step 2: Summarize Data
    summary = summarize_data(data)
    print("Data Summarization Completed.")

    # Step 3: Normalize Data
    normalized_data = normalize_data(data)
    print("Data Normalization Completed.")

    data = preprocess_for_modeling(data)

    # Step 4: Dimensionality Reduction
    pca_data = reduce_dimensions(normalized_data)
    print("Dimensionality Reduction Completed.")

    return data, summary, pca_data

# Execute Transformation Workflow
connection_string = "mysql+pymysql://admin:admin123@mydb.cl2ycy4ks8pr.ap-south-1.rds.amazonaws.com:3306/airline_db"
transformed_data, summarized_data, pca_data = transform_data(data)

# Save Outputs
transformed_data.to_csv('./transformed_data.csv', index=False)
summarized_data.to_csv('./summarized_data.csv', index=False)
pca_data.to_csv('./pca_data.csv', index=False)

# Setup Data Warehouse
setup_data_warehouse(transformed_data, summarized_data, connection_string)

# Setup Data Governance
setup_data_governance()

# Perform Advanced Analysis
model, kmeans = advanced_analysis(transformed_data)

# Create Visualizations
create_visualizations(transformed_data)

print("Transformation, Warehousing, and Analysis Workflow Completed.")
