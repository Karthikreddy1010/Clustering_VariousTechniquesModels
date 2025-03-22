"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Plot relational plots between multiple columns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Relational Plots', fontsize=16)

    # Balance vs Bonus_miles
    sns.scatterplot(data=df, x='Balance', y='Bonus_miles', ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Balance vs Bonus Miles')
    axes[0, 0].set_xlabel('Balance')
    axes[0, 0].set_ylabel('Bonus Miles')

    # Balance vs Flight_miles_12mo
    sns.scatterplot(data=df, x='Balance', y='Flight_miles_12mo', ax=axes[0, 1], color='green')
    axes[0, 1].set_title('Balance vs Flight Miles (12 Months)')
    axes[0, 1].set_xlabel('Balance')
    axes[0, 1].set_ylabel('Flight Miles (12 Months)')

    # Bonus_miles vs Flight_miles_12mo
    sns.scatterplot(data=df, x='Bonus_miles', y='Flight_miles_12mo', ax=axes[1, 0], color='orange')
    axes[1, 0].set_title('Bonus Miles vs Flight Miles (12 Months)')
    axes[1, 0].set_xlabel('Bonus Miles')
    axes[1, 0].set_ylabel('Flight Miles (12 Months)')

    # Days_since_enroll vs Balance
    sns.scatterplot(data=df, x='Days_since_enroll', y='Balance', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Days Since Enrollment vs Balance')
    axes[1, 1].set_xlabel('Days Since Enrollment')
    axes[1, 1].set_ylabel('Balance')

    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Plot a categorical plot for the 'Award' column."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Award', hue='Award', palette='Set2')
    plt.title('Count of Awards', fontsize=16)
    plt.xlabel('Award', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.legend(title='Award', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('categorical_plot.png', bbox_inches='tight')
    plt.close()
    return


def plot_statistical_plot(df):
    """Plot a statistical plot (histogram) for the 'Balance' column."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Balance'], kde=True, color='teal', bins=30)
    plt.title('Distribution of Balance', fontsize=16)
    plt.xlabel('Balance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig('statistical_plot.png')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculate statistical moments for a given column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = df[col].skew()
    excess_kurtosis = df[col].kurtosis()
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the dataset: clean, handle missing values, and remove outliers."""
    # Remove duplicates
    df = df.drop_duplicates()

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Fill missing values
    def replace_null(data, cols):
        for col in cols:
            if data[col].dtype in ['int64', 'float64']:
                median = data[col].median()
                data[col] = data[col].fillna(median)
            else:
                mode = data[col].mode()[0]
                data[col] = data[col].fillna(mode)
        return data

    cols = ['Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles', 
            'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12', 
            'Days_since_enroll', 'Award']
    df = replace_null(df, cols)

    # Remove outliers
    def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    numeric_columns = ['Balance', 'Bonus_miles', 'Flight_miles_12mo', 'Days_since_enroll']
    for column in numeric_columns:
        df = remove_outliers(df, column)

    return df


def writing(moments, col):
    """Print statistical moments for a given column."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    if abs(moments[2]) < 0.5:
        skewness = 'not skewed'
    elif moments[2] > 0:
        skewness = 'right skewed'
    else:
        skewness = 'left skewed'

    if moments[3] < 0:
        kurtosis = 'platykurtic'
    elif moments[3] > 0:
        kurtosis = 'leptokurtic'
    else:
        kurtosis = 'mesokurtic'

    print(f'The data was {skewness} and {kurtosis}.')
    return


def perform_clustering(df, col1, col2):
    """Perform clustering on the dataset using KMeans."""
    def plot_elbow_method():
        inertia = []
        for k in range(2, 11):  # Start from 2 clusters
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df[[col1, col2]])
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), inertia, marker='o', linestyle='--', color='blue')
        plt.title('Elbow Method for Optimal K', fontsize=16)
        plt.xlabel('Number of Clusters', fontsize=14)
        plt.ylabel('Inertia', fontsize=14)
        plt.savefig('elbow_plot.png')
        plt.close()
        return

    def calculate_silhouette_score(k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[[col1, col2]])
        score = silhouette_score(df[[col1, col2]], kmeans.labels_)
        print(f'Silhouette Score for k={k}: {score:.2f}')
        return score

    # Gather data and scale
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[[col1, col2]])
    df_scaled = pd.DataFrame(df_scaled, columns=[col1, col2])

    # Plot elbow method
    plot_elbow_method()

    # Perform clustering with k=5
    optimal_k = 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    centres = kmeans.cluster_centers_

    # Calculate silhouette score for k=5
    final_score = calculate_silhouette_score(optimal_k)
    print(f'Final Silhouette Score for k={optimal_k}: {final_score:.2f}')

    return labels, df_scaled, centres[:, 0], centres[:, 1], labels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot the clustered data."""
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', s=50, label='Data Points')
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('Cluster Visualization', fontsize=16)
    plt.xlabel(data.columns[0], fontsize=14)
    plt.ylabel(data.columns[1], fontsize=14)
    plt.legend()
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression fitting on the dataset."""
    # Gather data and prepare for fitting
    x = df[col1].values.reshape(-1, 1)
    y = df[col2].values

    # Fit model
    model = LinearRegression()
    model.fit(x, y)

    # Predict across x
    y_pred = model.predict(x)

    return df, x, y_pred


def plot_fitted_data(data, x, y):
    """Plot the fitted data."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, data['Balance'], c='blue', label='Data Points')
    plt.plot(x, y, c='red', label='Fitted Line')
    plt.title('Fitted Data: Balance vs Bonus Miles', fontsize=16)
    plt.xlabel('Bonus Miles', fontsize=14)
    plt.ylabel('Balance', fontsize=14)
    plt.legend()
    plt.savefig('fitting.png')
    plt.close()
    return


def main():
    # Load the dataset
    df = pd.read_csv("data.csv")  # Replace with the actual file path
    
    # Debugging: Print column names and first few rows
    print("Column Names:", df.columns)
    print("First Few Rows:\n", df.head())
    
    # Strip leading/trailing spaces from column names (if any)
    df.columns = df.columns.str.strip()
    
    # Check if required columns exist
    required_columns = ['Balance', 'Bonus_miles', 'Award']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the dataset. Available columns: {df.columns}")
    
    # Proceed with preprocessing and analysis
    df = preprocessing(df)
    col = 'Balance'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 'Bonus_miles', 'Balance')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'Bonus_miles', 'Balance')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()