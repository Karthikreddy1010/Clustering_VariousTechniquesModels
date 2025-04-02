import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple, List


def plot_relational_plot(df: pd.DataFrame) -> None:
    """Generate a relational plot (pairplot) of numerical columns."""
    try:
        numerical_cols = ['model_year', 'milage', 'engine', 'price']
        numerical_cols = [col for col in numerical_cols 
                         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numerical_cols) > 1:
            print(f"Creating relational plot with columns: {numerical_cols}")
            sns.set_theme(style="ticks", palette="pastel")
            g = sns.pairplot(df[numerical_cols], diag_kind='kde', plot_kws={'alpha': 0.8})
            g.figure.suptitle('Relational Plot of Numerical Columns', y=1.02, fontsize=16)
            plt.savefig('relational_plot.png', bbox_inches='tight')
            plt.close()
            print("Relational plot saved successfully.")
        else:
            print(f"Not enough numerical columns for relational plot. Found: {numerical_cols}")
    except Exception as e:
        print(f"Error creating relational plot: {str(e)}")


def plot_categorical_plot(df: pd.DataFrame) -> None:
    """Generate a categorical plot (countplot) of the 'fuel_type' column."""
    try:
        if 'fuel_type' in df.columns:
            print("Creating fuel type distribution plot...")
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            ax = sns.countplot(
                x='fuel_type', data=df, palette='viridis',
                order=df['fuel_type'].value_counts().index,
                hue='fuel_type', legend=False
            )
            plt.title('Distribution of Fuel Types', fontsize=16)
            plt.xlabel('Fuel Type', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            
            for p in ax.patches:
                ax.annotate(
                    f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 10),
                    textcoords='offset points', fontsize=10
                )
            plt.savefig('categorical_plot.png', bbox_inches='tight')
            plt.close()
            print("Fuel type plot saved successfully.")
        else:
            print("Column 'fuel_type' not found in dataframe.")
    except Exception as e:
        print(f"Error creating categorical plot: {str(e)}")


def plot_statistical_plot(df: pd.DataFrame) -> None:
    """Generate a statistical plot (histogram) of the 'price' column."""
    try:
        if 'price' in df.columns:
            if pd.api.types.is_numeric_dtype(df['price']):
                print("Creating price distribution histogram...")
                plt.figure(figsize=(10, 6))
                sns.set_theme(style="whitegrid")
                sns.histplot(df['price'], kde=True, color='blue', bins=30, edgecolor='black')
                plt.title('Distribution of Car Prices', fontsize=16)
                plt.xlabel('Price', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.savefig('statistical_plot.png', bbox_inches='tight')
                plt.close()
                print("Price histogram saved successfully.")
            else:
                print("Column 'price' exists but is not numeric.")
        else:
            print("Column 'price' not found in dataframe.")
    except Exception as e:
        print(f"Error creating statistical plot: {str(e)}")


def statistical_analysis(df: pd.DataFrame, col: str) -> Tuple[Optional[float], ...]:
    """Calculate statistical metrics for a specified column."""
    try:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                stddev = df[col].std()
                skew = df[col].skew()
                excess_kurtosis = df[col].kurtosis()
                return mean, stddev, skew, excess_kurtosis
            else:
                print(f"Column '{col}' is not numeric. Type: {df[col].dtype}")
        else:
            print(f"Column '{col}' not found in dataframe.")
        return None, None, None, None
    except Exception as e:
        print(f"Error during statistical analysis: {str(e)}")
        return None, None, None, None


def preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[MinMaxScaler]]:
    """Preprocess the data by handling missing values, outliers, and normalization."""
    try:
        print("\nStarting data preprocessing...")
        print(f"Initial shape: {df.shape}")
        
        # Clean 'engine' column if it exists
        if 'engine' in df.columns:
            # Remove non-numeric characters and convert to numeric
            df['engine'] = pd.to_numeric(
                df['engine'].astype(str).str.replace('[^0-9]', '', regex=True),
                errors='coerce'
            )
        
        # Handle missing values after cleaning
        df = df.dropna()
        print(f"After dropping NA values: {df.shape}")
        
        if df.empty:
            print("Warning: Dataframe is empty after dropping NA values!")
            return df, None
        
        # Identify numerical columns (now including cleaned 'engine')
        numerical_cols = ['model_year', 'milage', 'engine', 'price']
        numerical_cols = [col for col in numerical_cols 
                         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        print(f"Numerical columns found: {numerical_cols}")
        
        # Remove outliers using IQR method
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after = len(df)
            print(f"Outlier removal for {col}: {before} -> {after} records")
        
        if df.empty:
            print("Warning: Dataframe is empty after outlier removal!")
            return df, None
        
        # Normalize numerical data
        if numerical_cols:
            print("Normalizing numerical columns...")
            scaler = MinMaxScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            print("Normalization completed.")
            return df, scaler
        else:
            print("No numerical columns available for normalization.")
            return df, None
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return df, None


def writing(moments: Tuple, col: str) -> None:
    """Print statistical metrics and interpret skewness and kurtosis."""
    try:
        if None in moments:
            print(f"Cannot display statistics for {col} - invalid data.")
            return
        
        mean, stddev, skew, excess_kurtosis = moments
        
        print(f'\nStatistics for {col}:')
        print(f'{"Mean:":<20}{mean:.2f}')
        print(f'{"Standard Deviation:":<20}{stddev:.2f}')
        print(f'{"Skewness:":<20}{skew:.2f}')
        print(f'{"Excess Kurtosis:":<20}{excess_kurtosis:.2f}')
        
        # Interpret skewness
        if skew > 1:
            print("The distribution is highly right-skewed.")
        elif skew > 0.5:
            print("The distribution is moderately right-skewed.")
        elif skew < -1:
            print("The distribution is highly left-skewed.")
        elif skew < -0.5:
            print("The distribution is moderately left-skewed.")
        else:
            print("The distribution is approximately symmetric.")
        
        # Interpret kurtosis
        if excess_kurtosis > 1:
            print("The distribution has heavy tails (leptokurtic).")
        elif excess_kurtosis < -1:
            print("The distribution has light tails (platykurtic).")
        else:
            print("The distribution has normal tails (mesokurtic).")
    except Exception as e:
        print(f"Error in writing statistics: {str(e)}")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Calculate and visualize the correlation matrix for numerical columns."""
    try:
        numerical_cols = ['model_year', 'milage', 'engine', 'price']
        numerical_cols = [col for col in numerical_cols 
                         if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numerical_cols) > 1:
            print(f"\nCalculating correlations for: {numerical_cols}")
            corr_matrix = df[numerical_cols].corr()
            
            print("\nCorrelation Matrix:")
            print(corr_matrix.to_string(float_format="%.2f"))
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
            plt.title('Correlation Matrix of Numerical Columns', fontsize=16)
            plt.savefig('correlation_matrix.png', bbox_inches='tight')
            plt.close()
            print("Correlation matrix plot saved successfully.")
        else:
            print(f"Not enough numerical columns for correlation matrix. Found: {numerical_cols}")
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")


def perform_clustering(df: pd.DataFrame, col1: str, col2: str) -> Tuple:
    """Perform K-Means clustering on two specified columns."""
    try:
        # Validate columns
        for col in [col1, col2]:
            if col not in df.columns:
                print(f"Column '{col}' not found in dataframe.")
                return (None,) * 5
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column '{col}' is not numeric. Type: {df[col].dtype}")
                return (None,) * 5
        
        data = df[[col1, col2]].values
        print(f"\nPerforming clustering on {col1} vs {col2} with {len(data)} points...")

        def plot_elbow_method():
            inertias = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            plt.figure(figsize=(10, 6))
            plt.plot(K, inertias, 'bo-', markersize=8)
            plt.xlabel('Number of Clusters', fontsize=12)
            plt.ylabel('Inertia', fontsize=12)
            plt.title('Elbow Method for Optimal k', fontsize=16)
            plt.savefig('elbow_plot.png', bbox_inches='tight')
            plt.close()

        def calculate_metrics(k):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            inertia = kmeans.inertia_
            return score, inertia, kmeans

        # Plot elbow method
        plot_elbow_method()
        
        # Perform clustering with k=3
        k = 4
        score, inertia, kmeans = calculate_metrics(k)
        
        print(f"\nClustering results (k={k}):")
        print(f"Silhouette Score: {score:.3f}")
        print(f"Inertia: {inertia:.2f}")
        print("Cluster Centers:")
        print(kmeans.cluster_centers_)
        
        if score > 0.7:
            print("Strong clustering structure.")
        elif score > 0.5:
            print("Reasonable clustering structure.")
        else:
            print("Weak clustering structure.")
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        return labels, data, centers[:, 0], centers[:, 1], kmeans.labels_
    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        return (None,) * 5


def plot_clustered_data(labels, data, x_centers, y_centers, cluster_labels) -> None:
    """Visualize the clustering results."""
    try:
        if data is None:
            print("No clustering data to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        # Plot data points
        scatter = plt.scatter(
            data[:, 0], data[:, 1], c=labels, cmap='viridis',
            s=50, alpha=0.6, label='Data Points'
        )
        
        # Plot cluster centers
        plt.scatter(
            x_centers, y_centers, c='red', s=200, alpha=0.75,
            marker='X', label='Cluster Centers'
        )
        
        # Annotate centers
        for i, (x, y) in enumerate(zip(x_centers, y_centers)):
            plt.text(x, y, f'Center {i+1}', fontsize=12, ha='right')
        
        plt.title('Clustering Results (k=3)', fontsize=16)
        plt.xlabel('Feature 1 (Normalized)', fontsize=12)
        plt.ylabel('Feature 2 (Normalized)', fontsize=12)
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.savefig('clustering.png', bbox_inches='tight')
        plt.close()
        print("Clustering plot saved successfully.")
    except Exception as e:
        print(f"Error plotting clustered data: {str(e)}")


def perform_fitting(df: pd.DataFrame, col1: str, col2: str) -> Tuple:
    """Perform linear regression on two specified columns."""
    try:
        # Validate columns
        for col in [col1, col2]:
            if col not in df.columns:
                print(f"Column '{col}' not found in dataframe.")
                return (None,) * 3
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column '{col}' is not numeric. Type: {df[col].dtype}")
                return (None,) * 3
        
        if len(df) < 2:
            print("Not enough data points for regression (need at least 2).")
            return (None,) * 3
        
        print(f"\nPerforming linear regression: {col1} -> {col2}")
        X = df[[col1]].values
        y = df[col2].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        print("\nRegression Results:")
        print(f"Coefficient: {model.coef_[0]:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        print(f"R-squared: {model.score(X, y):.4f}")
        
        return np.hstack((X, y.reshape(-1, 1))), x_range, y_pred
    except Exception as e:
        print(f"Error during regression: {str(e)}")
        return (None,) * 3


def plot_fitted_data(data, x_range, y_pred) -> None:
    """Visualize the linear regression results."""
    try:
        if data is None:
            print("No regression data to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        # Plot original data
        plt.scatter(
            data[:, 0], data[:, 1], c='blue', 
            label='Data points', alpha=0.6
        )
        
        # Plot regression line
        plt.plot(
            x_range, y_pred, c='red', 
            label='Regression line', linewidth=2
        )
        
        plt.title('Linear Regression Results', fontsize=16)
        plt.xlabel('Independent Variable', fontsize=12)
        plt.ylabel('Dependent Variable', fontsize=12)
        plt.legend()
        plt.savefig('regression.png', bbox_inches='tight')
        plt.close()
        print("Regression plot saved successfully.")
    except Exception as e:
        print(f"Error plotting regression: {str(e)}")


def main() -> None:
    """Main function to execute the data analysis pipeline."""
    try:
        print("Starting data analysis...")
        
        # Load data
        print("\nLoading data...")
        df = pd.read_csv('used_car.csv')
        
        print("\n=== Data Summary ===")
        print(f"Total records: {len(df)}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nSummary statistics:")
        print(df.describe(include='all'))
        
        # Preprocess data
        df_processed, scaler = preprocessing(df)
        
        if df_processed.empty:
            print("\nProcessing stopped: Empty dataframe after preprocessing.")
            return
        
        print("\n=== Processed Data Summary ===")
        print(f"Records after preprocessing: {len(df_processed)}")
        print("\nProcessed data sample:")
        print(df_processed.head())
        
        # Visualization and analysis
        plot_relational_plot(df_processed)
        plot_statistical_plot(df_processed)
        plot_categorical_plot(df_processed)
        plot_correlation_matrix(df_processed)
        
        # Statistical analysis
        col = 'price'
        moments = statistical_analysis(df_processed, col)
        if None not in moments:
            writing(moments, col)
        
        # Clustering
        clustering_results = perform_clustering(df_processed, 'milage', 'price')
        plot_clustered_data(*clustering_results)
        
        # Regression
        fitting_results = perform_fitting(df_processed, 'engine', 'price')
        plot_fitted_data(*fitting_results)
        
        print("\nAnalysis completed successfully.")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == '__main__':
    main()