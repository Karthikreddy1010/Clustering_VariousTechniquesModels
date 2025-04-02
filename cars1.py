import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Plot enhanced relational plots between
    multiple columns with better styling."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Vehicle Data Analysis: Relational Plots', fontsize=18, y=1.02, fontweight='bold')
    # Plot 1: Milage vs Price
    sns.scatterplot(
        data=df, x='milage', y='price', 
        ax=axes[0, 0], color='royalblue', alpha=0.7, edgecolor='w', s=80
    )
    axes[0, 0].set_title('Milage vs Price', fontsize=14, pad=10)
    axes[0, 0].set_xlabel('Milage (miles)', fontsize=12)
    axes[0, 0].set_ylabel('Price ($)', fontsize=12)
    axes[0, 0].grid(True, linestyle='--', alpha=0.4)
    # Plot 2: Model Year vs Price
    sns.scatterplot(
        data=df, x='model_year', y='price', 
        ax=axes[0, 1], color='forestgreen', alpha=0.7, edgecolor='w', s=80
    )
    axes[0, 1].set_title('Model Year vs Price', fontsize=14, pad=10)
    axes[0, 1].set_xlabel('Model Year', fontsize=12)
    axes[0, 1].set_ylabel('Price ($)', fontsize=12)
    axes[0, 1].grid(True, linestyle='--', alpha=0.4)
    # Plot 3: Fuel Type vs Price
    sns.boxplot(
        data=df, x='fuel_type', y='price', ax=axes[1, 0], 
        hue='fuel_type', palette='viridis', width=0.6, 
        linewidth=1.5, fliersize=4
    )
    axes[1, 0].set_title('Fuel Type vs Price', fontsize=14, pad=10)
    axes[1, 0].set_xlabel('Fuel Type', fontsize=12)
    axes[1, 0].set_ylabel('Price ($)', fontsize=12)
    axes[1, 0].grid(True, linestyle='--', alpha=0.4)
    if axes[1, 0].get_legend() is not None:
        axes[1, 0].get_legend().remove()
    # Plot 4: Milage vs Price (Regression)
    sns.regplot(
        data=df, x='milage', y='price', 
        ax=axes[1, 1], color='crimson', scatter_kws={'alpha':0.4, 's':60},
        line_kws={'color': 'darkred', 'lw': 2}
    )
    axes[1, 1].set_title('Milage vs Price (with Regression)', fontsize=14, pad=10)
    axes[1, 1].set_xlabel('Milage (miles)', fontsize=12)
    axes[1, 1].set_ylabel('Price ($)', fontsize=12)
    axes[1, 1].grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def plot_categorical_plot(df):
    """Plot an enhanced categorical plot for the 'fuel_type' column."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6))
    # Get dynamic palette based on unique fuel types
    fuel_types = df['fuel_type'].unique()
    palette = sns.color_palette("husl", n_colors=len(fuel_types))
    ax = sns.countplot(
        data=df, 
        x='fuel_type', 
        palette=palette,
        hue='fuel_type',
        edgecolor='black',
        linewidth=1,
        saturation=0.9,
        dodge=False
    )
    # Only remove legend if it exists
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', 
            va='center', 
            xytext=(0, 5), 
            textcoords='offset points',
            fontsize=12
        )
    plt.title('Distribution of Vehicle Fuel Types', fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('Fuel Type', fontsize=14, labelpad=10)
    plt.ylabel('Number of Vehicles', fontsize=14, labelpad=10)
    sns.despine(left=True, bottom=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Plot an enhanced histogram of price distribution with detailed statistics
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 6))
    # Create histogram with KDE
    ax = sns.histplot(
        data=df, x='price', kde=True, 
        color='teal', bins=30,
        edgecolor='white', linewidth=0.5
    ) 
    # Calculate statistics
    mean = df['price'].mean()
    median = df['price'].median()
    std = df['price'].std()
    skewness = df['price'].skew()
    kurtosis = df['price'].kurtosis()
    # Add statistical lines
    ax.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean: ${mean:,.0f}')
    ax.axvline(median, color='green', linestyle='--', linewidth=1.5, label=f'Median: ${median:,.0f}')
    ax.axvline(mean + std, color='blue', linestyle=':', linewidth=1, label=f'+1 Std Dev: ${mean + std:,.0f}')
    ax.axvline(mean - std, color='blue', linestyle=':', linewidth=1, label=f'-1 Std Dev: ${mean - std:,.0f}')
    # Add statistical annotations
    stats_text = (f'Standard Deviation: ${std:,.0f}\n'
                 f'Skewness: {skewness:.2f} (Right-skewed)\n'
                 f'Kurtosis: {kurtosis:.2f} (Leptokurtic)')
    plt.annotate(stats_text, xy=(0.72, 0.75), xycoords='axes fraction',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12)
    # Formatting
    plt.title('Vehicle Price Distribution with Statistics', fontsize=16, pad=20)
    plt.xlabel('Price ($)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(loc='upper right')
    # Improve x-axis formatting
    ax.xaxis.set_major_formatter('${x:,.0f}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
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
    # Handle missing values
    numerical_cols = ['milage', 'price', 'model_year']
    categorical_cols = ['fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    # Remove outliers using IQR
    def remove_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    for col in ['milage', 'price']:
        if col in df.columns:
            df = remove_outliers(df, col)
    
    return df


def writing(moments, col):
    """Print statistical moments for a given column."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    skewness = 'not skewed' if abs(moments[2]) < 0.5 else 'right skewed' if moments[2] > 0 else 'left skewed'
    kurtosis = 'platykurtic' if moments[3] < 0 else 'leptokurtic' if moments[3] > 0 else 'mesokurtic'
    print(f'The data was {skewness} and {kurtosis}.')
    return


def perform_clustering(df, col1, col2):
    """Perform clustering on the dataset using KMeans."""
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[[col1, col2]])
    df_scaled = pd.DataFrame(df_scaled, columns=[col1, col2])
    
    def plot_elbow_method():
        inertia = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_scaled)
            inertia.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), inertia, marker='o', linestyle='--', color='blue')
        plt.title('Elbow Method for Optimal K', fontsize=16)
        plt.xlabel('Number of Clusters', fontsize=14)
        plt.ylabel('Inertia', fontsize=14)
        plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    plot_elbow_method()
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    centres = kmeans.cluster_centers_
    
    score = silhouette_score(df_scaled, labels)
    print(f'Silhouette Score for k={optimal_k}: {score:.2f}')
    
    # Inverse transform centers for original scale visualization
    centres_original = scaler.inverse_transform(centres)
    return labels, df_scaled, centres_original[:, 0], centres_original[:, 1]


def plot_clustered_data(labels, data, xkmeans, ykmeans, df, col1, col2):
    """Plot the clustered data in original scale."""
    plt.figure(figsize=(10, 6))
    plt.scatter(df[col1], df[col2], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('Cluster Visualization (Original Scale)', fontsize=16)
    plt.xlabel(col1, fontsize=14)
    plt.ylabel(col2, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression fitting on the dataset."""
    scaler = MinMaxScaler()
    x = df[[col1]].values
    y = df[col2].values
    
    # Scale both features
    x_scaled = scaler.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_scaled, y)
    y_pred = model.predict(x_scaled)
    
    x_original = scaler.inverse_transform(x_scaled)
    
    # Print regression coefficients
    print(f"\nRegression Coefficients:")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    return x_original, y, y_pred


def plot_fitted_data(x, y, y_pred, col1, col2):
    """Plot the fitted data with actual vs predicted."""
    plt.figure(figsize=(10, 6))
    
    # Plot actual data points
    plt.scatter(x, y, c='blue', label='Actual Data', alpha=0.6)
    
    # Plot regression line (sorted for smooth line)
    sorted_idx = x.flatten().argsort()
    plt.plot(x[sorted_idx], y_pred[sorted_idx], 
             c='red', linewidth=2, label='Regression Line')
    
    plt.title(f'Linear Regression: {col1} vs {col2}', fontsize=16)
    plt.xlabel(col1, fontsize=14)
    plt.ylabel(col2, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('fitting.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def main():
    df = pd.read_csv("used_car.csv")
    df = preprocessing(df)
    print("First Five rows in a dataset:\n")
    print(df.head())
    print("Information of data:\n")
    print(df.info())
    print("Summary stastictics of a data:\n")
    print(df.describe())
    print("Correlation of data:\n")
    print(df.select_dtypes(include=['number']).corr())
    
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    
    moments = statistical_analysis(df, 'price')
    writing(moments, 'price')
    
    clustering_results = perform_clustering(df, 'milage', 'price')
    plot_clustered_data(clustering_results[0], clustering_results[1], 
                       clustering_results[2], clustering_results[3],
                       df, 'milage', 'price')
    
    fitting_results = perform_fitting(df, 'milage', 'price')
    plot_fitted_data(fitting_results[0], fitting_results[1], 
                    fitting_results[2], 'milage', 'price')
    return


if __name__ == '__main__':
    main()