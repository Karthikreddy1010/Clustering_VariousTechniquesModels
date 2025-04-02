"""
This is the template file for the clustering and fitting assignment.
Updated to address all deprecation warnings while maintaining functionality.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Plot relational plots between multiple columns."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Relational Plots', fontsize=16)
    # GrLivArea vs SalePrice
    sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', ax=axes[0, 0],
                    color='blue', alpha=0.6)
    axes[0, 0].set_title('Living Area vs Sale Price')
    axes[0, 0].set_xlabel('Above Grade Living Area (sqft)')
    axes[0, 0].set_ylabel('Sale Price ($)')
    # TotalBsmtSF vs SalePrice
    sns.scatterplot(data=df, x='TotalBsmtSF', y='SalePrice', ax=axes[0, 1],
                    color='green', alpha=0.6)
    axes[0, 1].set_title('Basement Area vs Sale Price')
    axes[0, 1].set_xlabel('Total Basement Area (sqft)')
    axes[0, 1].set_ylabel('Sale Price ($)')
    # OverallQual vs SalePrice (updated to fix warning)
    sns.boxplot(data=df, x='OverallQual', y='SalePrice',
                ax=axes[1, 0], hue='OverallQual', palette='viridis', legend=False)
    axes[1, 0].set_title('Overall Quality vs Sale Price')
    axes[1, 0].set_xlabel('Overall Quality (1-10)')
    axes[1, 0].set_ylabel('Sale Price ($)')
    # YearBuilt vs SalePrice
    sns.scatterplot(data=df, x='YearBuilt', y='SalePrice',
                    ax=axes[1, 1], color='purple', alpha=0.6)
    axes[1, 1].set_title('Year Built vs Sale Price')
    axes[1, 1].set_xlabel('Year Built')
    axes[1, 1].set_ylabel('Sale Price ($)')
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.close()
    return


def plot_categorical_plot(df):
    """Plot a categorical plot showing price distribution
    by neighborhood with enhanced formatting."""
    plt.figure(figsize=(16, 8))
    # Get top 15 neighborhoods by median price
    neighborhood_stats = df.groupby('Neighborhood')['SalePrice'].agg(['median', 'count'])
    top_neighborhoods = neighborhood_stats.nlargest(15, 'median').index
    # Filter data
    plot_data = df[df['Neighborhood'].isin(top_neighborhoods)]
    # Create color palette
    palette = sns.color_palette("husl", len(top_neighborhoods))
    # Create boxplot
    ax = sns.boxplot(
        data=plot_data,
        x='Neighborhood',
        y='SalePrice',
        order=top_neighborhoods,
        palette=palette,
        hue='Neighborhood',
        dodge=False,
        showfliers=False,
        width=0.8
    )
    # Add median value annotations
    medians = plot_data.groupby('Neighborhood')['SalePrice'].median().loc[top_neighborhoods]
    for i, (neighborhood, median) in enumerate(zip(top_neighborhoods, medians)):
        ax.text(
            i, 
            median + 5000, 
            f'${median:,.0f}', 
            ha='center', 
            va='bottom',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
        )
    # Formatting
    plt.title('Top 15 Neighborhoods by Median Home Price', fontsize=16, pad=20)
    plt.xlabel('Neighborhood', fontsize=14)
    plt.ylabel('Sale Price ($)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Create custom legend
    legend_patches = [
        plt.Rectangle((0,0), 1, 1, color=palette[i], alpha=0.6) 
        for i in range(len(top_neighborhoods))
    ]
    plt.legend(
        legend_patches,
        [f"{n} (n={neighborhood_stats.loc[n, 'count']})" for n in top_neighborhoods],
        title='Neighborhood (sample size)',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10
    )
    # Add summary statistics
    plt.annotate(
        f"Total homes: {len(plot_data):,}\nTime period: {df['YrSold'].min()}-{df['YrSold'].max()}",
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Plot a statistical violin-swarm plot showing home price distribution by decade built.
    Includes median markers and price trend line.
    """
    plt.figure(figsize=(16, 8)) 
    # Create decade bins (e.g., 1980s, 1990s)
    df['DecadeBuilt'] = (df['YearBuilt'] // 10) * 10
    decades = sorted(df['DecadeBuilt'].unique())
    decade_labels = [f"{decade}s" for decade in decades]
    # Create the base violin plot with proper hue assignment
    ax = sns.violinplot(
        data=df,
        x='DecadeBuilt',
        y='SalePrice',
        hue='DecadeBuilt',  # Explicit hue assignment
        palette="coolwarm",
        inner=None,         # No inner bars
        linewidth=1,
        saturation=0.8,
        legend=False        # We'll add custom legend
    )
    # Add swarm plot overlay to show individual homes
    sns.swarmplot(
        data=df,
        x='DecadeBuilt',
        y='SalePrice',
        color='black',
        alpha=0.15,
        size=3,
        ax=ax
    )
    # Calculate and plot medians
    medians = df.groupby('DecadeBuilt')['SalePrice'].median()
    for i, decade in enumerate(decades):
        ax.scatter(
            i, 
            medians.loc[decade], 
            color='white', 
            edgecolor='black',
            s=100,
            zorder=10,
            label='Median' if i == 0 else None
        )
    # Add price trend line
    sns.regplot(
        x=np.arange(len(decades)),
        y=medians.sort_index().values,
        scatter=False,
        color='darkred',
        line_kws={'linestyle':'--', 'alpha':0.7},
        ax=ax,
        label='Price Trend'
    )
    # Create custom legend for decades
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ax.collections[i].get_facecolor()[0], 
                           label=decade_labels[i]) 
                      for i in range(len(decades))]
    # Add legends
    plt.legend(
        handles=legend_elements,
        title='Decade Built',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    # Formatting
    plt.title('Home Price Distribution by Decade Built', fontsize=16, pad=20)
    plt.xlabel('Decade Built', fontsize=14)
    plt.ylabel('Sale Price ($)', fontsize=14)
    plt.xticks(range(len(decades)), decade_labels)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # Add statistical annotations
    plt.annotate(
        f"Total homes: {len(df):,}\nTime period: {df['YrSold'].min()}-{df['YrSold'].max()}\n"
        f"Correlation: {df['YearBuilt'].corr(df['SalePrice']):.2f}",
        xy=(0.02, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
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
    # Handle categorical columns
    categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
                       'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
                       'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                       'SaleType', 'SaleCondition']  
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    # Fill missing values for all columns
    cols = df.columns
    df = replace_null(df, cols)
    # Remove outliers from numerical columns
    def remove_outliers(data, column):
        if data[column].dtype in ['int64', 'float64']:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return data
    numeric_columns = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual',
                      'YearBuilt', 'GarageArea', 'LotArea'] 
    for column in numeric_columns:
        if column in df.columns:
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
        plt.plot(range(2, 11), inertia, marker='o', linestyle='--',
                 color='blue')
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
    # Determine optimal k (here we use k=3 based on elbow plot analysis)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    centres = kmeans.cluster_centers_  
    # Calculate silhouette score for optimal k
    final_score = calculate_silhouette_score(optimal_k)
    print(f'Final Silhouette Score for k={optimal_k}: {final_score:.2f}')
    return labels, df_scaled, centres[:, 0], centres[:, 1], labels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot the clustered data."""
    plt.figure(figsize=(10, 8))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis',
                s=50, alpha=0.6, label='Data Points')
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', s=200,
                label='Cluster Centers')
    plt.title('Cluster Visualization', fontsize=16)
    plt.xlabel(data.columns[0], fontsize=14)
    plt.ylabel(data.columns[1], fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig('clustering.png')
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression fitting on the dataset."""
    # Scale the data (only the feature variable)
    scaler = MinMaxScaler()
    x = df[col1].values.reshape(-1, 1)
    y = df[col2].values 
    # Scale x
    x_scaled = scaler.fit_transform(x)
    # Fit model
    model = LinearRegression()
    model.fit(x_scaled, y)
    # Predict
    y_pred = model.predict(x_scaled)
    # Get original x values for plotting
    x_original = x
    # Print model coefficients
    print(f"\nRegression results for {col1} vs {col2}:")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"R-squared: {model.score(x_scaled, y):.4f}")
    return x_original, y, y_pred


def plot_fitted_data(x, y, y_pred, col1, col2):
    """Plot the fitted data showing both
    original points and regression line."""
    plt.figure(figsize=(10, 6))
    # Plot original data points
    plt.scatter(x, y, c='blue', alpha=0.5, label='Actual Data')
    # Plot regression line
    sorted_idx = np.argsort(x.flatten())
    plt.plot(x[sorted_idx], y_pred[sorted_idx], 'r-', linewidth=2, label='Regression Line')
    plt.title(f'Linear Regression: {col1} vs {col2}', fontsize=16)
    plt.xlabel(col1, fontsize=14)
    plt.ylabel(col2, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fitting.png')
    plt.close()
    return


def main():
    # Load the dataset
    try:
        df = pd.read_csv("surprisehousing.csv")
    except FileNotFoundError:
        print("Error: Could not find 'housing_data.csv'")
        return 
    # Debugging: Print column names and first few rows
    print("\nColumn Names:", df.columns)
    print("\nFirst Five Rows:\n", df.head())
    print("\nSummary of numerical columns:\n", df.describe())
    # Strip leading/trailing spaces from column names (if any)
    df.columns = df.columns.str.strip()
    # Check if required columns exist
    required_columns = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'OverallQual']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the dataset. "
                          f"Available columns: {df.columns}")
    # Proceed with preprocessing
    print("\nPreprocessing data...")
    df = preprocessing(df)
    # Select a column for statistical analysis
    col = 'SalePrice'
    # Generate plots
    print("\nGenerating relational plot...")
    plot_relational_plot(df)
    print("Generating statistical plot...")
    plot_statistical_plot(df)
    print("Generating categorical plot...")
    plot_categorical_plot(df)
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    moments = statistical_analysis(df, col)
    writing(moments, col)
    # Perform clustering on GrLivArea vs SalePrice
    print("\nPerforming clustering...")
    clustering_results = perform_clustering(df, 'GrLivArea', 'SalePrice')
    plot_clustered_data(*clustering_results)
    # Perform fitting on GrLivArea vs SalePrice
    print("\nPerforming linear regression...")
    fitting_results = perform_fitting(df, 'GrLivArea', 'SalePrice')
    plot_fitted_data(*fitting_results, 'GrLivArea', 'SalePrice')
    print("\nAnalysis complete. All plots saved.")
    return


if __name__ == '__main__':
    main()
