# Coffee Quality Identification - Google Colab Notebook

## Instructions:
1. Upload `Dataset_6.csv` to `/content/` folder in Google Colab
2. Run each section one by one
3. All sections are clearly marked and commented

---

## Section 1: Import Libraries and Load Data

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")

# Load the dataset
df = pd.read_csv('/content/Dataset_6.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
df.head()
```

---

## Section 2: Data Exploration

```python
# Check data types and missing values
print("Dataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum().sum(), "total missing values")

# Identify quality-related columns
quality_columns = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 
                   'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 
                   'Overall', 'Total Cup Points']

print("\nQuality metrics columns:")
for col in quality_columns:
    if col in df.columns:
        print(f"  - {col}")

# Display basic statistics
print("\nBasic Statistics:")
df[quality_columns].describe()

# Visualize distribution of Total Cup Points
plt.figure(figsize=(10, 6))
plt.hist(df['Total Cup Points'].dropna(), bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Total Cup Points')
plt.ylabel('Frequency')
plt.title('Distribution of Coffee Quality (Total Cup Points)')
plt.axvline(df['Total Cup Points'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df["Total Cup Points"].mean():.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Average Total Cup Points: {df['Total Cup Points'].mean():.2f}")
print(f"Highest Score: {df['Total Cup Points'].max():.2f}")
print(f"Lowest Score: {df['Total Cup Points'].min():.2f}")
```

---

## Section 3: Data Preprocessing

```python
# Select quality metric columns for clustering
clustering_features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 
                      'Balance', 'Uniformity', 'Clean Cup', 'Sweetness', 'Overall']

# Create clean dataset
df_cluster = df[clustering_features].copy()

# Check for missing values
print("Missing values:")
print(df_cluster.isnull().sum())

# Remove rows with missing values
df_cluster_clean = df_cluster.dropna()

print(f"Original: {len(df_cluster)} rows")
print(f"After cleaning: {len(df_cluster_clean)} rows")

# Keep Total Cup Points for analysis
df_total_points = df.loc[df_cluster_clean.index, 'Total Cup Points']

# Standardize the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster_clean)

print("Data standardized successfully!")
print(f"Scaled data shape: {X_scaled.shape}")
```

---

## Section 4: Find Optimal Number of Clusters

```python
# Use Elbow Method to find optimal number of clusters
inertias = []
silhouette_scores = []
K_range = range(2, 11)  # Test from 2 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal k
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters: {optimal_k}")
print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")
```

---

## Section 5: Apply K-Means Clustering

```python
# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels
df_cluster_clean['Cluster'] = cluster_labels
df_cluster_clean['Total Cup Points'] = df_total_points.values

print(f"K-Means clustering completed with {optimal_k} clusters!")
print("\nCluster distribution:")
print(df_cluster_clean['Cluster'].value_counts().sort_index())

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"\nAverage Silhouette Score: {silhouette_avg:.3f}")

# Analyze cluster characteristics
print("\nCluster Analysis:")
cluster_analysis = df_cluster_clean.groupby('Cluster')[clustering_features + ['Total Cup Points']].mean()
print(cluster_analysis.round(2))

# Visualize cluster characteristics
plt.figure(figsize=(14, 6))
cluster_analysis[clustering_features].T.plot(kind='bar')
plt.title('Average Quality Metrics by Cluster')
plt.xlabel('Quality Metrics')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Analyze quality by cluster
plt.figure(figsize=(10, 6))
cluster_quality = df_cluster_clean.groupby('Cluster')['Total Cup Points'].agg(['mean', 'std', 'count'])
plt.bar(cluster_quality.index, cluster_quality['mean'], 
        yerr=cluster_quality['std'], capsize=5, alpha=0.7, edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Average Total Cup Points')
plt.title('Average Coffee Quality by Cluster')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

---

## Section 6: Visualize Clusters using PCA

```python
# Use PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                     cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', 
           s=200, label='Cluster Centers', edgecolors='black', linewidth=2)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Coffee Quality Clusters - PCA Visualization')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
```

---

## Section 7: Hierarchical Clustering (Alternative)

```python
# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

df_cluster_clean['Hierarchical_Cluster'] = hierarchical_labels

print(f"Hierarchical clustering completed!")
print("\nCluster distribution:")
print(df_cluster_clean['Hierarchical_Cluster'].value_counts().sort_index())

# Compare methods
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
print(f"\nHierarchical Silhouette Score: {hierarchical_silhouette:.3f}")
print(f"K-Means Silhouette Score: {silhouette_avg:.3f}")

# Visualize both methods
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                cmap='viridis', s=50, alpha=0.6)
axes[0].set_title('K-Means Clustering')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, 
                cmap='plasma', s=50, alpha=0.6)
axes[1].set_title('Hierarchical Clustering')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Section 8: Insights and Patterns

```python
# Categorize quality
def categorize_quality(points):
    if points >= 90:
        return 'Exceptional (90+)'
    elif points >= 85:
        return 'Excellent (85-90)'
    elif points >= 80:
        return 'Very Good (80-85)'
    else:
        return 'Good (<80)'

df_cluster_clean['Quality_Category'] = df_cluster_clean['Total Cup Points'].apply(categorize_quality)

# Analyze cluster composition
print("Cluster Composition by Quality Category:")
quality_by_cluster = pd.crosstab(df_cluster_clean['Cluster'], df_cluster_clean['Quality_Category'])
print(quality_by_cluster)

# Visualize
quality_by_cluster.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Quality Category Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.legend(title='Quality Category')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Top metrics by cluster
print("\nTop Quality Metrics by Cluster:")
for cluster_id in sorted(df_cluster_clean['Cluster'].unique()):
    cluster_data = df_cluster_clean[df_cluster_clean['Cluster'] == cluster_id]
    avg_scores = cluster_data[clustering_features].mean().sort_values(ascending=False)
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Average Total Cup Points: {cluster_data['Total Cup Points'].mean():.2f}")
    print(f"  Top 3 Metrics:")
    for i, (metric, score) in enumerate(avg_scores.head(3).items(), 1):
        print(f"    {i}. {metric}: {score:.2f}")
```

---

## Section 9: Final Summary

```python
# Final Summary
print("="*60)
print("COFFEE QUALITY CLUSTERING ANALYSIS - SUMMARY")
print("="*60)

print(f"\nDataset: {len(df_cluster_clean)} coffee samples")
print(f"Features: {len(clustering_features)} quality metrics")
print(f"Optimal clusters: {optimal_k}")
print(f"Silhouette Score: {silhouette_avg:.3f}")

print("\nCluster Characteristics:")
for cluster_id in sorted(df_cluster_clean['Cluster'].unique()):
    cluster_data = df_cluster_clean[df_cluster_clean['Cluster'] == cluster_id]
    avg_quality = cluster_data['Total Cup Points'].mean()
    count = len(cluster_data)
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Samples: {count} ({count/len(df_cluster_clean)*100:.1f}%)")
    print(f"  Average Quality: {avg_quality:.2f}")

print("\nKey Insights:")
print("1. Coffee samples grouped into distinct quality clusters")
print("2. Each cluster has unique quality characteristics")
print("3. Clustering helps identify quality patterns")
print("4. Can guide quality assessment decisions")

# Comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# 1. Cluster distribution
ax1 = plt.subplot(2, 3, 1)
cluster_counts = df_cluster_clean['Cluster'].value_counts().sort_index()
ax1.bar(cluster_counts.index, cluster_counts.values, alpha=0.7)
ax1.set_title('Samples per Cluster')
ax1.set_xlabel('Cluster')
ax1.set_ylabel('Count')

# 2. Average quality
ax2 = plt.subplot(2, 3, 2)
avg_quality = df_cluster_clean.groupby('Cluster')['Total Cup Points'].mean()
ax2.bar(avg_quality.index, avg_quality.values, alpha=0.7, color='green')
ax2.set_title('Average Quality by Cluster')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Total Cup Points')

# 3. PCA visualization
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=30, alpha=0.6)
ax3.set_title('Cluster Visualization (PCA)')
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')

# 4. Quality distribution
ax4 = plt.subplot(2, 3, 4)
ax4.hist(df_cluster_clean['Total Cup Points'], bins=25, alpha=0.7, color='orange')
ax4.set_title('Overall Quality Distribution')
ax4.set_xlabel('Total Cup Points')
ax4.set_ylabel('Frequency')

# 5. Boxplot by cluster
ax5 = plt.subplot(2, 3, 5)
cluster_quality_data = [df_cluster_clean[df_cluster_clean['Cluster'] == i]['Total Cup Points'].values 
                       for i in sorted(df_cluster_clean['Cluster'].unique())]
ax5.boxplot(cluster_quality_data, labels=sorted(df_cluster_clean['Cluster'].unique()))
ax5.set_title('Quality Distribution by Cluster')
ax5.set_xlabel('Cluster')
ax5.set_ylabel('Total Cup Points')

# 6. Heatmap
ax6 = plt.subplot(2, 3, 6)
cluster_metrics = df_cluster_clean.groupby('Cluster')[clustering_features].mean()
sns.heatmap(cluster_metrics.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax6)
ax6.set_title('Quality Metrics Heatmap')

plt.suptitle('Coffee Quality Clustering Analysis - Overview', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nAnalysis Complete!")
```

