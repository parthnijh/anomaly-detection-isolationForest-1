# Anomaly Detection using Isolation Forest

This project uses **Isolation Forest**, an unsupervised machine learning model, to detect anomalies (fraudulent transactions) in a credit card dataset.

---

## 📌 Steps Followed

1. Loaded the dataset from `data/creditcard.csv`  
2. Dropped the `Class` column to define features `X`
3. Scaled features using `StandardScaler` to prevent the `Amount` column from dominating
4. Trained the `IsolationForest` model to predict anomalies
5. Added the model's predictions to the dataframe (`-1` for anomaly, `1` for normal)
6. Reduced dimensionality from 30 features to 2 using PCA for visualization
7. Created scatter plot: blue = normal, red = anomaly

---

## 📈 Visualization

Used PCA to convert high-dimensional features into 2D:

```python
plt.scatter(inliers["pca1"], inliers["pca2"], c="blue", s=3, alpha=0.3)
plt.scatter(outliers["pca1"], outliers["pca2"], c="red", s=8, alpha=0.8)
