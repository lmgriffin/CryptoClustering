# CryptoClustering

### CODE GIVEN FROM MODULES

# Define a DataFrame to hold the values for k and the corresponding inertia
elbow_data = {"k": k, "inertia": inertia}

# Create the DataFrame from the elbow data
df_elbow = pd.DataFrame(elbow_data)

# Review the DataFrame
df_elbow.head()

 ----------------------------------
# Plot the DataFrame
df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)

------------------------------------

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot.line(x="k", y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot_pca


---------------------

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_stocks_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_stocks_pca)
    inertia.append(model.inertia_)
    
    ----------------------------

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inertia = []

------------------------------------

# Initialize the K-Means model with n_clusters=3
model = KMeans(n_clusters=3)

# Fit the model for the df_stocks_pca DataFrame
model.fit(df_stocks_pca)

# Predict the model segments (clusters)
stock_clusters = model.predict(df_stocks_pca)

# Print the stock segments
print(stock_clusters)

---------------------------
# Create the scatter plot with x="PC1" and y="PC2"
df_stocks_pca_predictions.hvplot.scatter(
    x="PC1",
    y="PC2",
    by="StockCluster",
    title = "Scatter Plot by Stock Segment - PCA=2"
)

-----------------------------------------

# Creating a DataFrame with the PCA data
df_stocks_pca = pd.DataFrame(stocks_pca_data, columns=["PC1", "PC2"])

# Copy the tickers names from the original data
df_stocks_pca["Ticker"] = df_stocks.index

# Set the Ticker column as index
df_stocks_pca = df_stocks_pca.set_index("Ticker")

# Review the DataFrame
df_stocks_pca.head()

-----------------------------------

# Fit the df_stocks_scaled data to the PCA
stocks_pca_data = pca.fit_transform(df_stocks_scaled)

# Review the first five rose of the PCA data
# using bracket notation ([0:5])
stocks_pca_data[:5]

# Create the PCA model instance where n_components=2
pca = PCA(n_components=2)

--------------------------------

# Define the model Kmeans model using the optimal value of k for the number of clusters.
model = KMeans(n_clusters=3, random_state=0)

# Fit the model
model.fit(customers_pca_df)

# Make predictions
k_3 = model.predict(customers_pca_df)

# Create a copy of the customers_pca_df DataFrame
customer_pca_predictions_df = customers_pca_df.copy()

# Add a class column with the labels
customer_pca_predictions_df["customer_segments"] = k_3