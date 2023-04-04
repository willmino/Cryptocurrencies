# Cryptocurrencies
Unsupervised Machine Learning Models

## Purpose
An unsupervised Machine Learning (ML) model was used to help an investing bank proive a cryptocurrency investment portfolio for its customers.
I created a report that includes cryptocurrencies currently being traded on the market. Each currency was classified in a distinct risk group
based on several parameters of the dataset including: total coins mined, total coin supply, block chain algorithm, and prooftype (proof of stake; for transcation verification).

Unsupervised ML is used typically when trends need to be found in a data set. When we eliminate the supervisor of our data, the target variable, or sometimes not evening having a target variable,
we need to determine what question to ask about our data set so we can determine what trends may arise.

One effective way to discover trends using unsupervised ML is dimensionality reduction.
The process of principal component analysis serves to reduce the dimensions of a data set to contain the data with the highest variance.
PCA is accomplished by a series of transformation involving the plotting of variance and covariance values of a dataset.
Once these transformations and plots are accomplished, a new coordinate system called the prinicipal component space is established.
Within the principal component space are eigenvectors representing the direction of spread of data. These eigenvectors have an eigenvalue which quantifies
the magnitude of the variance in a particular direction.
The key takeaway is that we want to extract a subset of data from the principal component space which contains the most variation of data.
A high variation of data indicates that we are more likely to find interesting information in the dataset.
Thus, we use prinicpal component analysis to determine the proportions of data which have the most useful information for us.


## Analysis

To determine which cryptocurrencies were the most promising investments for the bank's customers, I first loaded the crpyto_data.csv file into python Pandas.
I cleaned the dataset by eliminating any data with negative values for "TotalCoinsMined". I also only selected cryptocurrncies that were currently being mined.
After cleaning the data, it was determined there were 532 traded Cryptocurrencies available.

The `get_dummies()` function was then performed on the dataset to numerically encode string values. This allowed the dataset to be compatible with our KMeans clustering algorithm.
The data was also scaled using the `StandardScaler.fit_transform()` function.

After scaling the data, Principal Component Analysis (PCA) was performed to reduce the dimensionality of the data.
The entire data set was reduced to 3 principal components. 

`pca = PCA(n_components=3)`

`df_pca = pca.fit_transform(X)`

Once we had our data transformed, it was time to actually DO SOMETHING with the data.
The unsupervised clustering ML algorithm began here with KMeans. We needed to group the data in some way to observe any kindof phenomena.
We would need to first cluster our data points so we could later return to the initial dataset and find interesting relationships among specific independent variables.

This clustering was accomplished with the below code.
First, a triage of KMeans clustering `k` values was determined using a for loop. The range of k values was set from 1 to 10.

`for i in k:`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`km = KMeans(n_clusters=i, random_state=0)`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`km.fit(pcs_df)`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`inertia.append(km.inertia_)`

The data was plotted using hvplot

`elbow_data = {"k": k, "inertia": inertia}`

`df_elbow = pd.DataFrame(elbow_data)`


`df_elbow.hvplot.line(x='k', y='inertia', title='Elbow Curve')`

![]()