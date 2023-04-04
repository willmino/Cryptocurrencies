# Cryptocurrencies
Unsupervised Machine Learning Models

## Purpose
An unsupervised Machine Learning (ML) model was used to help an investing bank provide a cryptocurrency investment portfolio for its customers.
I created a report that includes cryptocurrencies currently being traded on the market. Each currency was classified in a distinct risk group
based on several parameters of the dataset including: total coins mined, total coin supply, block chain algorithm, and prooftype (proof of stake; for transaction verification).

Unsupervised ML is used typically when trends need to be found in a data set. When we eliminate the supervisor of our data, the target variable, or sometimes not evening having a target variable,
we need to determine what question to ask about our data set so we can determine what trends may arise.

### A note on Dimensionality Reduction

One effective way to discover trends using unsupervised ML is dimensionality reduction.
The process of principal component analysis serves to reduce the dimensions of a data set to contain the data with the highest variance.
PCA is accomplished by a series of transformations involving the plotting of variance and covariance values of a dataset.
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

When the k value triage plot, known as the "Elbow Curve", was plotted, we could begin to determine the number of ideal clusters for this dataset.
A rule of thumb for determining the k value, the number of clusters needed for the data, is to observe when the curve begins to change towards a flat line.
In the Elbow Curve plotted below, we can see the curve begins to look flat first at the value of k=5.
Thus, k = 5 was chosen and we would then cluster the data into 5 Classes.

![elbow_curve](https://github.com/willmino/Cryptocurrencies/blob/main/images/elbow_curve.png)

### KMeans Clustering

KMeans clustering was then carried out with k =5, for  clustering into 5 Classes.

`model = KMeans(n_clusters=5, random_state=42)`

`model.fit(pcs_df)`

`predictions = model.predict(pcs_df)`

The predictions for clustering were finalized to a list of classes. This "Class" list contained numerical labels for each cluster group
ranging from 0 to 4 (a total of 5 groups).

The clusters were plotted in 3D based on each Principal Component (PC). PC1 was plotted on the X axis. PC2 was plotted on the Y axis. PC3 was plotted on the Z axis..

Below is the code for the 3D plot:

`import plotly.express as px`

`fig = px.scatter_3d(`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     clustered_df,`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     x="PC 1",`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     y="PC 2",`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     z="PC 3",`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     color="Class",`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     symbol="Class",`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     hover_name = "CoinName",`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     hover_data = ["Algorithm"],`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`     width=800,`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;` )`

`fig.update_layout(legend=dict(x=0, y=1))`

`fig.show()`


![3D_plot](https://github.com/willmino/Cryptocurrencies/blob/main/images/3D_plot.png)

We can clearly see the five different clustering groups represented in the 3D plot above. 
They are denoted by the colors dark purple, light purple, salmon, orange, and yellow.

### Observing Trends in the Data

After going back to our clustered dataset, we looked at the non scaled data along with the Classes that were assigned to each cluster.

For example, attempts were made to observe the relationship among clusters when viewing the "TotalCoinsMined" ad "TotalCoinSupply" independent variables.

Here is the plot of the scaled 2D data.

![2D_plot](https://github.com/willmino/Cryptocurrencies/blob/main/images/2D_plot.png)s

We can see that on a 2D plot, Classes 0 and 1 are clustered very similarly.

`mined_df.groupby("Class").mean()["TotalCoinsMined"]`

To attempt to observe the similarties between these clusters, I looked at different summary statistics of the data.
For example, grouping this data set by "Class" and looking at mean as the aggregation function of "TotalCoinsMined" reveals that
Classes 0 and 1 had a similar number of mined coins. `Class 0: 2.38E09`. `Class 1: 2.25E09`. These are in the range of billions of coins.
The similarity between these two Classes is apparent in the 2D plot.

We can see that Classes 3 and 4 exhibited average "TotalCoinsMined" values in the range of 1.16E10 to 4.08E10, the next most similar range. 
Classes 3 and 4 are also visually close together in terms of their clustering in the 2D plot. 
Class 2 is a bit of an outlier in this data set. Its "TotalCoinsMined" value is at 9.90E11. This is in the range of hundreds of billions. This is the most unique cluster out of all the data because it is only one point.
Thus, Class 2 is depicted as the most distant from any of the other Class.

![grouped_by_class_mean_aggregation](https://github.com/willmino/Cryptocurrencies/blob/main/images/Class.png)

One other trend I wanted to observe in the data set was that if clustering was influenced by the "ProofType" of each cryptocurrency. I avoided looking at "Algorithm" among the data
because there were so many unique algorithms in each "Class".
To observe the different prooftypes of each class I used the following code: `mined_df.loc[mined_df["Class"]==0]["ProofType"].unique()` and I changed the numerical input in the `==0` argument.

So if I wanted to look at class 1, I would write `mined_df.loc[mined_df["Class"]==1]["ProofType"].unique()` instead.

I was able to observe a small list of common prooftypes among the classes.
In fact, Class 0 had the most common and also limited number of prooftype stakes. The prooftypes for Class 0 were 'PoW' and 'dPoW/PoW'.
I found this interesting in the dataset because there were about 233 cryptos with either of these prooftypes for staking.
Since it was so common, it seems like this prooftype is the most favorable for a crypto currency. Upon further examination of the coins in the dataset,
it was revealed that Class 0 contained Bitcoin, Ethereum, and Litecoin, three of the largest and most stable crypto currencies to date. Class 0 with all of this in mind
made a favorable case for its potential attractiveness to the bank's customers.

## Conclusion

I was tasked with generating this report for Accountability Accounting to prepare a portfolio of the most favorable cryptocurrencies in which to invest.
The bank had no knowledge of the crypto sector and it was up to me to determine a rationale for why certain cryptocurrencies were important for customers.
Scarcity is one of the main attractors for cryptocurrency, since only a limited number of coins can be made. Its important for investors
to know that their crypto investment should be in a coin that has as high scarcity as practically possible. In this way, the most value can be accumulated over time due to scarcity.

From the unsupervised machine learning analysis on the Cryptocurrency dataset, we attempted to observe some trends in the data.
It appeared that the clustering model showed several classes of clusters with similar data.
From the clusters of the most similar cryptos in the dataset, Cluster 0 exhibited the lowest average "TotalCoinsMined" value together with the lowest "TotalCoinSupply" value.
While many of these cryptos are available, customers will be pleased to know that additional coins will not be made once the final "Bitcoin" (for example) is mined.
This allows for scarcity and increasing value over time to be established for these cryptocurrencies.

It was also interesting to see that the KMeans algorithm used in this analysis was able to accurately cluster the largest and most successful cryptocurrencies into the same group, Class 0.
Additionally Class 0 contained only two types of proof of stake mechanisms ("prooftype" column). The most credible cryptocurrencies
in this group were Bitcoin, Ethereum, and Litecoin, among others. With most of the coins using the strong and popular 'PoW' proof of stake mechanism,
Class 0 looked like the most credible Class of cryptos for customers to invest in.
Thus, my recommended cryptocurrency portfolio is comprised of Class 0 investments (Bitcoin, Ethereumm Litecoin, etc.) which will serve as highly attractive long term investments for the accumulation of value due to scarcity for Accountability Accounting's customers.
