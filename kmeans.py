import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from datetime import timedelta
from IPython.display import display

## data preparation

#__________________
# read the datafile
df_initial = pd.read_csv('data_ba.csv',encoding="utf-8")

print('\n <<< Cantidad de ejemplos y características >>>')
print(df_initial.shape)

print('\n <<< Primeras filas del dataset >>>')
print(df_initial.head(5))
print (display(df_initial[:5]))



print('\n <<< Información del dataset >>>')
print(df_initial.info())



print('\n <<< Información estadística del dataset >>>')
print(df_initial.describe())

df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])

print('Transacciones desde {} hasta {}'.format(df_initial['InvoiceDate'].min(),
                                    df_initial['InvoiceDate'].max()))


# --Agrupando datos por customerID--
# Creando TotalSum
df_initial['TotalSum'] = df_initial['Quantity'] * df_initial['UnitPrice']
# Creando snapshot date
snapshot_date = df_initial['InvoiceDate'].max() + timedelta(days=1)
print('snapshot_date',snapshot_date)
# Agrupando por CustomerID
data_process = df_initial.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSum': 'sum'})
# Renombrando las columnas
data_process.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

# Print top 5 rows and shape of dataframe
print(data_process.head())
print('{:,} rows; {:,} columns'
      .format(data_process.shape[0], data_process.shape[1]))

print('montos desde {} hasta {}'.format(data_process['MonetaryValue'].min(),
                                    data_process['MonetaryValue'].max()))

print ('información estadística RFM')
print(data_process.describe())


# # Plot RFM distributions
# plt.figure(figsize=(12,10))
# # Plot distribution of R
# plt.subplot(3, 1, 1); sns.distplot(data_process['Recency'])
# # Plot distribution of F
# plt.subplot(3, 1, 2); sns.distplot(data_process['Frequency'])
# # Plot distribution of M
# plt.subplot(3, 1, 3); sns.distplot(data_process['MonetaryValue'])
# # Show the plot
# #plt.show()

print ('################################### ')


# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x
# apply the function to Recency and MonetaryValue column
data_process['Recency'] = [neg_to_zero(x) for x in data_process['Recency']]
data_process['MonetaryValue'] = [neg_to_zero(x) for x in data_process['MonetaryValue']]
# unskew the data
rfm_log = data_process[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log, axis = 1).round(3)

# scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = data_process.index, columns = rfm_log.columns)

# the Elbow method
wcss = {}
data_process = np.nan_to_num(data_process)
rfm_scaled = np.nan_to_num(rfm_scaled)
for k in range(1, 11):
    kmeans = KMeans(n_clusters= k, init= 'k-means++', max_iter= 300)
    kmeans.fit(data_process)
    wcss[k] = kmeans.inertia_
# plot the WCSS values
# sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
# plt.xlabel('K Numbers')
# plt.ylabel('WCSS')
# plt.show()

# clustering
clus = KMeans(n_clusters= 3, init= 'k-means++', max_iter= 300)
clus.fit(data_process)
print ('clus.labels_',clus.labels_)
print ('clus.labels_',len(clus.labels_))
print ('data_process',len(data_process))
data_process['K_Cluster'] = clus.labels_
print ('data_process',len(data_process))
#data_process.assign(Cluster = clus.labels_)
print(data_process.head())

