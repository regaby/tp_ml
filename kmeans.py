# Autora: Ing. Gabriela Rivero (regaby@gmail.com)
# Agosto 2021

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.core.accessor import register_dataframe_accessor
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
from datetime import timedelta
from IPython.display import display
from sklearn.manifold import TSNE
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline

## data preparation

#__________________
# read the datafile
df_initial = pd.read_csv('data_ba.csv',encoding="utf-8")

print('\n <<< Cantidad de ejemplos y características >>>')
print(df_initial.shape)

print('\n <<< Primeras filas del dataset >>>')
print(df_initial.head(5))
#print (display(df_initial[:5]))



print('\n <<< Información del dataset >>>')
print(df_initial.info())

print('{:,} transacciones no tiene CustomerID'
      .format(df_initial[df_initial.CustomerID.isnull()].shape[0]))



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
#print(data_process.head())
print('{:,} rows; {:,} columns'
      .format(data_process.shape[0], data_process.shape[1]))

print('montos desde {} hasta {}'.format(data_process['MonetaryValue'].min(),
                                    data_process['MonetaryValue'].max()))

print ('información estadística RFM')
print(data_process.describe())

def plot_rfm(dataset):
    # Plot RFM distributions
    plt.figure(figsize=(12,10))
    # Plot distribution of R
    plt.subplot(3, 1, 1); sns.distplot(dataset['Recency'])
    # Plot distribution of F
    plt.subplot(3, 1, 2); sns.distplot(dataset['Frequency'])
    # Plot distribution of M
    plt.subplot(3, 1, 3); sns.distplot(dataset['MonetaryValue'])
    # Show the plot
    plt.show()

print ("\nDistribución RFM - sin normalizar")
# plot_rfm(data_process)

print ('Normalizo el dataset')
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

## Dataset sin normalizar
data_process_array = np.nan_to_num(data_process)
## Dataset normalizado
print ("\nDistribución RFM - Normalizada")
# plot_rfm(rfm_scaled)
rfm_scaled = np.nan_to_num(rfm_scaled)



def get_elbow(dataset):
    # the Elbow method
    wcss = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters= k, init= 'k-means++', max_iter= 300)
        kmeans.fit(dataset)
        wcss[k] = kmeans.inertia_
    # plot the WCSS values
    sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
    plt.xlabel('K Numbers')
    plt.ylabel('WCSS')
    plt.show()

# elbow sin normalizar
print ('\nElbow con dataset sin normalizar')
# get_elbow(data_process_array)
# elbow normalizado
print ('\nElbow con dataset normalizado')
# get_elbow(rfm_scaled)



############################################################## metricas

def bench_k_means(init, data, test_label, clusters_number):
    """Benchmark to evaluate the KMeans initialization methods.
    """
    t0 = time()

    clus = KMeans(n_clusters= clusters_number, init= init, max_iter= 300)

    clus.fit(data)
    labels = clus.labels_
    # print ('clus.labels_',np.unique(clus.labels_))

    estimator = make_pipeline(StandardScaler(), clus).fit(data)
    fit_time = time() - t0
    results = [test_label, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_,
                                 metric="euclidean", sample_size=300,)
    ]

    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))
    return labels


def plt_cluster(labels, original_df_rfm,test_label):
    t0 = time()
    # Create a cluster label column in original dataset
    df_new = original_df_rfm.assign(Cluster = labels)

    # Initialise TSNE
    model = TSNE(random_state=1)
    df_new_array = np.nan_to_num(df_new)
    transformed = model.fit_transform(df_new_array)

    # Plot t-SNE
    #plt.title('{} Clusters- {}'.format(clusters_number, test_label))
    plt.title('{}'.format(test_label))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=labels, style=labels, palette="Set1")
    plot_time = time() - t0
    print ('Plot time {} {}'.format(plot_time, test_label))

print ('\nTest 1: cluster 3-DS no normalizado y normalizado')
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
label0 = bench_k_means(init="k-means++", data=data_process_array, test_label="DS s/norm.", \
    clusters_number=3)
label1 = bench_k_means(init="k-means++", data=rfm_scaled, test_label="DS Norma.", \
    clusters_number=3)
data_process['K_Cluster_SN'] = label0
data_process['K_Cluster_N'] = label1

print ('K_Cluster_SN', data_process['K_Cluster_SN'].value_counts())
print ('K_Cluster_N', data_process['K_Cluster_N'].value_counts())

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt_cluster(labels=label0, original_df_rfm= data_process, test_label="DS s/norm.")
plt.subplot(2, 1, 2)
plt_cluster(labels=label1, original_df_rfm= data_process, test_label="DS norm.")
plt.tight_layout()
plt.show()

 # test 2
print ('\nTest 2: cluster 3-DS normalizado-diferentes init')
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

label31 = bench_k_means(init="k-means++", data=data_process_array, test_label="K=3 DSN km++", \
    clusters_number=3)
label32 = bench_k_means(init="random", data=data_process_array, test_label="K=3 DSN random", \
    clusters_number=3)
pca = PCA(n_components=3).fit(data_process_array)
label33 = bench_k_means(init=pca.components_, data=data_process_array, test_label="K=3 DSN PCA", \
    clusters_number=3)
