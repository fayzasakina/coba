# import streamlit
import streamlit as st
st.title("K-Means Clustering Autentikasi Catatan Bank")

# baca data asli
import pandas as pd
import numpy as np
data_asli = pd.read_csv('BankNote_Authentication.csv')

if st.checkbox('Tampilkan Data Asli'):
    st.subheader('Data Asli')
    st.write(data_asli)

# visualisasi data asli
import matplotlib.pyplot as plt
if st.checkbox('Tampilkan Grafik - Grafik Data Asli'):
    st.subheader('Grafik Line')
    st.line_chart(data_asli)
    st.subheader('Grafik Histogram')
    df = pd.DataFrame(data_asli, columns = ['variance', 'skewness', 'curtosis', 'entropy'])
    df.hist()
    plt.show()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

# baca data preproses
data_normalisasi = pd.read_csv('data_normalisasi_bank.csv')
data_normalisasi = data_normalisasi.drop('noname', axis=1)
if st.checkbox('Tampilkan Data Hasil Pre-Prosessing'):
    st.subheader('Data Pre-Prosessing')
    st.write("(columns={0: 'variance', 1: 'skewness', 2: 'curtosis', 3: 'entropy'})")
    st.write(data_normalisasi)

# k-means clustering
if st.checkbox('Tampilkan Proses dan Hasil K-Means Clustering (Sudah Dilakukan)'):
    st.subheader('Proses dan Hasil K-Means Clustering')
    import sklearn.cluster 
    ## menentukan dan mengkonfigurasi fungsi kmeans
    kmeans_data = sklearn.cluster.KMeans(n_clusters=7, random_state=101).fit(data_normalisasi)
    ## mencari nilai pusat dari masing - masing cluster
    st.write('nilai pusat masing-masing cluster', kmeans_data.cluster_centers_)
    ## menampilkan hasil kluster
    st.write('hasil kluster', kmeans_data.labels_)
    ## menambahkan kolom "kluster" dalam data
    data_tercluster = data_normalisasi
    data_tercluster["kluster"] = kmeans_data.labels_
    st.dataframe(data_tercluster)

# evaluasi model / kmeans clustering
    import sklearn.metrics
    st.write("Evaluasi hasil K-Means Clustering")
    st.write(sklearn.metrics.silhouette_score(data_tercluster, kmeans_data.labels_))