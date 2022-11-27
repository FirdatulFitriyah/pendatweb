import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Choirinnisa' Fitria ")
st.write("##### Nim   : 2004111000149 ")
st.write("##### Kelas : Penambangan Data C ")
description, upload_data, preporcessing, modeling, implementation = st.tabs(["Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with description:
    st.write("""# Deskripsi Data""")
    st.write("Dataset yang digunakan adalah Cirrhosis Prediction Dataset yang diambil dari https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset")
    st.write("Total datanya adalah 418 data")
    st.write("Informasi Atribut")
    st.write("1) ID: pengidentifikasi unik")
    st.write("2) N_Days: jumlah hari antara pendaftaran dan kematian yang lebih awal, transplantasi, atau waktu analisis studi pada Juli 1986")
    st.write("3) Status: status pasien C (disensor), CL (disensor karena tx hati), atau D (meninggal)")
    st.write("4) Obat : jenis obat D-penicillamine atau placebo")
    st.write("5) Umur: umur dalam [hari]")
    st.write("6) Jenis Kelamin: M (laki-laki) atau F (perempuan)")
    st.write("7) Asites: adanya asites N (Tidak) atau Y (Ya)")
    st.write("8) Hepatomegali: adanya hepatomegali N (Tidak) atau Y (Ya)")
    st.write("9) Laba-laba: keberadaan laba-laba N (Tidak) atau Y (Ya)")
    st.write("10) Edema: adanya edema N (tidak ada edema dan tidak ada terapi diuretik untuk edema), S (ada edema tanpa diuretik, atau edema teratasi dengan diuretik), atau Y (edema meskipun dengan terapi diuretik)")
    st.write("11) Bilirubin: bilirubin serum dalam [mg/dl]")
    st.write("12) Kolesterol: kolesterol serum dalam [mg/dl]")
    st.write("13) Albumin: albumin dalam [gm/dl]")
    st.write("14) Tembaga: tembaga urin dalam [ug/hari]")
    st.write("15) Alk_Phos: alkaline phosphatase dalam [U/liter]")
    st.write("16) SGOT: SGOT dalam [U/ml]")
    st.write("17) Trigliserida: trigliserida dalam [mg/dl]")
    st.write("18) Trombosit: trombosit per kubik [ml/1000]")
    st.write("19) Protrombin: waktu protrombin dalam detik [s]")
    st.write("20) Stadium: stadium histologis penyakit (1, 2, 3, atau 4)")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/choirinnisafitria/Web-Data-Mining ")

with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    df[["ID", "N_Days", "Status", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders","Edema","Bilirubin","Cholesterol","Albumin","Copper","Alk_Phos","SGOT","Tryglicerides","Platelets","Prothrombin"]].agg(['min','max'])

    df.ID.value_counts()
    # df = df.drop(columns=["date"])

    X = df.drop(columns="ID")
    y = df.ID
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.ID).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.ID).columns.values.tolist()
    
    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape

with modeling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)

with implementation:
    st.write("# Implementation")
    Snoring_Rate = st.number_input('Masukkan tingkat mendengkur : ')
    Respiration_Rate = st.number_input('Masukkan laju respirasi : ')
    Body_Temperature = st.number_input('Masukkan suhu tubuh : ')
    Limb_Movement = st.number_input('Masukkan gerakan ekstremitas : ')
    Blood_Oxygen = st.number_input('Masukkan oksigen darah : ')
    Eye_Movement = st.number_input('Masukkan gerakan mata : ')
    Sleeping_Hours = st.number_input('Masukkan jam tidur : ')
    Heart_Rate = st.number_input('Masukkan detak jantung : ')

    def submit():
        # input
        inputs = np.array([[
            Snoring_Rate,
            Respiration_Rate,
            Body_Temperature,
            Limb_Movement,
            Blood_Oxygen,
            Eye_Movement,
            Sleeping_Hours,
            Heart_Rate
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka Deteksi Stres Manusia di dalam dan melalui Tidur : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

