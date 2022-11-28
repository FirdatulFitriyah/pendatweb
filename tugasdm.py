from pyrsistent import v
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
st.write("##### Nama  : Firdatul Fitriyah ")
st.write("##### Nim   : 200411100020")
st.write("##### Kelas : Penambangan Data C ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Heart Failure Prediction (Prediksi Gagal Jantung) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data")
    st.write(" Penyakit kardiovaskular (CVDs) adalah penyebab kematian nomor 1 secara global, merenggut sekitar 17,9 juta nyawa setiap tahun, yang merupakan 31% dari semua kematian di seluruh dunia. Gagal jantung adalah kejadian umum yang disebabkan oleh CVD dan kumpulan data ini berisi 12 fitur yang dapat digunakan untuk memprediksi kematian akibat gagal jantung. ")
    st.write(" Sebagian besar penyakit kardiovaskular dapat dicegah dengan mengatasi faktor risiko perilaku seperti penggunaan tembakau, pola makan yang tidak sehat dan obesitas, kurangnya aktivitas fisik, dan penggunaan alkohol yang berbahaya dengan menggunakan strategi populasi luas. ")
    st.write(" Orang dengan penyakit kardiovaskular atau yang memiliki risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia, atau penyakit yang sudah ada) memerlukan deteksi dan penanganan dini di mana model pembelajaran mesin dapat sangat membantu. ")
    st.write("""# Deskripsi Data""")
    st.write("Total datanya adalah 299 data pasien")
    st.write("Informasi Atribut")
    st.write("1) Age : Umur dari pasien penyakit gagal jantung ")
    st.write("2) Anemia : Menurunnya hemoglobin atau sel darah merah dalam tubuh (1 = Ya, 0 = Tidak) ")
    st.write("3) High Blood Pressure : Hipertensi ")
    st.write("4) Creatinine Phosphoki nase (CPK) : Enzim CPK dalam darah ")
    st.write("5) Diabetes : Pasien menderita diabetes atau tidak (1 = Ya, 0 = Tidak) ")
    st.write("6) Ejection Fraction : Volume darah yang mengalir meninggalkan jantung setiap jantung berkontraksi ")
    st.write("7) Platelets : Jumlah trombosit dalam tubuh ")
    st.write("8) Sex : Jenis Kelamin (1 = Laki-Laki, 0 = Perempuan ")
    st.write("9) Serum Creatinine : Jumlah kreatinin serum yang terdapat pada darah ")
    st.write("10) Serum Sodium : Jumlah natrium serum yang terdapat pada darah ")
    st.write("11) Smoking : Perokok atau tidak perokok (1 = Ya, 0 = Tidak)")
    st.write("12) Time : Periode tindak lanjut (6 hari)")
    st.write("13) (Target) Death Event  : Pasien yang telah meninggal dalam masa tindak lanjut (1 = Gagal Jantung, 0 = Tidak Gagal jantung)")

    st.write("###### Aplikasi ini untuk : Heart Failure Prediction (Prediksi Gagal Jantung) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link :https://github.com/FirdatulFitriyah/pendatweb  ")

with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    df[["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking",	"time",	"DEATH_EVENT"]].agg(['min','max'])

    df.DEATH_EVENT.value_counts()
    #df = df.drop(columns=["date"])

    X = df.drop(columns="DEATH_EVENT")
    y = df.DEATH_EVENT
    "### Membuang fitur yang tidak diperlukan"
    df

    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.DEATH_EVENT).columns.values.tolist()

    "### Label"
    labels

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.DEATH_EVENT).columns.values.tolist()
    
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
    age = st.number_input('Masukkan Umur : ')
    anaemia = st.number_input('Masukkan nilai Anemia : ')
    creatinine_phosphokinase = st.number_input('Masukkan nilai Enzim CPK dalam Darah : ')
    diabetes = st.number_input('Masukkan nilai Diabetes : ')
    ejection_fraction = st.number_input('Masukkan Nilai Fraksi Ejeksi : ')
    high_blood_pressure = st.number_input('Masukkan nilai hipertensi  : ')
    platelets = st.number_input('Masukkan Nilai Jumlah trombosit dalam tubuh : ')
    serum_creatinine = st.number_input('Masukkan Nilai Jumlah Creatin Serum : ')
    serum_sodium = st.number_input('Masukkan Nilai Jumlah Natrium Serum : ')
    sex = st.number_input('Masukkan Jenis Kelamin : ')
    smoking = st.number_input('Masukkan Smoking : ')
    time = st.number_input('Masukkan Nilai Periode Tindak Lanjut : ')

    def submit():
        # input
        inputs = np.array([[
            age,
            anaemia,
            creatinine_phosphokinase,
            diabetes,
            ejection_fraction,
            high_blood_pressure,
            platelets,
            serum_creatinine,
            serum_sodium,
            sex,
            smoking,
            time,
            ]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang di masukkan, maka anda prediksi cuaca : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

