import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Firdatul Fitriyah ")
st.write("##### Nim   : 200411100020 ")
st.write("##### Kelas : Penambangan Data C ")

description, upload_data, preprocessing, modeling, implementation = st.tabs(["Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Kidney Stone Prediction based on Urine Analysis (Prediksi Batu Ginjal Berdasarkan Analisis Urin) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis")
    st.write(" Suatu endapan kecil dan keras yang terbentuk di ginjal dan sering menyakitkan saat buang air kecil. Batu ginjal adalah endapan keras yang terbuat dari mineral dan garam asam yang mengendap dalam urin yang terkonsentrasi. Batu ginjal ini dapat menyakitkan saat melewati saluran kemih, tetapi biasanya tidak menyebabkan kerusakan permanen. Gejala yang paling umum berupa nyeri parah, biasanya di sisi perut, yang sering disertai dengan mual. Perawatan meliputi pereda nyeri dan banyak minum air putih untuk membantu meluruhkan batu. Prosedur medis mungkin diperlukan untuk mengambil atau menghancurkan batu yang besar. ")
    st.write(" Dataset ini dapat digunakan untuk memprediksi adanya batu ginjal berdasarkan analisis urin. ")
    st.write(" 79 spesimen urin, dianalisis dalam upaya untuk menentukan apakah karakteristik fisik urin tertentu mungkin terkait dengan pembentukan kristal kalsium oksalat. ")
    st.write("""# Deskripsi Data""")
    st.write("Total datanya adalah 79 data")
    st.write("Informasi Atribut")
    st.write("1) gravity : berat jenis, densitas urin relatif terhadap air ")
    st.write("2) ph : logaritma negatif dari ion hidrogen ")
    st.write("3) osmo : osmolaritas (mOsm), satuan yang digunakan dalam biologi dan kedokteran tetapi tidak dalam kimia fisik. Osmolaritas sebanding dengan konsentrasi molekul dalam larutan ")
    st.write("4) cond : konduktivitas (mMho miliMho). Satu Mho adalah satu timbal balik Ohm. Konduktivitas sebanding dengan konsentrasi muatan ion dalam larutan ")
    st.write("5) urea : konsentrasi urea dalam milimol per liter ")
    st.write("6) calc : kalsium konsentrasi (CALC) dalam milimol-liter ")
    st.write("7) target : penentuan termasuk batu ginjal atau tidak ")
    st.write ("""0 = Tidak Adanya Batu Ginjal""")
    st.write ("""1 = Adanya Batu Ginjal""")

    st.write("###### Aplikasi ini untuk : Kidney Stone Prediction based on Urine Analysis (Prediksi Batu Ginjal Berdasarkan Analisis Urin) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link :https://github.com/FirdatulFitriyah/pendatweb  ")

with upload_data:
    st.write("""# Dataset Asli """)
    df = pd.read_csv('https://raw.githubusercontent.com/FirdatulFitriyah/pendatweb/main/kindey%20stone%20urine%20analysis.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
   
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['target'])
    y = df['target'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.target).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]]
    })

    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Precipitation = st.number_input('Masukkan preciptation (curah hujan) : ')
        Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
        Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
        Wind = st.number_input('Masukkan wind (angin) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Precipitation,
                Temp_Max,
                Temp_Min,
                Wind
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
