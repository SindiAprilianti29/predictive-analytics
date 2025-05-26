# Laporan Proyek Machine Learning Predictive Analytics - Sindi Aprilianti

## Domain Project: Kesehatan
Dalam kehidupan modern yang sangat bergantung dengan teknologi digital saat ini, kemampuan individu untuk tetap fokus menjadi masalah tersendiri. Penggunaan sosial media yang terus menerus, notifikasi yang tak berhenti, dan juga kecenderungan untuk sering pindah aplikasi berpotensi memecah perhatian dan mengganggu konsentrasi. Fenomena ini dikenal sebagai digital distraction. 

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1: Dari serangkaian fitur yang tersedia, fitur manakan yang paling berpengaruh terhadap skor fokus seseorang?
- Pernyataan Masalah 2: Apakah mungkin memprediksi tingkat fokus seseorang berdasarkan data perilaku digital?
- Pernyataan Masalah 3: Sejauh maan akurasi model machine learning dalam memperkirakan skor fokus menggunakan dataset tersebut?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1: Mengetahui fitur-fitur yang paling berkorelasi dengan penurunan atau peningkatan focus_score.
- Jawaban pernyataan masalah 2: Membuat model machine learning yang dapat memprediksi focus_score secara akurat berdasarkan variabel dalam dataset
- Jawaban pernyataan masalah 3: Membandingkan beberapa algoritma regresi dan memilih model dengan error terkecil berdasarkan metrik evaluasi

### Solution Statements
- Menerapkan dua algoritma regresi
  Menggunakan algoritma Linear Regression dan Random Forest Regressor
- Melakukan data preparation
  Melakukan normalisasi fitur dengan StandardScaler, train test split dengan proporsi 80:20, dan outlier handling dengan metode IQR.
- Melakukan evaluasi dan seleksi model
  Menggunakan MSE untuk mengukur sejauh mana nilai prediksi berbeda dari nilai aktual focus_score dan memilih model yang memiliki nilai MSE paling rendah

## Data Understanding
Dataset yang digunakan dalam project ini berasal dari Kaggle, yang berjudul Mental Health and Digital Behaviour (2020-2024), yang dapat diunduh pada tautan berikut: 
(https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-digital-behavior-20202024).

### Variabel-variabel pada dataset Mental Health and Digital Behaviour (2020-2024) adalah sebagai berikut:
- daily_screen_time_min : merupakan total screen time harian (menit)
- num_app_switches: merupakan frekuensi berpindah aplikasi dalam sehari
- sleep_hours: merupakan durasi tidur harian (jam)
- notification_count: merupakan jumlah notifikasi yang diterima
- social_media_time_min: merupakan waktu penggunaan media sosial (menit)
- mood_Score: merupakan skor suasana hati (1-10)
- digital_wellbeing_score: merupakan skor kesejahteraan digital gabungan
- focus_score: merupaka skor fokus (1-10)

Beberapa tahapan EDA yang dilakukan pada dataset:
- Melihat jumlah entri data dan fitur menggunakan df.shape
- Melihat contoh data dengan df.head()
- Melihat statistik deskriptif dari data dengan df.describe()
- Melihat missing values dengan df.isnull().sum()
- Melihat data duplikat dengan df.duplicated.sum()
- Menampilkan boxplot untuk melihat outlier
- Hapus outlier dengan metode IQR
- Menampilkan histogram, pairplot, dan correlation_matrix untuk memahami karakteristik dan hubungan yang ada dalam dataset. 

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.
Pada tahap ini dilakukan dua langkah data preparation
- Train test split
  Data dibagi menjadi dua bagian, yaitu data train dan data test. Tujuannya untuk melatih model pada data train dan menguji performa model pada data test yang belum pernah dilihat oleh model sebelumnya, sehingga dapat evaluasi kemampuan generalisasi model. Pada project ini, pembagiannya dilakukan dengan rasio sebesar 80:20, dengan data train sebesar 80% dan data test 20%, ini ideal untuk dataset yang memiliki entri data yang sedikit, karena dataset yang digunakan memiliki entri data kurang dari 1000.
- Normalisasi menggunakan StandardScaler()
  Karena fitur memiliki skala yang berbeda-beda, maka dilakukan normalisasi agar semua fitur memiliki rata-rata 0 dan standar deviasi 1. Ini penting agar model dapat bekerja dengan lebih optimal dan stabil. normalisasi dilakukan dengan StandardScaler.
  
## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
Model yang digunakan dalam menyelesaikan masalah ini adalah Random Forest dan Linear Regression. 
- Random Forest
  Random Forest Regressor merupakan sebuah algoritma ensemble yang basisnya adalah decision tree, bekerja dengan membangun beberapa trees (pohon keputusan) dan menggabungkan hasil prediksi dari semua pohon tersebut, dengan mengambil nilai rata-ratanya. Model ini tidak terlalu sensitif terhadap outlier dan missing values, hanya saja modelnya menghabiskan waktu training dan prediksi yang relatif lebih lama, terutama jika estimator besar dan datasetnya juga besar. 
Parameter yang digunakan adalah n_estimator=50, ini merupakan jumlah pohon keputusan yang digunakan dalam forest. Semakin banyak jumlahnya, maka akan semakin stabil hasil prediksi, namun waktu komputasinya juga akan meningkat. max_dept=16 merupakan maksimum dari tiap pohon, ini mencegah pohon menjadi terlalu dalam dan menyebabkan overfitting. random_state=55 untuk memastikan hasil yang konsisten saat dijalankan ulang. n_jobs=1 akan mengaktifkan pemrosesan paralel untuk memanfaatkan seluruh core CPU saat training model.

  Tahapan yang dilakukan setelah proses data preparation (melakukan normalisasi dengan standardScaler dan train test split), yaitu:
  1. melakukan inisialisasi model, RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
  2. training model, RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
  3. prediksi dan ealuasi model, RF.predict(X_train)

- Linear Regression
  Merupakan model untuk memetakan hubungan linear antara satu atau lebih fitur (independen) dengan target output (dependen). Ini cocok digunakan saat hubungan antar variabel cenderung linear dan interpretasinya sederhana, hanya saja sensitif terhadap outlier dan kurang fleksibel untuk hubungan non-linear. Parameter pada linear regression biasanya menggunakan default sehingga tidak ada yang di set. 

  Tahapan yang dilakukan adalah
  1. inisialisasi model,  LinearRegression()
  2. Latih model dengan data training, fit(X_train, y_train)
  3. Hitung error MSE di data training dan simpan hasilnya mean_squared_error

     Model terbaik untuk permasalahan ini dapat diketahui setelah melakukan evaluasi menggunakan metrik MSE pada data training dan testing. Dari hasil yang didapat, model yang cocok untuk masalah ini adalah Linear Regression karena MSE yang dihasilkan lebih kecil. Linear regression juga lebih sederhana dan mudah diinterpretasi.

## Evaluation
Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE), yaitu salah satu metrik untuk regresi. MSE mengukur rata-rata dari kuadrat selisih antara nilai actual dengan nilai yang diprediksi sebagai model. Berikut merupakan formula dari MSE:

![image](https://github.com/user-attachments/assets/4b20f12e-515e-49a1-a811-28cf6d862fc7)

Hasil evaluasi kedua model menggunakan metrik MSE adalah sebagai berikut

![image](https://github.com/user-attachments/assets/396b7772-1ce0-4f09-b1aa-156e8771fe01)

Pada gambar tersebut, terlihat bahwa nilai MSE dari model Linear Regression (LR) lebih rendah dibandingkan dengan model Random Forest (RF). Nilai MSE yang lebih rendah menunjukkan bahwa model LR memiliki tingkat kesalahan prediksi yang lebih kecil. 

Berikutnya, ketika dilakukan prediksi terhadap data uji, hasil prediksi dari model linear regression lebih sesuai dengan nilai aktual dibandingkan hasil prediksi dengan random forest. 

![image](https://github.com/user-attachments/assets/430cacf0-02b3-4be6-9df8-3318fb4309c3)

Hal ini menunjukkan bahwa model Linear Regression lebih sesuai digunakan untuk kasus ini karena memberikan prediksi yang lebih akurat dan konsisten dengan data sebenarnya.
