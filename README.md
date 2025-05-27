# Laporan Proyek Machine Learning Predictive Analytics - Sindi Aprilianti

## Domain Project: Kesehatan
Kemajuan pesat dalam teknologi digital serta peningkatan penggunaan perangkat digital telah menyebabkan perubahan yang signifikan dalam pola perilaku harian individu. Salah satu fungsi kognitif yang sangat dipengaruhi oleh interaksi dengan teknolofi digital adalah fokus, yaitu kemampuan seseorang untuk mempertahankan perhatian dalam jangka waktu tertentu, yang sangat penting bagi produktivitas dan kesejahteraan mental. Husain et al. (2024) dalam penelitiannya menyampaikan bahwa kecanduan media sosial juga ternyata berkorelasi negatif dengan kemampuan fokus dan kesadaran penuh pada mahasiswa. Semakin tinggi kecanduan media sosial, semakin sulit bagi individu untuk mempertahankan perhatian dan keterlibatan penuh dalam aktivitas sehari-hari, akibat gangguan terus-menerus dari notifikasi dan informasi dari sosial media. Dengan mengembangkan model prediksi yang akurat, dapat diberikan intervensi yang tepat, guna membantu individu mengatur waktu pemakaian perangkat digital sekaligus menjaga keseimbangan kesehatan mental mereka. Dataset Mental Health and Digital Behaviour (2020-2024) digunakan dalam pembuatan model machine learning untuk memperbaiki focus_score berdasarkan berbagai fitur perilaku digital, seprti durasi penggunaan layar, jumlah notifikasi, frekuensi pergantian aplikais, dan lainnya. 

Daftar pustaka: Husain, M., Mushtaq, N., Mahsud, N.M., Afzal, H., Naseer, S., Hussain, D. (2024). The Effect of Social Media Addiction on Attention Span and 
Aggression among University Students. Kurdish Studies. Available at https://www.researchgate.net/profile/Muhammad-Hussain-223/publication/381403305_Effect_of_Social_Media_Addiction_on_Attention_Span_and_Aggression_among_University_Students/links/666b9e3d85a4ee7261c0ef4f/Effect-of-Social-Media-Addiction-on-Attention-Span-and-Aggression-among-University-Students.pdf  

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

- Dataset ini terdiri atas 500 baris data dan 9 kolom

  ![image](https://github.com/user-attachments/assets/15be7fb5-f6a6-4318-ad2f-8435eb6fd40e)

- Tidak terdapat missing values

  ![image](https://github.com/user-attachments/assets/0d29c266-7221-47ca-99d9-12f54ecc71b3)

- Tidak terdapat duplikat data

  ![image](https://github.com/user-attachments/assets/08ba3f59-4ec4-440c-9cde-bf69b0b24e60)

- Semua fitur, kecuali anxiety_level, memiliki outlier
  
  ![image](https://github.com/user-attachments/assets/384851be-136a-448f-b105-b78184265200)
  ![image](https://github.com/user-attachments/assets/6a5ff161-1ea2-4a53-969e-d108ac233589)
  ![image](https://github.com/user-attachments/assets/c5e0f024-5741-423b-9afe-d62828d396b6)
  ![image](https://github.com/user-attachments/assets/f51ac7f6-78fa-470c-955d-212f9ed46348)
  ![image](https://github.com/user-attachments/assets/6b841d0d-a7d1-4c1c-85b3-85766b39b8dc)
  ![image](https://github.com/user-attachments/assets/c4bc3eb9-c610-4256-b798-7a87add939d6)
  ![image](https://github.com/user-attachments/assets/60fdbc79-06d2-40e0-9b41-ee5cc72984c8)
  ![image](https://github.com/user-attachments/assets/60695ef6-64d9-4dc5-a6aa-411313c761ae)
  ![image](https://github.com/user-attachments/assets/facdc0cf-2b84-4f93-8a3d-c52afd53ebd0)


- Hampir semua fitur memiliki sebaran data distribusi normal, mayoritas datanya berada di sekitar rata-rata, kecuali pada fitur anxiety_level yang datanya menjorok ke kiri, menunjukkan bahwa mayoritas orang memiliki anxiety level yang tinggi.
  ![image](https://github.com/user-attachments/assets/193d6d82-4913-454b-9448-9c0ef5be97b2)

- 

### Variabel-variabel pada dataset Mental Health and Digital Behaviour (2020-2024) adalah sebagai berikut:
- daily_screen_time_min : merupakan total screen time harian (menit)
- num_app_switches: merupakan frekuensi berpindah aplikasi dalam sehari
- sleep_hours: merupakan durasi tidur harian (jam)
- notification_count: merupakan jumlah notifikasi yang diterima
- social_media_time_min: merupakan waktu penggunaan media sosial (menit)
- mood_Score: merupakan skor suasana hati (1-10)
- digital_wellbeing_score: merupakan skor kesejahteraan digital gabungan
- focus_score: merupaka skor fokus (1-10)
- anxiety_level: merupakan skor level anxiety (1-10)

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

Hasil menunjukkan bahwa model Linear Regression lebih sesuai digunakan untuk kasus ini karena mampu memberikan prediksi yang lebih akurat dan konsisten dengan data aktual. Analisis korelasi mengindikasikan bahwa fitur notification_count dengan nilai korelasi (-0.34), daily_Screen_time(-0.28), dan num_app_switches(-0.25) memiliki pengaruh negatif terhadap focus_score. Artinya, semakin tinggi nilai ketiga fitur tersebut, maka skor fokus cenderung menurun. Sebaliknya, fitur digital_wellbeing_score menunjukkan korelasi positif sebesar 0.41 dengan focus_score, yang berarti peningkatan skor kesejahteraan digital berkaitan dengan peningkatan skor fokus. 
