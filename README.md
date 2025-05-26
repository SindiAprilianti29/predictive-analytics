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


Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1: 
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
Model yang digunakan dalam menyelesaikan masalah ini adalah Random Forest dan Linear Regression. 
- Random Forest
  Random Forest Regressor merupakan sebuah algoritma ensemble yang basisnya adalah decision tree, bekerja dengan membangun beberapa trees (pohon keputusan) dan menggabungkan hasil prediksi dari semua pohon tersebut, dengan mengambil nilai rata-ratanya.
  Parameter yang digunakan adalah n_estimator=50, ini merupakan jumlah pohon keputusan yang digunakan dalam forest. Semakin banyak jumlahnya, maka akan semakin stabil hasil prediksi, namun waktu komputasinya juga akan meningkat. max_dept=16 merupakan maksimum dari tiap pohon, ini mencegah pohon menjadi terlalu dalam dan menyebabkan overfitting. random_state=55 untuk memastikan hasil yang konsisten saat dijalankan ulang. n_jobs=1 akan mengaktifkan pemrosesan paralel untuk memanfaatkan seluruh core CPU saat training model.

  Tahapan yang dilakukan setelah proses data preparation (melakukan normalisasi dengan standardScaler dan train test split), yaitu:
1. melakukan inisialisasi model, RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
2. training model, RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
3. prediksi dan ealuasi model, RF.predict(X_train)
- Linear Regression

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Metrik evaluasi yang digunakan adalah Mean Squared Error (MSE), yaitu salah satu metrik untuk regresi. MSE mengukur rata-rata dari kuadrat selisih antara nilai actual dengan nilai yang diprediksi sebagai model, dengan hasil sebagai berikut

![image](https://github.com/user-attachments/assets/396b7772-1ce0-4f09-b1aa-156e8771fe01)

Pada gambar, terlihat bahwa LR (Linear Regression) memiliki MSE yang lebih rendah daripada RF (Random Forest). Hal ini menunjukkan bahwa model Linear Regression lebih sesuai digunakan untuk kasus ini. 

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.
