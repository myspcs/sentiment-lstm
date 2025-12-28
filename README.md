# Türkçe Film Yorumları Duygu Analizi (LSTM)

Bu projede Türkçe film yorumları üzerinde derin öğrenme tabanlı bir **duygu analizi** sistemi geliştirilmiştir. Model, verilen bir yorumun **pozitif** veya **negatif** olduğunu tahmin etmektedir.

---

## 1. Proje Konusu ve Seçilme Gerekçesi

Duygu analizi, kullanıcı yorumlarının otomatik olarak analiz edilmesini sağlayan önemli bir doğal dil işleme (NLP) problemidir.  
Film platformları ve sosyal medya uygulamalarında kullanıcı yorumları karar verme süreçlerini doğrudan etkilemektedir.

Bu projede Türkçe dilinde duygu analizi yapılması tercih edilmiştir çünkü Türkçe NLP çalışmaları, İngilizceye kıyasla daha sınırlıdır.

---

## 2. Veri Seti

Projede **mteb/turkish_movie_sentiment** veri seti kullanılmıştır.

- Türkçe film yorumlarından oluşur  
- Etiketler: `pozitif (1)` ve `negatif (0)`  
- Gerçek kullanıcı yorumları içerir  

Veri seti linki:  
https://huggingface.co/datasets/mteb/turkish_movie_sentiment

---

## 3. Yöntem / Algoritma Seçimi

Bu projede **LSTM (Long Short-Term Memory)** tabanlı bir sinir ağı kullanılmıştır.

### Neden LSTM?
- Metin verileri sıralı yapıdadır  
- LSTM, uzun dönemli bağımlılıkları öğrenebilir  

### Karşılaştırmalı Analiz:
- **Bag of Words / TF-IDF:** Bağlam bilgisi yok  
- **CNN:** Yerel örüntülerde başarılı  
- **Transformer:** Güçlü fakat karmaşık ve maliyetli  
- **LSTM:** Performans ve sadelik dengesi sağlar  

Bu nedenle LSTM tercih edilmiştir.

---

## 4. Model Eğitimi ve Değerlendirilmesi

- Veri seti **train / validation / test** olarak ayrılmıştır  
- Kayıp fonksiyonu: Binary Cross Entropy  
- Optimizasyon: Adam  

### Kullanılan Metrikler:
- Accuracy  
- Precision  
- Recall  
- F1-score  

Model, eğitim sürecinde **epoch bazlı validation sonuçları** ile izlenmiş,  
eğitim tamamlandıktan sonra **bağımsız test seti** üzerinde nihai performans değerlendirmesi yapılmıştır.

Ayrıca test seti için **Confusion Matrix** görsel olarak gösterilmiştir.

---

## 5. Proje Yapısı

```text
sentiment_lstm/
├── train.py
├── model.py
├── serve.py
├── sentiment_lstm.pth
├── vocab.pkl
├── requirements.txt
├── results/
│   └── confusion_matrix_test.png
└── README.md

```

---

## 6. Çalıştırma

### Model Eğitimi
```bash
python train.py
```

### Web Arayüzü (Gradio)
```bash
python serve.py
```
---

## 7. Kullanılan Teknolojiler
- Python
- PyTorch
- HuggingFace Datasets
- Scikit-learn
- Gradio



