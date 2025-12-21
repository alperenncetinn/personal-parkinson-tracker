# Parkinson AI Telemonitoring System

Bu proje, Parkinson hastalarının hastalık seyrini (UPDRS skorunu) ses analizi ve yapay zeka kullanarak uzaktan takip etmeyi sağlayan bir tele-tıp uygulamasıdır. Sistem, doktorların hastalarını yönetebileceği, verileri etiketleyebileceği ve hastaların evlerinden ses kaydı göndererek durumlarını izleyebileceği entegre bir platform sunar.

## Temel Ozellikler

### 1. Yapay Zeka Tabanli Ses Analizi
Sistem, kullanicidan alinan ham ses verisini (WAV) isleyerek 20'den fazla akustik oznitelik cikarir. Bu oznitelikler arasinda Jitter, Shimmer, HNR, RPDE, DFA ve PPE gibi Parkinson hastaligi ile iliskili vokal biyobelirtecler bulunur.
XGBoost regresyon modeli, bu ozellikleri kullanarak hastanin UPDRS (Unified Parkinson's Disease Rating Scale) skorunu tahmin eder.

### 2. Kisisel Kalibrasyon ve Bias Katmani
Her insanin ses yapisi farklidir. Sistem, genel bir model kullanmak yerine her hasta icin "Kisisel Bias" (Sapma) hesaplar.
- **Ilk Kalibrasyon:** Doktor, klinikte hastanin gercek UPDRS skorunu girer ve ses kaydi alir. Modelin tahmini ile gercek skor arasindaki fark (Bias) kaydedilir.
- **Evden Takip:** Hasta evden ses gonderdiginde, modelin ham tahmini bu bias degeri ile duzeltilir. Bu sayede model hatasi minimize edilir ve hastanin kisisel degisimi (Delta) dogru bir sekilde izlenir.

### 3. Surekli Ogrenme (Continuous Learning)
Doktorlar, mevcut bir hasta icin farkli zamanlarda (ornegin ilac aldiktan sonra veya kontrol muayenelerinde) yeni veri girisleri yapabilir. Bu veriler egitim setine eklenir ve model yeniden egitildiginde, sistem o hastanin ses karakteristiklerini ve hastaligin farkli evrelerini ogrenir. Bu yontemle model zamanla kisisellesir.

### 4. Coklu Kullanici ve Rol Yonetimi (Multi-Tenancy)
- **Doktor Paneli:** Doktorlar sadece kendi ekledikleri hastalari gorur ve yonetir. Hasta ekleme, veri girisi, model egitimi ve hasta takibi bu panelden yapilir.
- **Hasta Paneli:** Hastalar kendi kullanici adi ve sifreleri ile giris yaparak ses kaydi gonderebilir ve gecmis olcumlerini grafiksel olarak izleyebilir.

## Kurulum ve Calistirma

### Gereksinimler
- Python 3.9 veya uzeri
- Sanal ortam (Virtualenv) onerilir.

### Kurulum Adimlari

1. Proje dizinine gidin ve sanal ortami olusturun:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   # venv\Scripts\activate   # Windows
   ```

2. Gerekli kutuphaneleri yukleyin:
   ```bash
   pip install -r requirements.txt
   ```
   *Not: Ses analizi icin `parselmouth` (Praat) ve `audiorecorder` kutuphaneleri gereklidir.*

3. Uygulamayi baslatin:
   ```bash
   streamlit run app.py
   ```

## Kullanim Senaryosu (Klinik Protokol)

Sistemin en yuksek basarimla calismasi icin asagidaki protokol onerilir:

1. **Ilk Muayene (Baseline):** Doktor, hastayi sisteme ekler. Klinik UPDRS skorunu girer ve ilk ses kaydini alir. "Sistemi Yeniden Egit" butonuna basilarak model hastayi tanir.
2. **Ilac Etkisi (Opsiyonel):** Doktor, "Mevcut Hastaya Veri Ekle" secenegi ile hastanin ilac aldiktan sonraki (ON donemi) sesini ve skorunu sisteme girer. Model hastanin hem ilacli hem ilacsiz durumunu ogrenir.
3. **Evden Takip:** Hasta, kendisine verilen sifre ile sisteme girer. Belirli araliklarla (ornegin haftada bir) ses kaydi gonderir. Sistem, kisisellestirilmis tahmini ve baseline'a gore degisimi (Delta) raporlar.
4. **Kontrol:** Doktor panelinden hastanin zaman icindeki degisim grafigi incelenir.

## Teknik Yapi

- **Backend/Frontend:** Streamlit
- **Veri Isleme:** Pandas, NumPy
- **Makine Ogrenmesi:** XGBoost (Regressor), Scikit-learn
- **Ses Isleme:** Praat (Parselmouth), Pydub
- **Gorsellestirme:** Altair

## Dosya Yapisi

- `app.py`: Ana uygulama kodu.
- `users_db.csv`: Kullanici veritabani (Sifreli saklama onerilir).
- `patient_logs.csv`: Hasta olcum gecmisi.
- `new_patients_data.csv`: Modelin yeniden egitilmesi icin biriktirilen klinik veriler.
- `patient_recordings/`: Kaydedilen ses dosyalari.
- `feature_cols.json` & `scaler.pkl`: Modelin tutarliligi icin gerekli meta veriler.

## Veri Guvenligi Notu

Ses dosyalari ve hasta verileri hassas icerik tasir. `.gitignore` dosyasi, bu verilerin versiyon kontrol sistemine (Git) gonderilmesini engellemek icin yapilandirilmistir. `patient_recordings/`, `*.csv` ve `__pycache__` dosyalari gonderilmez.
