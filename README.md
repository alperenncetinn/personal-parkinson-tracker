# 🧠 Parkinson AI Telemonitoring System

Bu proje, Parkinson hastalarının ses kayıtlarını analiz ederek **UPDRS (Unified Parkinson's Disease Rating Scale)** skorunu tahmin eden yapay zeka destekli bir telemonitoring sistemidir.

##  Özellikler

- **Çift Arayüz:** Doktorlar ve hastalar için özelleştirilmiş paneller.
- **AI Model:** XGBoost tabanlı regresyon modeli (UCI Parkinson Dataset ile eğitilmiş).
- **Kişisel Kalibrasyon (Personal Bias Layer):** Her hasta için doktorun belirlediği klinik baseline değerine göre modeli kalibre eder.
- **Anomali Tespiti:** Beklenmedik iyileşme veya hızlı kötüleşme durumlarını tespit edip uyarır.
- **Ses Analizi:** Praat (Parselmouth) kütüphanesi ile Jitter, Shimmer, HNR gibi biyobelirteçleri çıkarır.
- **Güvenli Kayıt:** Hasta ses kayıtları yerel olarak arşivlenir.

## 🛠️ Kurulum

### 1. Gereksinimler
- Python 3.8+
- Sanal ortam (önerilir)

### 2. Bağımlılıkları Yükleyin

```bash
# Sanal ortam oluştur (Opsiyonel)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Paketleri yükle
pip install streamlit pandas numpy xgboost praat-parselmouth scikit-learn audiorecorder
```

### 3. Uygulamayı Başlatın

```bash
streamlit run app.py
```

##  Giriş Bilgileri (Demo)

Sistemi test etmek için aşağıdaki demo hesaplarını kullanabilirsiniz:

| Rol | Kullanıcı Adı | Şifre |
|---|---|---|
| **Doktor** | `doktor` | `123` |
| **Hasta** | `ali` | `123` |

Doktor panelinden yeni hastalar oluşturabilirsiniz.

## Proje Yapısı

```
parkinson-final/
├── app.py                  # Ana uygulama dosyası (Streamlit)
├── users_db.csv            # Kullanıcı veritabanı
├── patient_logs.csv        # Tahmin geçmişi ve loglar
├── new_patients_data.csv   # Yeni eklenen hastaların klinik verileri (Eğitim için)
├── scaler.pkl              # Normalizasyon ölçekleyicisi
├── feature_cols.json       # Eğitilen modelin öznitelik listesi
├── trained_model.json      # Eğitilmiş XGBoost modeli
└── patient_recordings/     # Arşivlenen ses dosyaları (Git'e dahil edilmez)
```

## Nasıl Çalışır?

1. **Eğitim:** Sistem, UCI Parkinson veri seti ve doktorun girdiği yeni hasta verileriyle eğitilir (`train_model`).
2. **Kişiselleştirme:** Model **mutlak UPDRS** tahmini yapar. Ancak her hastanın ses karakteristiği farklı olduğu için, doktorun ilk ölçümüne göre bir **Bias (Sapma)** hesaplanır.
3. **Tahmin:** Hasta evden ses gönderdiğinde:
   `Final Skor = Global Model Tahmini + Kişisel Bias`
   formülüyle sonuç üretilir.

##  Notlar

- Ses analizi için **Praat** yazılımının Python sarmalayıcısı olan `parselmouth` kullanılır.
- Ses kayıtları `.gitignore` dosyası ile repodan hariç tutulmuştur (KVKK/Gizlilik).

---
**Geliştirici:** [Adınız/Ekibiniz]
