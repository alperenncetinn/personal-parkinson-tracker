import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import parselmouth
from parselmouth.praat import call
import os
from datetime import datetime
import shutil
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score

# Ses kayıt bileşeni
try:
    from audiorecorder import audiorecorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# --------------------------------------------------------
# 1. AYARLAR VE DOSYA YÖNETİMİ
# --------------------------------------------------------
st.set_page_config(page_title="Parkinson AI System", page_icon="🧠", layout="wide")

# Dosya İsimleri
USERS_FILE = "users_db.csv"        # Kullanıcılar
HISTORY_FILE = "patient_logs.csv"   # Tahmin Geçmişi (Evden Gönderilen Sesler)
ORIGINAL_DATA = "parkinsons_updrs.data" # Orijinal Eğitim Verisi
NEW_DATA_FILE = "new_patients_data.csv" # Yeni Eklenen Hastaların "Label"lı Verisi (Kalibrasyon)
MODEL_FILE = "live_model.json" # Eğitilen modelin kaydedildiği yer

# Başlangıç Fonksiyonları
def init_system():
    # 1. Kullanıcı Tablosu
    if not os.path.exists(USERS_FILE):
        data = {
            'Username': ['doktor', 'ali', 'ayse'],
            'Password': ['123', '123', '123'],
            'Role': ['Doctor', 'Patient', 'Patient'],
            'Name': ['Dr. Zekeriya', 'Ali Yılmaz', 'Ayşe Demir'],
            'ID': [999, 1, 2], 
            'Doctor_ID': [0, 999, 999],
            'Age': [0, 65, 72],
            'Sex': [0, 0, 1]
        }
        pd.DataFrame(data).to_csv(USERS_FILE, index=False)
    
    # 2. Yeni Veri Deposu (Cold Start Çözümü)
    if not os.path.exists(NEW_DATA_FILE):
        # Define full header including UPDRS_baseline
        CSV_HEADER = "subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,UPDRS_baseline,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE"
        with open(NEW_DATA_FILE, 'w') as f:
            f.write(CSV_HEADER + '\n')
        st.success('Yeni veri dosyası oluşturuldu ve başlık eklendi.')

    # 3. Admin Kontrolü (Sonradan eklenen özellik)
    if os.path.exists(USERS_FILE):
        try:
            users_df = pd.read_csv(USERS_FILE)
            if 'Role' in users_df.columns and not users_df['Username'].isin(['admin']).any():
                # Admin yoksa ekle
                max_id = users_df['ID'].max() if not users_df.empty else 0
                admin_user = pd.DataFrame([{
                    'Username': 'admin',
                    'Password': '123', # Kolay test için 123
                    'Role': 'Admin',
                    'Name': 'System Admin',
                    'ID': max_id + 99,
                    'Doctor_ID': 0,
                    'Age': 0,
                    'Sex': 0
                }])
                combined = pd.concat([users_df, admin_user], ignore_index=True)
                combined.to_csv(USERS_FILE, index=False)
        except Exception:
            pass

init_system()

# --------------------------------------------------------
# 2. YAPAY ZENA MOTORU (EĞİTİM VE TAHMİN)
# --------------------------------------------------------

def augment_audio_features(df, augment_factor=3):
    """
    Ses özelliklerine veri artırma (Data Augmentation) uygular.
    Gerçek ses dosyasını değiştirmek yerine, özellik değerlerine
    kontrollü gürültü ekleyerek sentetik veri üretir.
    
    Bu yaklaşım:
    - Overfitting'i azaltır
    - Modelin genelleme kapasitesini artırır
    - Az veriyle çalışırken daha robust sonuçlar verir
    """
    if len(df) == 0:
        return df
    
    augmented_rows = []
    
    # Ses özellikleri (augmentation uygulanacak sütunlar)
    audio_feature_cols = [
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    
    for idx, row in df.iterrows():
        # Orijinal satırı koru
        augmented_rows.append(row)
        
        # Augmented versiyonlar oluştur
        for i in range(augment_factor - 1):
            new_row = row.copy()
            
            for col in audio_feature_cols:
                if col in new_row.index and pd.notna(new_row[col]):
                    original_val = float(new_row[col])
                    # %5-15 arası rastgele gürültü ekle
                    noise_factor = np.random.uniform(0.95, 1.05)
                    # Ek olarak küçük bir Gaussian gürültü
                    gaussian_noise = np.random.normal(0, abs(original_val) * 0.02)
                    new_row[col] = original_val * noise_factor + gaussian_noise
            
            # test_time'ı da hafifçe değiştir (farklı zaman simülasyonu)
            if 'test_time' in new_row.index:
                new_row['test_time'] = float(new_row['test_time']) + np.random.uniform(-0.5, 0.5)
            
            augmented_rows.append(new_row)
    
    return pd.DataFrame(augmented_rows)


def train_model():
    """
    Geliştirilmiş Eğitim Fonksiyonu (Overfitting Önleme):
    
    1. Data Augmentation: Yeni hasta verilerine kontrollü gürültü ekleme
    2. GroupShuffleSplit: Hasta bazlı train/test ayrımı (data leakage önleme)
    3. Güçlü Regularization: reg_alpha ve reg_lambda artırıldı
    4. Minimum Veri Kontrolü: Yetersiz veriyle eğitim uyarısı
    5. Early Stopping: Overfitting tespiti için
    """
    status = st.empty()
    progress = st.progress(0)
    status.info("🧠 Yapay Zeka Motoru: Veriler hazırlanıyor...")
    
    # ========== 1. VERİ YÜKLEME ==========
    df_orig = pd.read_csv(ORIGINAL_DATA)
    
    new_patient_count = 0
    new_record_count = 0
    
    if os.path.exists(NEW_DATA_FILE):
        df_new = pd.read_csv(NEW_DATA_FILE)
        new_record_count = len(df_new)
        new_patient_count = df_new['subject#'].nunique() if not df_new.empty else 0
        
        # ========== MINIMUM VERİ KONTROLÜ ==========
        MIN_RECORDS_FOR_TRAINING = 10
        if new_record_count > 0 and new_record_count < MIN_RECORDS_FOR_TRAINING:
            st.warning(f"""
            ⚠️ **Yetersiz Veri Uyarısı**
            
            Şu an yeni hasta verisi sayısı: **{new_record_count}** kayıt
            Önerilen minimum: **{MIN_RECORDS_FOR_TRAINING}** kayıt
            
            Az veriyle eğitim **overfitting** riskini artırır!
            Devam etmek istiyorsanız 'Yine de Eğit' seçeneğini kullanın.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if not st.checkbox("⚠️ Riski kabul ediyorum, yine de eğit"):
                    st.info("💡 Daha fazla hasta verisi ekledikten sonra tekrar deneyin.")
                    return None, None
        
        # ========== 2. VERİ ARTIRMA (DATA AUGMENTATION) ==========
        status.info("🔄 Veri artırma (augmentation) uygulanıyor...")
        progress.progress(20)
        
        # Sadece YENİ hasta verilerine augmentation uygula
        # Orijinal UCI verisi zaten yeterince büyük
        if not df_new.empty:
            augment_factor = max(3, 15 // max(1, new_record_count))  # Az veri = daha fazla augmentation
            augment_factor = min(augment_factor, 5)  # Maksimum 5x
            
            df_new_augmented = augment_audio_features(df_new, augment_factor=augment_factor)
            st.info(f"📈 Veri Artırma: {new_record_count} → {len(df_new_augmented)} kayıt ({augment_factor}x)")
            
            full_data = pd.concat([df_orig, df_new_augmented], ignore_index=True)
        else:
            full_data = df_orig
    else:
        full_data = df_orig
    
    progress.progress(30)
    
    # ========== 3. FEATURE HAZIRLIĞI ==========
    status.info("🔧 Özellikler hazırlanıyor...")
    
    # subject# ÇIKARILACAK (feature leakage engellenir)
    X = full_data.drop(['subject#', 'total_UPDRS', 'motor_UPDRS'], axis=1, errors='ignore')
    
    # Baseline/delta sütunları varsa çıkar
    cols_to_drop = [c for c in ['UPDRS_baseline', 'delta_UPDRS'] if c in X.columns]
    if cols_to_drop:
        X = X.drop(cols_to_drop, axis=1)
    
    # Hedef: MUTLAK total_UPDRS
    y = full_data['total_UPDRS']
    
    # Patient-wise split için group bilgisi
    groups = full_data['subject#']
    
    # NaN temizliği
    non_na_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[non_na_mask].reset_index(drop=True)
    y = y.loc[non_na_mask].reset_index(drop=True)
    groups = groups.loc[non_na_mask].reset_index(drop=True)
    
    # Sayısal tiplere dönüştür
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    progress.progress(40)
    
    # ========== 4. HASTA BAZLI CROSS-VALIDATION (GroupShuffleSplit) ==========
    status.info("🔀 Hasta bazlı train/test ayrımı yapılıyor (GroupShuffleSplit)...")
    
    # Benzersiz hasta sayısını kontrol et
    unique_patients = groups.nunique()
    
    if unique_patients >= 5:
        # GroupShuffleSplit: Aynı hasta ASLA hem train hem test'te olmaz
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        train_patients = groups.iloc[train_idx].nunique()
        test_patients = groups.iloc[test_idx].nunique()
        
        st.success(f"✅ Hasta Bazlı Ayrım: {train_patients} hasta (train) | {test_patients} hasta (test)")
    else:
        # Çok az hasta varsa standart split kullan ama uyar
        st.warning(f"⚠️ Sadece {unique_patients} benzersiz hasta var. Standart split kullanılıyor.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write(f"📊 Train Ort. UPDRS: {y_train.mean():.2f} | Test Ort. UPDRS: {y_test.mean():.2f}")
    
    progress.progress(50)
    
    # ========== 5. SCALING ==========
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    progress.progress(60)
    
    # ========== 6. MODEL EĞİTİMİ (Güçlü Regularization) ==========
    status.info("🤖 Model eğitiliyor (güçlü regularization aktif)...")
    
    model = xgb.XGBRegressor(
        n_estimators=500,           # Azaltıldı (overfitting önleme)
        learning_rate=0.03,         # Düşürüldü (daha yavaş öğrenme)
        max_depth=4,                # AZALTILDI (karmaşıklık sınırı)
        min_child_weight=5,         # ARTIRILDI (yaprak başına min örnek)
        gamma=0.2,                  # ARTIRILDI (bölünme için min kazanç)
        subsample=0.7,              # Azaltıldı (her ağaç için veri örneklemi)
        colsample_bytree=0.7,       # Azaltıldı (her ağaç için feature örneklemi)
        
        # ===== GÜÇLENDİRİLMİŞ REGULARIZATION =====
        reg_alpha=1.0,              # L1 regularization (ARTIRILDI: 0 → 1.0)
        reg_lambda=5.0,             # L2 regularization (ARTIRILDI: 1 → 5.0)
        
        n_jobs=-1,
        random_state=42,
        
        # Early stopping için
        early_stopping_rounds=50
    )
    
    # Early stopping ile eğit
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    progress.progress(80)
    
    # ========== 7. KAYDETME ==========
    status.info("💾 Model kaydediliyor...")
    
    model.get_booster().save_model(MODEL_FILE)
    
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    import json
    with open('feature_cols.json', 'w') as f:
        json.dump(list(X.columns), f)
    
    progress.progress(90)
    
    # ========== 8. DEĞERLENDİRME ==========
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    # Overfitting kontrolü
    overfit_gap = train_score - test_score
    
    progress.progress(100)
    
    # Sonuç raporu
    st.divider()
    st.subheader("📊 Eğitim Raporu")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Train R²", f"{train_score:.3f}")
    col2.metric("Test R²", f"{test_score:.3f}")
    col3.metric("Overfit Gap", f"{overfit_gap:.3f}", 
                delta_color="inverse" if overfit_gap > 0.1 else "normal")
    
    if overfit_gap > 0.15:
        st.error("⚠️ YÜKSEK OVERFIT TESPİT EDİLDİ! Daha fazla veri toplamanız önerilir.")
    elif overfit_gap > 0.10:
        st.warning("⚠️ Hafif overfit belirtisi. Dikkatli olun.")
    else:
        st.success("✅ Model dengeli görünüyor (düşük overfit riski).")
    
    st.info(f"""
    **Eğitim Özeti:**
    - Feature sayısı: {len(X.columns)}
    - Train: {len(X_train)} kayıt | Test: {len(X_test)} kayıt
    - Benzersiz hasta: {unique_patients}
    - Yeni hasta verisi: {new_record_count} (augmented)
    - Early stopping round: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}
    """)
    
    status.success("✅ Eğitim Tamamlandı!")
    
    return scaler, list(X.columns)

def get_active_model():
    """
    Eğitilmiş modeli yükler:
    - Model'i live_model.json'dan
    - Scaler'ı scaler.pkl'den
    - Feature columns'u feature_cols.json'dan
    - Baseline lookup'ı NEW_DATA_FILE'dan (sadece explicit clinical baselines)
    """
    if not os.path.exists(MODEL_FILE):
        return None, None, None, None, None
        
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    # Scaler'ı pickle'dan yükle
    import pickle
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.error("⚠️ Scaler dosyası bulunamadı! Lütfen modeli yeniden eğitin.")
        return None, None, None, None, None
    
    # Feature columns'u JSON'dan yükle (sıralama tutarlılığı için)
    import json
    if os.path.exists('feature_cols.json'):
        with open('feature_cols.json', 'r') as f:
            feature_columns = json.load(f)
    else:
        st.error("⚠️ Feature columns dosyası bulunamadı! Lütfen modeli yeniden eğitin.")
        return None, None, None, None, None
    
    #  Dict'leri başlat
    baselines = {}
    calib_biases = {}
    
    if os.path.exists(NEW_DATA_FILE):
        clinical_data = pd.read_csv(NEW_DATA_FILE)
        
        # Her hasta için İLK klinik kaydı kullan
        for patient_id, group in clinical_data.groupby('subject#'):
            row = group.iloc[0]  # First clinical measurement
            pid = int(patient_id)
            
            # Baseline değerini al
            true_updrs = float(row.get('UPDRS_baseline', row['total_UPDRS']))
            baselines[pid] = true_updrs
            
            # Kişisel bias hesapla: bias = clinical_UPDRS - pred_global
            try:
                # Feature'ları row'dan direkt al (sıralama garantili)
                feats = row[feature_columns]
                calib_row = pd.DataFrame([feats.values], columns=feature_columns)
                
                # Global model tahmini
                pred_global = model.predict(scaler.transform(calib_row))[0]
                
                # Bias = Gerçek - Tahmin
                bias = true_updrs - float(pred_global)
                calib_biases[pid] = bias
            except Exception:
                # Feature eksikse bias hesaplanmaz, sadece baseline kaydet
                pass
    
    return model, scaler, feature_columns, baselines, calib_biases

def extract_audio_features(audio_path):
    """Parselmouth ile Gerçek Ses Analizi"""
    sound = parselmouth.Sound(audio_path)
    dur = sound.get_total_duration()
    # Süre kontrolü: talimatla tutarlı olacak şekilde 1-15 sn
    if not (1.0 <= dur <= 15.0):
        raise ValueError(f"Ses kaydı süresi uygun değil ({dur:.2f} sn). Lütfen 1 ile 15 saniye arasında kayıt yükleyin.")
        
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    
    # Pitch Mean (Jitter Abs için gerekli)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    mean_period = 1.0 / mean_f0 if mean_f0 > 0 else 0.0
    
    # Temel Özellikler
    jitter = call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    jitter_abs = jitter * mean_period
    
    shimmer = call([sound, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = call([sound, pulses], "Get shimmer (local_dB)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_db = call(harmonicity, "Get mean", 0, 0) # dB cinsinden
    
    # NHR Hesabı: dB -> Linear dönüşümünün tersi
    # HNR_dB = 10 * log10(Harmonic / Noise)
    # NHR = Noise / Harmonic = 10 ^ (-HNR_dB / 10)
    nhr = 10 ** (-hnr_db / 10) if hnr_db != -200 else 1.0
    
    # Diğer detaylar (Modelin 20 sütununa uyması için türetiyoruz/sabitliyoruz)
    # Not: Bazıları Praat'ta doğrudan olmadığı için yaklaşık katsayılarla türetiliyor
    return {
        'Jitter(%)': jitter, 
        'Jitter(Abs)': jitter_abs, 
        'Jitter:RAP': call(pulses, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3), 
        'Jitter:PPQ5': call(pulses, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3), 
        'Jitter:DDP': call(pulses, "Get jitter (ddp)", 0.0, 0.0, 0.0001, 0.02, 1.3),
        
        'Shimmer': shimmer, 
        'Shimmer(dB)': shimmer_db, 
        'Shimmer:APQ3': call([sound, pulses], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6), 
        'Shimmer:APQ5': call([sound, pulses], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6), 
        'Shimmer:APQ11': call([sound, pulses], "Get shimmer (apq11)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6), 
        'Shimmer:DDA': call([sound, pulses], "Get shimmer (dda)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6),
        
        'NHR': nhr, 
        'HNR': hnr_db, 
        
        'RPDE': 0.4, 'DFA': 0.6, 'PPE': 0.2, 'test_time': 0
    }

# --------------------------------------------------------
# 3. KULLANICI ARAYÜZLERİ
# --------------------------------------------------------

def login_page():
    st.markdown("<h1 style='text-align: center;'>Parkinson AI Sistemi</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        with st.form("login"):
            u = st.text_input("Kullanıcı Adı")
            p = st.text_input("Şifre", type="password")
            if st.form_submit_button("Giriş", type="primary"):
                users = pd.read_csv(USERS_FILE, dtype={'Username': str, 'Password': str})
                user = users[(users['Username']==u) & (users['Password']==p)]
                if not user.empty:
                    st.session_state['user'] = user.iloc[0].to_dict()
                    st.rerun()
                else:
                    st.error("Hatalı Giriş")
        st.info("Demo: `doktor`/`123` veya `ali`/`123`")

def doctor_panel():
    st.title("👨‍⚕️ Doktor Kontrol Paneli")
    
    tab1, tab2, tab3 = st.tabs(["📊 Hasta Takibi", "➕ Yeni Hasta Ekle", "⚙️ AI Model Yönetimi"])
    
    # --- TAB 1: TAKİP ---
    with tab1:
        if not os.path.exists(USERS_FILE):
            st.error("Kullanıcı veritabanı bulunamadı.")
        else:
            users = pd.read_csv(USERS_FILE)
            
            # ✅ Sadece giriş yapan doktora ait hastaları filtrele
            current_doctor_id = st.session_state['user']['ID']
            
            # Tip uyumluluğu için
            users['Doctor_ID'] = pd.to_numeric(users['Doctor_ID'], errors='coerce')
            
            patients = users[
                (users['Role'] == 'Patient') & 
                (users['Doctor_ID'] == current_doctor_id)
            ]
            
            if patients.empty:
                st.info("Listenizde kayıtlı hasta bulunmuyor. 'Yeni Hasta Ekle' sekmesinden ekleyebilirsiniz.")
            else:
                selected_patient = st.selectbox("Hasta Seçin", patients['Name'])
                pat_id = patients[patients['Name'] == selected_patient]['ID'].values[0]
                
                # === Phase 4: Decision Support - Show BOTH Clinical + Predictions ===
                
                # 1. Get Clinical Baseline (from doctor's calibration)
                clinical_baseline = None
                if os.path.exists(NEW_DATA_FILE):
                    try:
                        clinical_data = pd.read_csv(NEW_DATA_FILE)
                        # Tip dönüşümü: subject# sütununu int'e çevir
                        clinical_data['subject#'] = pd.to_numeric(clinical_data['subject#'], errors='coerce').astype('Int64')
                        pat_id_int = int(pat_id)
                        patient_clinical = clinical_data[clinical_data['subject#'] == pat_id_int]
                        if not patient_clinical.empty:
                            baseline_value = patient_clinical['UPDRS_baseline'].values[0]
                            # NaN kontrolü
                            if pd.notna(baseline_value):
                                clinical_baseline = float(baseline_value)
                    except Exception as e:
                        st.warning(f"Klinik baseline yüklenirken hata: {e}")
                
                # 2. Get Prediction History (from patient's home uploads)
                predictions_exist = False
                if os.path.exists(HISTORY_FILE):
                    hist = pd.read_csv(HISTORY_FILE)
                    pat_data = hist[hist['Subject_ID'] == pat_id]
                    predictions_exist = not pat_data.empty
                
                # 3. Display Results
                if clinical_baseline is None and not predictions_exist:
                    st.info("Bu hastaya ait henüz ne klinik kalibrasyon verisi ne de tahmin geçmişi bulunmuyor.")
                else:
                    # Create combined visualization
                    st.subheader(f"📊 {selected_patient} - Hastalık Takibi")
                    
                    # ========== KAYIT SAYISI VE GÜVENİLİRLİK GÖSTERGESİ ==========
                    # Klinik kayıt sayısını hesapla
                    clinical_record_count = 0
                    if os.path.exists(NEW_DATA_FILE):
                        try:
                            clin_df = pd.read_csv(NEW_DATA_FILE)
                            clin_df['subject#'] = pd.to_numeric(clin_df['subject#'], errors='coerce')
                            clinical_record_count = len(clin_df[clin_df['subject#'] == int(pat_id)])
                        except:
                            pass
                    
                    # Evden gönderilen kayıt sayısı
                    home_record_count = len(logs) if 'logs' in dir() and not logs.empty else 0
                    if os.path.exists(HISTORY_FILE):
                        try:
                            home_df = pd.read_csv(HISTORY_FILE)
                            home_record_count = len(home_df[home_df['Subject_ID'] == pat_id])
                        except:
                            pass
                    
                    total_records = clinical_record_count + home_record_count
                    
                    # Güvenilirlik hesapla
                    if total_records >= 10:
                        reliability = "� Yüksek"
                        reliability_desc = "Model bu hasta için güvenilir tahminler yapabilir."
                    elif total_records >= 5:
                        reliability = "🟡 Orta"
                        reliability_desc = "Daha fazla kayıt güvenilirliği artırır."
                    else:
                        reliability = "🔴 Düşük"
                        reliability_desc = "En az 10 kayıt önerilir. Sonuçları dikkatli yorumlayın."
                    
                    # Bilgi kutusu
                    with st.expander("📋 Veri & Model Güvenilirliği", expanded=True):
                        rcol1, rcol2, rcol3 = st.columns(3)
                        rcol1.metric("Klinik Kayıt", f"{clinical_record_count}")
                        rcol2.metric("Evden Gönderilen", f"{home_record_count}")
                        rcol3.metric("Güvenilirlik", reliability)
                        
                        if total_records < 10:
                            st.warning(f"⚠️ {reliability_desc}")
                        else:
                            st.success(f"✅ {reliability_desc}")
                    
                    st.divider()
                    
                    # Show clinical baseline if exists
                    if clinical_baseline is not None:
                        st.success(f"🏥 **Klinik Kalibrasyon (Doktor Ölçümü):** {clinical_baseline:.1f} UPDRS")
                    
                    # Show predictions chart
                    if predictions_exist:
                        st.write("### 📈 Evden Gönderilen Sesler (AI Tahminleri)")
                        
                        # Create chart data
                        # Create chart data
                        # Prediction -> Prediction_Personal
                        chart_data = pat_data[['Date', 'Prediction_Personal']].copy()
                    # Load patient logs
                    logs = pd.DataFrame()
                    if os.path.exists(HISTORY_FILE):
                        all_logs = pd.read_csv(HISTORY_FILE)
                        logs = all_logs[all_logs['Subject_ID'] == pat_id].copy()
                    
                    # Tarih formatını datetime'a çevir
                    if not logs.empty:
                        logs['Date'] = pd.to_datetime(logs['Date'])
                    
                    # --- ÖZET METRİKLER (KPI) ---
                    if not logs.empty:
                        last_record = logs.iloc[-1]
                        last_score = last_record['Prediction_Personal'] # Changed from 'Personal_Prediction' to 'Prediction_Personal' based on original code's pat_data
                        patient_baseline = clinical_baseline # Use the clinical_baseline for delta calculation
                        if 'Delta_Baseline' in last_record:
                            delta = last_record['Delta_Baseline']
                        else:
                            delta = last_score - patient_baseline if patient_baseline is not None else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Son Ölçüm", f"{last_score:.1f}", f"{delta:+.1f} (Baseline'a göre)")
                        col2.metric("Klinik Baseline", f"{patient_baseline:.1f}" if patient_baseline else "N/A")
                        col3.metric("Toplam Kayıt", len(logs))
                        col4.metric("Son Kayıt Tarihi", last_record['Date'].strftime("%d.%m.%Y"))
                        
                        st.divider()

                    if logs.empty:
                        st.info("Bu hastaya ait henüz evden gönderilen veri yok.")
                    else:
                        # --- GRAFİK ALANI (ALTAIR) ---
                        st.subheader("📈 UPDRS Gidişat Analizi")
                        
                        import altair as alt
                        
                        # Ana Çizgi (Kişisel Tahmin)
                        base = alt.Chart(logs).encode(x=alt.X('Date:T', title='Tarih', axis=alt.Axis(format="%d %b")))
                        
                        line = base.mark_line(point=True, color='#2980b9', strokeWidth=3).encode(
                            y=alt.Y('Prediction_Personal:Q', title='UPDRS Skoru', scale=alt.Scale(domain=[0, 100])), # Changed from 'Personal_Prediction' to 'Prediction_Personal'
                            tooltip=[alt.Tooltip('Date', title='Tarih', format='%d.%m.%Y %H:%M'), 
                                     alt.Tooltip('Prediction_Personal', title='Skor', format='.1f'), # Changed from 'Personal_Prediction' to 'Prediction_Personal'
                                     alt.Tooltip('Delta', title='Delta', format='+.1f')] # Changed from 'Delta_Baseline' to 'Delta' based on original code's pat_data
                        ).interactive()
                        
                        # Baseline Çizgisi (Referans)
                        if clinical_baseline: # Use clinical_baseline here
                            rule = base.mark_rule(color='red', strokeDash=[5, 5]).encode(
                                y=alt.datum(clinical_baseline),
                                size=alt.value(2)
                            )
                            chart = (line + rule).properties(height=350)
                        else:
                            chart = line.properties(height=350)
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        st.caption("🔵 Mavi Çizgi: Hastanın evden gönderdiği ölçümler | 🔴 Kırmızı Çizgi: Klinik Baseline (Hedef/Referans)")
                        
                        # --- TABLO ALANI ---
                        st.subheader("📋 Detaylı Veri Dökümü")
                        
                        # Tablo için veri hazırlığı (Gereksiz sütunları gizle)
                        display_df = logs[['Date', 'Prediction_Personal', 'Delta', 'Prediction_Global']].copy() # Changed column names
                        display_df = display_df.sort_values('Date', ascending=False)
                        
                        st.dataframe(
                            display_df,
                            column_config={
                                "Date": st.column_config.DatetimeColumn(
                                    "Tarih & Saat",
                                    format="D MMM YYYY, HH:mm",
                                ),
                                "Prediction_Personal": st.column_config.ProgressColumn( # Changed from 'Personal_Prediction'
                                    "Kişisel Skor",
                                    help="Bias düzeltmesi yapılmış nihai skor",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=100,
                                ),
                                "Delta": st.column_config.NumberColumn( # Changed from 'Delta_Baseline'
                                    "Değişim (Δ)",
                                    help="Baseline'a göre değişim",
                                    format="%.1f",
                                ),
                                "Prediction_Global": st.column_config.NumberColumn( # Changed from 'Raw_Model_Pred'
                                    "Ham Model",
                                    format="%.1f",
                                )
                            },
                            use_container_width=True,
                            hide_index=True
                        )

                    # --- HASTA SİLME BÖLÜMÜ ---
                    st.divider()
                    st.markdown("### ⚠️ Tehlikeli Bölge")
                    with st.expander(f"🔴 {selected_patient} isimli hastayı sil"):
                        st.warning("Bu işlem geri alınamaz! Hastanın tüm kayıtları, ses dosyaları ve tahmin geçmişi silinecektir.")
                        if st.checkbox(f"Evet, {selected_patient} kullanıcısını silmek istiyorum."):
                            if st.button("🗑️ Hastayı Kalıcı Olarak Sil"):
                                try:
                                    # 1. Users DB'den sil
                                    users = users[users['ID'] != pat_id]
                                    users.to_csv(USERS_FILE, index=False)
                                    
                                    # 2. Klinik Veriden sil (NEW_DATA_FILE)
                                    if os.path.exists(NEW_DATA_FILE):
                                        nd = pd.read_csv(NEW_DATA_FILE)
                                        nd = nd[nd['subject#'] != pat_id]
                                        nd.to_csv(NEW_DATA_FILE, index=False)
                                        
                                    # 3. Loglardan sil (HISTORY_FILE)
                                    if os.path.exists(HISTORY_FILE):
                                        hist = pd.read_csv(HISTORY_FILE)
                                        hist = hist[hist['Subject_ID'] != pat_id]
                                        hist.to_csv(HISTORY_FILE, index=False)
                                        
                                    # 4. Ses klasörünü sil
                                    rec_dir = f"patient_recordings/{pat_id}"
                                    if os.path.exists(rec_dir):
                                        shutil.rmtree(rec_dir)
                                        
                                    st.success(f"Hasta {selected_patient} başarıyla silindi System yenileniyor...")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Silme sırasında hata oluştu: {str(e)}")

    # --- TAB 2: YENİ HASTA (COLD START) ---
    with tab2:
        operation_mode = st.radio("İşlem Türü Seçin:", ["➕ Yeni Hasta Kaydı", "📈 Mevcut Hastaya Veri Ekle"], horizontal=True)
        st.divider()

        if operation_mode == "➕ Yeni Hasta Kaydı":
            st.write("Yeni hasta ekleyip, ilk kalibrasyon verisini girin. Bu veri **eğitim setine** eklenecek.")
            
            # Hasta bilgileri
            c1, c2 = st.columns(2)
            name = c1.text_input("Ad Soyad")
            age = c2.number_input("Yaş", 20, 100, 60)
            
            c3, c4 = st.columns(2)
            username = c3.text_input("Kullanıcı Adı (Hasta Girişi İçin)")
            password = c4.text_input("Şifre", type="password")
            
            sex = st.selectbox("Cinsiyet", [0, 1], format_func=lambda x: "Erkek" if x==0 else "Kadın")
            initial_updrs = st.number_input("İlk Muayene UPDRS Skoru (Label)", 0, 100, 20)
            
            # Ses Kaydı Bölümü
            st.subheader("🎙️ Ses Kaydı")
            st.info("""
            **Ses Kaydı Talimatları:**
            1. Hasta derin bir nefes alsın
            2. Tek bir nefeste, sabit ve rahat bir sesle **"aaaaaaa"** desin
            3. Kayıt süresi **3-10 saniye** arasında olmalı
            4. Sessiz bir ortamda kayıt yapın
            """)
            
            # Kayıt yöntemi seçimi
            record_method = st.radio("Kayıt Yöntemi:", ["🎤 Tarayıcıdan Kaydet", "📁 Dosya Yükle"], horizontal=True, key="new_pat_method")
            
            audio_data = None
            
            if record_method == "🎤 Tarayıcıdan Kaydet":
                if AUDIO_RECORDER_AVAILABLE:
                    st.caption("Aşağıdaki butona tıklayarak kaydı başlatın. Tekrar tıklayarak durdurun.")
                    audio_data = audiorecorder("🎙️ Kayda Başla", "⏹️ Kaydı Durdur", key="doctor_recorder_new")
                    
                    if len(audio_data) > 0:
                        duration_sec = len(audio_data) / 1000.0
                        st.success(f"✅ Kayıt tamamlandı! ({duration_sec:.1f} sn)")
                        st.audio(audio_data.export().read(), format="audio/wav")
                        if duration_sec < 1 or duration_sec > 15:
                            st.warning("⚠️ Kayıt süresi limitler dışında (1-15 sn).")
                            audio_data = None
                else:
                    st.error("❌ Ses kayıt bileşeni yüklü değil.")
                    record_method = "📁 Dosya Yükle"
            
            if record_method == "📁 Dosya Yükle":
                wav_file = st.file_uploader("WAV dosyası yükleyin", type=["wav"], key="doctor_upload_new")
                if wav_file:
                    audio_data = wav_file
                    st.audio(wav_file, format="audio/wav")
            
            st.divider()
            
            # Kaydet butonu
            if st.button("💾 Hastayı Kaydet", type="primary"):
                if not name or not username or not password:
                    st.error("❌ Lütfen tüm hasta bilgilerini doldurun!")
                elif audio_data is None or (hasattr(audio_data, '__len__') and len(audio_data) == 0):
                    st.error("❌ Lütfen ses kaydı yapın veya dosya yükleyin!")
                else:
                    try:
                        # 1. Kullanıcıyı Kaydet
                        users = pd.read_csv(USERS_FILE, dtype={'Username': str, 'Password': str})
                        
                        if username in users['Username'].values:
                            st.error("❌ Bu kullanıcı adı zaten alınmış!")
                        else:
                            new_id = int(users['ID'].max()) + 1
                            new_user = pd.DataFrame([{
                                'Username': username, 'Password': password, 'Role': 'Patient',
                                'Name': name, 'ID': new_id, 'Doctor_ID': st.session_state['user']['ID'],
                                'Age': age, 'Sex': sex
                            }])
                            new_user.to_csv(USERS_FILE, mode='a', header=False, index=False)
                        
                            # 2. Sesi Kaydet ve Analiz Et
                            if hasattr(audio_data, 'export'):
                                audio_data.export("temp_calib.wav", format="wav")
                            else:
                                with open("temp_calib.wav", "wb") as f: f.write(audio_data.getbuffer())
                            
                            feats = extract_audio_features("temp_calib.wav")
                            
                            # 3. Eğitim Verisine Ekle (SÜTUN SIRASI GARANTİLİ)
                            # Header tanımı (Orijinal UCI + bizim eklediğimiz UPDRS_baseline)
                            CSV_HEADER = "subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,UPDRS_baseline,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE"
                            
                            # Değerler (Header sırasıyla birebir eşleşmeli)
                            row_values = [
                                new_id,                         # subject#
                                age,                            # age
                                sex,                            # sex
                                0,                              # test_time
                                round(initial_updrs * 0.7, 2),  # motor_UPDRS
                                initial_updrs,                  # total_UPDRS
                                initial_updrs,                  # UPDRS_baseline
                                feats.get('Jitter(%)', 0),
                                feats.get('Jitter(Abs)', 0),
                                feats.get('Jitter:RAP', 0),
                                feats.get('Jitter:PPQ5', 0),
                                feats.get('Jitter:DDP', 0),
                                feats.get('Shimmer', 0),
                                feats.get('Shimmer(dB)', 0),
                                feats.get('Shimmer:APQ3', 0),
                                feats.get('Shimmer:APQ5', 0),
                                feats.get('Shimmer:APQ11', 0),
                                feats.get('Shimmer:DDA', 0),
                                feats.get('NHR', 0),
                                feats.get('HNR', 0),
                                feats.get('RPDE', 0.4),
                                feats.get('DFA', 0.6),
                                feats.get('PPE', 0.2)
                            ]
                            
                            # Dosya yoksa header yaz, sonra veriyi ekle
                            file_exists = os.path.exists(NEW_DATA_FILE)
                            with open(NEW_DATA_FILE, 'a') as f:
                                if not file_exists:
                                    f.write(CSV_HEADER + '\n')
                                f.write(','.join(map(str, row_values)) + '\n')
                            
                            st.success(f"✅ Hasta {name} (ID: {new_id}) sisteme eklendi!")
                            st.info("💡 Modelin hastayı tanıması için 'AI Model Yönetimi' sekmesinden eğitimi başlatın.")
                            st.balloons()
                    except Exception as e:
                        st.error(f"❌ Hata: {str(e)}")

        elif operation_mode == "📈 Mevcut Hastaya Veri Ekle":
            st.info("Seçilen hastaya yeni bir kalibrasyon verisi (örneğin ilaç sonrası veya kontrol muayenesi) ekleyin.")
            
            if not os.path.exists(USERS_FILE):
                st.error("Kullanıcı veritabanı yok.")
            else:
                users = pd.read_csv(USERS_FILE)
                # Sadece bu doktora ait hastalar
                current_doctor_id = st.session_state['user']['ID']
                users['Doctor_ID'] = pd.to_numeric(users['Doctor_ID'], errors='coerce')
                patients = users[(users['Role'] == 'Patient') & (users['Doctor_ID'] == current_doctor_id)]
                
                if patients.empty:
                    st.warning("Hiç hastanız yok. Önce 'Yeni Hasta Kaydı' yapın.")
                else:
                    selected_name = st.selectbox("Hangi hasta için veri gireceksiniz?", patients['Name'])
                    pat_row = patients[patients['Name'] == selected_name].iloc[0]
                    pat_id = pat_row['ID']
                    
                    st.write(f"**Seçilen Hasta:** {selected_name} (ID: {pat_id}, Yaş: {pat_row['Age']})")
                    
                    new_updrs = st.number_input("Yeni Ölçülen UPDRS Skoru", 0, 100, 20, help="Şu anki muayenedeki skor.")
                    
                    # Ses Kayıdı
                    st.subheader("🎙️ Yeni Ses Kaydı")
                    rec_method_ex = st.radio("Kayıt Yöntemi:", ["🎤 Tarayıcıdan Kaydet", "📁 Dosya Yükle"], horizontal=True, key="exist_pat_method")
                    
                    audio_data_ex = None
                    
                    if rec_method_ex == "🎤 Tarayıcıdan Kaydet":
                        if AUDIO_RECORDER_AVAILABLE:
                            st.caption("Kaydı başlat/durdur:")
                            audio_data_ex = audiorecorder("🎙️ Kayda Başla", "⏹️ Kaydı Durdur", key="doctor_recorder_exist")
                            if len(audio_data_ex) > 0:
                                st.success("✅ Kayıt Alındı")
                                st.audio(audio_data_ex.export().read(), format="audio/wav")
                        else:
                            st.error("Kayıt bileşeni yok.")
                    
                    if rec_method_ex == "📁 Dosya Yükle":
                        wav_file_ex = st.file_uploader("WAV Yükle", type=["wav"], key="doctor_upload_exist")
                        if wav_file_ex:
                            audio_data_ex = wav_file_ex
                            st.audio(wav_file_ex, format="audio/wav")
                    
                    if st.button("💾 Ek Veriyi Kaydet", type="primary"):
                        if audio_data_ex is None or (hasattr(audio_data_ex, '__len__') and len(audio_data_ex) == 0):
                            st.error("Ses kaydı eksik.")
                        else:
                            try:
                                # Ses Analizi
                                if hasattr(audio_data_ex, 'export'):
                                    audio_data_ex.export("temp_calib_ex.wav", format="wav")
                                else:
                                    with open("temp_calib_ex.wav", "wb") as f: f.write(audio_data_ex.getbuffer())
                                
                                feats = extract_audio_features("temp_calib_ex.wav")
                                
                                # Veriyi Ekle (SÜTUN SIRASI GARANTİLİ)
                                CSV_HEADER = "subject#,age,sex,test_time,motor_UPDRS,total_UPDRS,UPDRS_baseline,Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP,Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA,NHR,HNR,RPDE,DFA,PPE"
                                
                                row_values = [
                                    pat_id,                          # subject#
                                    pat_row['Age'],                  # age
                                    pat_row['Sex'],                  # sex
                                    0,                               # test_time
                                    round(new_updrs * 0.7, 2),       # motor_UPDRS
                                    new_updrs,                       # total_UPDRS
                                    new_updrs,                       # UPDRS_baseline
                                    feats.get('Jitter(%)', 0),
                                    feats.get('Jitter(Abs)', 0),
                                    feats.get('Jitter:RAP', 0),
                                    feats.get('Jitter:PPQ5', 0),
                                    feats.get('Jitter:DDP', 0),
                                    feats.get('Shimmer', 0),
                                    feats.get('Shimmer(dB)', 0),
                                    feats.get('Shimmer:APQ3', 0),
                                    feats.get('Shimmer:APQ5', 0),
                                    feats.get('Shimmer:APQ11', 0),
                                    feats.get('Shimmer:DDA', 0),
                                    feats.get('NHR', 0),
                                    feats.get('HNR', 0),
                                    feats.get('RPDE', 0.4),
                                    feats.get('DFA', 0.6),
                                    feats.get('PPE', 0.2)
                                ]
                                
                                # Append (dosya zaten var olmalı)
                                with open(NEW_DATA_FILE, 'a') as f:
                                    f.write(','.join(map(str, row_values)) + '\n')
                                
                                st.success(f"✅ {selected_name} için yeni veri eklendi! Modeli yeniden eğitmeyi unutmayın.")
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"Hata: {str(e)}")

    # --- TAB 3: MODEL EĞİTİMİ (SENİN İSTEDİĞİN KISIM) ---
    with tab3:
        st.header("🧠 Yapay Zeka Beynini Güncelle")
        st.write("Yeni hasta eklendiğinde, modelin onu tanıması için 'Yeniden Eğit' butonuna basınız.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Orijinal Veri: {len(pd.read_csv(ORIGINAL_DATA))} satır")
        with col2:
            new_count = len(pd.read_csv(NEW_DATA_FILE)) if os.path.exists(NEW_DATA_FILE) else 0
            st.success(f"Yeni Öğrenilecek Hasta Verisi: {new_count} satır")
            
        if st.button("🚀 SİSTEMİ YENİDEN EĞİT", type="primary"):
            scaler, cols = train_model()
            st.balloons()

def admin_panel():
    st.title("🛡️ Yönetici (Admin) Paneli")
    st.markdown("Bu panelden sistemdeki kullanıcıları yönetebilir ve yeni doktorlar ekleyebilirsiniz.")
    
    tab1, tab2 = st.tabs(["👥 Kullanıcı Yönetimi / Silme", "👨‍⚕️ Yeni Doktor Ekle"])
    
    # --- TAB 1: KULLANICI YÖNETİMİ ---
    with tab1:
        if os.path.exists(USERS_FILE):
            users = pd.read_csv(USERS_FILE)
            
            # Admin kendisini silmesin
            users_view = users[users['Username'] != 'admin']
            
            st.dataframe(users_view[['ID', 'Role', 'Name', 'Username', 'Age']], use_container_width=True)
            
            st.divider()
            st.subheader("🗑️ Kullanıcı Sil")
            
            user_to_delete = st.selectbox("Silinecek Kullanıcıyı Seçin:", users_view['Username'].unique(), index=None, placeholder="Kullanıcı seç...")
            
            if user_to_delete:
                user_row = users_view[users_view['Username'] == user_to_delete].iloc[0]
                role = user_row['Role']
                u_id = user_row['ID']
                
                if role == 'Patient':
                    st.warning(f"⚠️ DİKKAT: '{user_to_delete}' bir HASTA hesabıdır. Silindiğinde tüm ses kayıtları, loglar ve veriler KALICI OLARAK silinir.")
                else:
                    st.warning(f"⚠️ DİKKAT: '{user_to_delete}' bir DOKTOR hesabıdır. Silindiğinde bu doktora bağlı hastalar sahipsiz kalabilir.")
                
                if st.button(f"🔴 {user_to_delete} Hesabını Kalıcı Olarak Sil"):
                    try:
                        # 1. Users DB'den sil
                        new_users = users[users['ID'] != u_id]
                        new_users.to_csv(USERS_FILE, index=False)
                        
                        # Eğer Hastaysa tüm verilerini temizle
                        if role == 'Patient':
                            # Klinik Veriden sil (NEW_DATA_FILE)
                            if os.path.exists(NEW_DATA_FILE):
                                nd = pd.read_csv(NEW_DATA_FILE)
                                # subject# tip dönüşümü güvenliği
                                nd['subject#'] = pd.to_numeric(nd['subject#'], errors='coerce')
                                nd = nd[nd['subject#'] != u_id]
                                nd.to_csv(NEW_DATA_FILE, index=False)
                                
                            # Loglardan sil (HISTORY_FILE)
                            if os.path.exists(HISTORY_FILE):
                                hist = pd.read_csv(HISTORY_FILE)
                                hist = hist[hist['Subject_ID'] != u_id]
                                hist.to_csv(HISTORY_FILE, index=False)
                                
                            # Ses klasörünü sil
                            rec_dir = f"patient_recordings/{u_id}"
                            if os.path.exists(rec_dir):
                                shutil.rmtree(rec_dir)
                                
                        st.success(f"✅ {user_to_delete} başarıyla silindi! Sayfa yenileniyor...")
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Silme hatası: {e}")
        else:
            st.error("Kullanıcı veritabanı bulunamadı.")

    # --- TAB 2: DOKTOR EKLEME ---
    with tab2:
        st.subheader("Yeni Doktor Profili Oluştur")
        
        with st.form("add_doctor_form"):
            d_name = st.text_input("Doktor Adı Soyadı", placeholder="Örn: Dr. Ayşe Yılmaz")
            d_user = st.text_input("Kullanıcı Adı", placeholder="doktor2")
            d_pass = st.text_input("Şifre", type="password")
            
            submitted = st.form_submit_button("Doktor Ekle")
            
            if submitted:
                if not d_name or not d_user or not d_pass:
                    st.error("Tüm alanları doldurunuz.")
                else:
                    users = pd.read_csv(USERS_FILE, dtype={'Username': str})
                    if d_user in users['Username'].values:
                        st.error("Bu kullanıcı adı zaten kullanımda.")
                    else:
                        new_id = int(users['ID'].max()) + 1
                        
                        new_doc = pd.DataFrame([{
                            'Username': d_user,
                            'Password': d_pass,
                            'Role': 'Doctor',
                            'Name': d_name,
                            'ID': new_id,
                            'Doctor_ID': 0, # Doktorların doktoru olmaz
                            'Age': 0, # N/A
                            'Sex': 0
                        }])
                        
                        new_doc.to_csv(USERS_FILE, mode='a', header=False, index=False)
                        st.success(f"✅ {d_name} sisteme başarıyla eklendi!")

def patient_panel():
    user = st.session_state['user']
    st.title(f"Hoşgeldiniz, {user['Name']}")
    
    # Modeli Yükle (artık 5 parametre döner)
    model, scaler, columns, baselines, calib_biases = get_active_model()
    
    if model is None:
        st.error("Sistem şu an bakımda (Model eğitilmemiş). Lütfen doktorunuza haber verin.")
        return
    
    # ✅ Patient ID'yi int olarak al
    pid = int(user['ID'])
    patient_baseline = baselines.get(pid, None)
    bias = calib_biases.get(pid, 0.0)
    
    if patient_baseline is not None:
        st.info(f"📊 Klinik Baseline: {patient_baseline:.1f} UPDRS | Kişisel Kalibrasyon: {bias:+.1f}")

    # Ses Kaydı Bölümü
    st.divider()
    st.subheader("🎙️ Ses Kaydı Analizi")
    
    st.info("""
    **Kayıt Talimatları:**
    1. 🔇 Sessiz bir ortama geçin
    2. 🫁 Derin bir nefes alın  
    3. 🗣️ Tek bir nefeste, sabit sesle **"aaaaaaa"** deyin
    4. ⏱️ Kayıt süresi **3-10 saniye** olmalı
    """)
    
    # Kayıt yöntemi seçimi
    record_method = st.radio("Kayıt Yöntemi:", ["🎤 Tarayıcıdan Kaydet", "📁 Dosya Yükle"], horizontal=True, key="patient_method")
    
    audio_data = None
    
    if record_method == "🎤 Tarayıcıdan Kaydet":
        if AUDIO_RECORDER_AVAILABLE:
            st.markdown("##### 🔴 Kayıt Kontrolü")
            st.caption("Mikrofon butonuna tıklayarak kaydı başlatın, tekrar tıklayarak durdurun.")
            
            # Kayıt bileşeni
            audio_data = audiorecorder("🎙️ Kayda Başla", "⏹️ Kaydı Durdur", key="patient_recorder")
            
            if len(audio_data) > 0:
                # Kayıt süresi hesapla (pydub'da len() milisaniye döndürür)
                duration_ms = len(audio_data)
                duration_sec = duration_ms / 1000.0
                
                st.success(f"✅ Kayıt tamamlandı!")
                
                # Detaylı bilgiler
                col1, col2, col3 = st.columns(3)
                col1.metric("⏱️ Süre", f"{duration_sec:.1f} sn")
                col2.metric("📊 Örnekleme", f"{audio_data.frame_rate} Hz")
                col3.metric("🔊 Kanal", f"{audio_data.channels}")
                
                # Kaydı dinle
                st.audio(audio_data.export().read(), format="audio/wav")
                
                # Süre kontrolü
                if duration_sec < 1:
                    st.warning("⚠️ Kayıt çok kısa! En az 1 saniye olmalı.")
                    audio_data = None
                elif duration_sec > 15:
                    st.warning("⚠️ Kayıt çok uzun! 15 saniyeden kısa olmalı.")
                    audio_data = None
        else:
            st.error("❌ Ses kayıt bileşeni kullanılamıyor. Dosya yükleme yöntemini kullanın.")
            record_method = "📁 Dosya Yükle"
    
    if record_method == "📁 Dosya Yükle":
        wav_file = st.file_uploader("WAV dosyası yükleyin", type=["wav"], key="patient_upload")
        if wav_file:
            audio_data = wav_file
            st.audio(wav_file, format="audio/wav")
            st.success("✅ Dosya yüklendi!")
    
    st.divider()
    
    # Analiz butonu
    if st.button("🔬 Analiz Et", type="primary"):
        if audio_data is None or (hasattr(audio_data, '__len__') and len(audio_data) == 0):
            st.error("❌ Lütfen önce ses kaydı yapın veya dosya yükleyin!")
        else:
            try:
                # 1. Dosya Kaydetme (Arşivleme)
                save_dir = f"patient_recordings/{pid}"
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = user['Name'].lower().replace(" ", "_")
                save_path = f"{save_dir}/{timestamp}_{safe_name}.wav"
                
                # Kaydet
                if hasattr(audio_data, 'export'):
                    # audiorecorder'dan gelen veri
                    audio_data.export("temp_pat.wav", format="wav")
                    audio_data.export(save_path, format="wav")
                else:
                    # Dosya yükleme
                    with open("temp_pat.wav", "wb") as f:
                        f.write(audio_data.getbuffer())
                    with open(save_path, "wb") as f:
                        f.write(audio_data.getbuffer())
                
                st.success(f"📁 Ses kaydı arşivlendi: {save_path}")
                
                # 2. Özellik Çıkar
                feats = extract_audio_features("temp_pat.wav")
                
                # 3. Tahmin için input hazırla (subject# YOK!)
                input_row = pd.DataFrame([{
                    'age': user['Age'],
                    'sex': user['Sex'],
                    'test_time': 0,
                    **feats
                }])
                input_row = input_row[columns]
                
                # 4. ✅ Global tahmin
                pred_global = model.predict(scaler.transform(input_row))[0]
                
                # 5. ✅ Kişisel kalibrasyon uygula
                pred_personal = max(0, pred_global + bias)
                
                # 6. Delta hesapla (None olarak başlat)
                delta = None
                
                # 7. Sonuçları göster
                if patient_baseline is not None:
                    delta = pred_personal - patient_baseline
                    
                    # 4 metrik göster
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Klinik Baseline", f"{patient_baseline:.1f}")
                    col2.metric("Ham Model", f"{pred_global:.1f}")
                    col3.metric("Kişisel Skor", f"{pred_personal:.1f}")
                    col4.metric("Δ", f"{delta:+.1f}", delta_color="inverse")
                    
                    # Anomali kontrolü
                    if delta < -10:
                        st.error("⚠️ Anormal iyileşme - kaydın doğruluğunu kontrol edin.")
                    elif delta > 15:
                        st.warning("⚠️ Hızlı kötüleşme - doktorunuza danışın.")
                else:
                    st.metric("UPDRS Tahmini (Global)", f"{pred_global:.1f}")
                    st.info("💡 Kişisel kalibrasyon için doktor kaydı gerekli.")
                
                # 8. Kaydet (hem global hem personal)
                rec = pd.DataFrame([{
                    'Date': datetime.now().strftime("%Y-%m-%d"), 
                    'Subject_ID': pid,
                    'Age': user['Age'], 
                    'Sex': user['Sex'], 
                    'Prediction_Global': pred_global,
                    'Prediction_Personal': pred_personal,
                    'Delta': delta,
                    'Baseline': patient_baseline,
                    'Jitter': feats['Jitter(%)'], 
                    'Shimmer': feats['Shimmer'], 
                    'HNR': feats['HNR']
                }])
                rec.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
                st.success("✅ Sonuçlar kaydedildi!")
            except Exception as e:
                st.error(f"❌ Hata: {str(e)}")

# --------------------------------------------------------
# ANA AKIŞ
# --------------------------------------------------------
if 'user' not in st.session_state or st.session_state['user'] is None:
    login_page()
else:
    # Sidebar - SSS
    with st.sidebar:
        st.divider()
        st.header("❓ SSS & Bilgi")
        
        with st.expander("UPDRS Nedir?"):
            st.caption("""
            **Birleşik Parkinson Hastalığı Değerlendirme Ölçeği.** 
            0-100+ arasında bir puandır.
            *0-10:* Sağlıklı/Çok Hafif
            *10-30:* Hafif/Orta
            *30+:* İleri Seviye
            """)
            
        with st.expander("Sistem Nasıl Çalışır?"):
            st.caption("""
            Sesinizden (Jitter, Shimmer gibi) 20 farklı özellik çıkarılır.
            Yapay zeka (XGBoost) bu özelliklerden tahmini UPDRS skorunu bulur.
            Ancak en önemlisi **Kişisel Kalibrasyondur**. Sistem sizin "normalinizi" öğrenir ve sadece **değişimleri** (kötüleşme/iyileşme) takip eder.
            """)
            
        with st.expander("Kayıt Nasıl Olmalı?"):
            st.caption("""
            - Arka plan sessiz olmalı.
            - Tek bir nefeste, sabit "aaaa" denilmeli.
            - Mikrofon ağza ne çok yakın ne çok uzak olmalı.
            - Süre 3-10 saniye idealdir.
            """)
            
        with st.expander("Delta (Δ) Nedir?"):
            st.caption("""
            Sizin ilk muayene (baseline) skorunuz ile şu anki durumunuz arasındaki farktır.
            **+ Değer:** Kötüleşme (Skor arttı)
            **- Değer:** İyileşme (Skor düştü)
            **0:** Stabil
            """)
            
        st.info("Version 1.0.0")
        
        if st.button("Çıkış Yap"):
            st.session_state['user'] = None
            st.rerun()

    # Yönlendirme
    role = st.session_state['user']['Role']
    if role == 'Doctor':
        doctor_panel()
    elif role == 'Admin':
        admin_panel()
    else:
        patient_panel()