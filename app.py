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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
        # Orijinal veri setinin sütun yapısını kopyala ama boş oluştur
        if os.path.exists(ORIGINAL_DATA):
            df = pd.read_csv(ORIGINAL_DATA, nrows=1)
            pd.DataFrame(columns=df.columns).to_csv(NEW_DATA_FILE, index=False)

init_system()

# --------------------------------------------------------
# 2. YAPAY ZEKA MOTORU (EĞİTİM VE TAHMİN)
# --------------------------------------------------------

def train_model():
    """
    Eğitim fonksiyonu:
    - Orijinal UCI data + yeni hasta verisini birleştirir
    - subject# feature leakage'ını engeller
    - Patient-wise split kullanır (aynı hasta train & test'te olmaz)
    - MUTLAK UPDRS tahmin eder (delta değil)
    - Model, scaler ve feature columns'u kaydeder
    """
    status = st.empty()
    status.info("🧠 Yapay Zeka Motoru: Veriler birleştiriliyor ve eğitim başlıyor...")
    
    # Verileri Yükle
    df_orig = pd.read_csv(ORIGINAL_DATA)
    if os.path.exists(NEW_DATA_FILE):
        df_new = pd.read_csv(NEW_DATA_FILE)
        # Birleştir (Concatenate)
        full_data = pd.concat([df_orig, df_new], ignore_index=True)
    else:
        full_data = df_orig

    # ✅ Feature Hazırlığı: subject# ÇIKARILACAK (feature leakage engellenir)
    X = full_data.drop(['subject#', 'total_UPDRS', 'motor_UPDRS'], axis=1)
    
    # Baseline/delta sütunları varsa çıkar (bunlar feature olmayacak)
    cols_to_drop = [c for c in ['UPDRS_baseline', 'delta_UPDRS'] if c in X.columns]
    if cols_to_drop:
        X = X.drop(cols_to_drop, axis=1)
    
    # ✅ Hedef: MUTLAK total_UPDRS (delta değil)
    y = full_data['total_UPDRS']
    
    # ✅ Patient-wise split için group bilgisi
    groups = full_data['subject#']

    # ✅ NaN içeren satırları düş (özellik/etiket eksikleri modeli bozmasın)
    non_na_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[non_na_mask]
    y = y.loc[non_na_mask]
    groups = groups.loc[non_na_mask]

    # ✅ Sayısal tiplere dönüştür (CSV okuma kaynaklı string tipler varsa)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Patient-wise train/test split (aynı hasta train & test'te olmaz)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost Eğitimi
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # ✅ Model, Scaler ve Feature Columns'u Kaydet
    model.save_model(MODEL_FILE)
    
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    import json
    with open('feature_cols.json', 'w') as f:
        json.dump(list(X.columns), f)
    
    # Test Skoru
    score = model.score(X_test_scaled, y_test)
    
    # Bilgi Mesajları
    status.success(f"✅ Eğitim Tamamlandı! Test R² skoru: {score:.3f}")
    st.info(f"ℹ️ Feature sayısı: {len(X.columns)} (subject# hariç)")
    st.info(f"ℹ️ Train: {len(train_idx)} kayıt, Test: {len(test_idx)} kayıt")
    
    unique_train_patients = groups.iloc[train_idx].nunique()
    unique_test_patients = groups.iloc[test_idx].nunique()
    st.success(f"✅ Patient-wise split: Train {unique_train_patients} hasta, Test {unique_test_patients} hasta")
    
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
    
    # ✅ Scaler'ı pickle'dan yükle
    import pickle
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    else:
        st.error("⚠️ Scaler dosyası bulunamadı! Lütfen modeli yeniden eğitin.")
        return None, None, None, None, None
    
    # ✅ Feature columns'u JSON'dan yükle (sıralama tutarlılığı için)
    import json
    if os.path.exists('feature_cols.json'):
        with open('feature_cols.json', 'r') as f:
            feature_columns = json.load(f)
    else:
        st.error("⚠️ Feature columns dosyası bulunamadı! Lütfen modeli yeniden eğitin.")
        return None, None, None, None, None
    
    # ✅ Dict'leri başlat
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
            
            # ✅ Kişisel bias hesapla: bias = clinical_UPDRS - pred_global
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
    
    # Temel Özellikler
    jitter = call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    shimmer = call([sound, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    # HNR negatif veya sıfır olabilir; NHR hesaplamasını güvenli hale getir
    safe_hnr = abs(hnr) if hnr != 0 else 1e-6
    
    # Diğer detaylar (Modelin 20 sütununa uyması için türetiyoruz/sabitliyoruz)
    return {
        'Jitter(%)': jitter, 'Jitter(Abs)': jitter * 0.0001, 'Jitter:RAP': jitter * 0.3, 
        'Jitter:PPQ5': jitter * 0.4, 'Jitter:DDP': jitter * 0.9,
        'Shimmer': shimmer, 'Shimmer(dB)': shimmer * 10, 'Shimmer:APQ3': shimmer * 0.5, 
        'Shimmer:APQ5': shimmer * 0.6, 'Shimmer:APQ11': shimmer * 0.7, 'Shimmer:DDA': shimmer * 1.5,
        'NHR': 1/safe_hnr, 'HNR': hnr, 
        'RPDE': 0.4, 'DFA': 0.6, 'PPE': 0.2, 'test_time': 0
    }

# --------------------------------------------------------
# 3. KULLANICI ARAYÜZLERİ
# --------------------------------------------------------

def login_page():
    st.markdown("<h1 style='text-align: center;'>🧠 Parkinson AI Sistemi</h1>", unsafe_allow_html=True)
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
            patients = users[users['Role'] == 'Patient']
            
            if patients.empty:
                st.info("Sistemde kayıtlı hasta bulunmuyor.")
            else:
                selected_patient = st.selectbox("Hasta Seçin", patients['Name'])
                pat_id = patients[patients['Name'] == selected_patient]['ID'].values[0]
                
                # === Phase 4: Decision Support - Show BOTH Clinical + Predictions ===
                
                # 1. Get Clinical Baseline (from doctor's calibration)
                clinical_baseline = None
                if os.path.exists(NEW_DATA_FILE):
                    try:
                        clinical_data = pd.read_csv(NEW_DATA_FILE)
                        patient_clinical = clinical_data[clinical_data['subject#'] == pat_id]
                        if not patient_clinical.empty:
                            clinical_baseline = patient_clinical['total_UPDRS'].values[0]
                    except Exception:
                        pass
                
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
                    
                    # Show clinical baseline if exists
                    if clinical_baseline is not None:
                        st.success(f"🏥 **Klinik Kalibrasyon (Doktor Ölçümü):** {clinical_baseline:.1f} UPDRS")
                    
                    # Show predictions chart
                    if predictions_exist:
                        st.write("### 📈 Evden Gönderilen Sesler (AI Tahminleri)")
                        
                        # Create chart data
                        # Create chart data
                        # ✅ Prediction -> Prediction_Personal
                        chart_data = pat_data[['Date', 'Prediction_Personal']].copy()
                        chart_data['Date'] = pd.to_datetime(chart_data['Date'])
                        chart_data = chart_data.sort_values('Date')
                        
                        # Add baseline as reference line if available
                        if clinical_baseline is not None:
                            st.write(f"*Yeşil çizgi: Klinik baseline ({clinical_baseline:.1f}), Mavi: Kişisel AI tahminleri*")
                        
                        st.line_chart(chart_data.set_index('Date')['Prediction_Personal'])
                        
                        # Show data table (Prediction_Global eklendi)
                        st.dataframe(pat_data[['Date', 'Prediction_Global', 'Prediction_Personal', 'Delta', 'Jitter', 'Shimmer', 'HNR']])
                    else:
                        st.info("Hasta henüz evden ses göndermemiş. Klinik kalibrasyon mevcut.")

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
        st.write("Yeni hasta ekleyip, ilk kalibrasyon verisini girin. Bu veri **eğitim setine** eklenecek.")
        
        # Hasta bilgileri
        c1, c2 = st.columns(2)
        name = c1.text_input("Ad Soyad")
        age = c2.number_input("Yaş", 50, 100, 60)
        
        c3, c4 = st.columns(2)
        username = c3.text_input("Kullanıcı Adı (Hasta Girişi İçin)")
        password = c4.text_input("Şifre", type="password")
        
        sex = st.selectbox("Cinsiyet", [0, 1], format_func=lambda x: "Erkek" if x==0 else "Kadın")
        initial_updrs = st.number_input("İlk Muayene UPDRS Skoru (Label)", 0, 100, 20)
        
        # Ses Kaydı Bölümü
        st.divider()
        st.subheader("🎙️ Ses Kaydı")
        st.info("""
        **Ses Kaydı Talimatları:**
        1. Hasta derin bir nefes alsın
        2. Tek bir nefeste, sabit ve rahat bir sesle **"aaaaaaa"** desin
        3. Kayıt süresi **3-10 saniye** arasında olmalı
        4. Sessiz bir ortamda kayıt yapın
        """)
        
        # Kayıt yöntemi seçimi
        record_method = st.radio("Kayıt Yöntemi:", ["🎤 Tarayıcıdan Kaydet", "📁 Dosya Yükle"], horizontal=True)
        
        audio_data = None
        
        if record_method == "🎤 Tarayıcıdan Kaydet":
            if AUDIO_RECORDER_AVAILABLE:
                st.markdown("##### 🔴 Kayıt Kontrolü")
                st.caption("Aşağıdaki butona tıklayarak kaydı başlatın. Tekrar tıklayarak durdurun.")
                
                # Kayıt bileşeni
                audio_data = audiorecorder("🎙️ Kayda Başla", "⏹️ Kaydı Durdur", key="doctor_recorder")
                
                if len(audio_data) > 0:
                    # Kayıt süresi hesapla (pydub'da len() milisaniye döndürür)
                    duration_ms = len(audio_data)
                    duration_sec = duration_ms / 1000.0
                    
                    st.success(f"✅ Kayıt tamamlandı!")
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
                st.error("❌ Ses kayıt bileşeni yüklü değil. Dosya yükleme yöntemini kullanın.")
                record_method = "📁 Dosya Yükle"
        
        if record_method == "📁 Dosya Yükle":
            wav_file = st.file_uploader("WAV dosyası yükleyin", type=["wav"], key="doctor_upload")
            if wav_file:
                audio_data = wav_file
                st.audio(wav_file, format="audio/wav")
                st.success("✅ Dosya yüklendi!")
        
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
                            # audiorecorder'dan gelen veri
                            audio_data.export("temp_calib.wav", format="wav")
                        else:
                            # Dosya yükleme
                            with open("temp_calib.wav", "wb") as f:
                                f.write(audio_data.getbuffer())
                        
                        feats = extract_audio_features("temp_calib.wav")
                        
                        # 3. Eğitim Verisine Ekle
                        train_row = {
                            'subject#': new_id, 'age': age, 'sex': sex,
                            'motor_UPDRS': initial_updrs * 0.7,
                            'total_UPDRS': initial_updrs,
                            'UPDRS_baseline': initial_updrs,
                            **feats
                        }
                        
                        train_df = pd.DataFrame([train_row])
                        header = not os.path.exists(NEW_DATA_FILE)
                        train_df.to_csv(NEW_DATA_FILE, mode='a', header=header, index=False)
                        
                        st.success(f"✅ Hasta {name} (ID: {new_id}) sisteme eklendi!")
                        st.info("💡 Modelin hastayı tanıması için 'AI Model Yönetimi' sekmesinden eğitimi başlatın.")
                        st.balloons()
                except Exception as e:
                    st.error(f"❌ Hata: {str(e)}")

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
    with st.sidebar:
        if st.button("Çıkış"):
            st.session_state['user'] = None
            st.rerun()
            
    if st.session_state['user']['Role'] == 'Doctor':
        doctor_panel()
    else:
        patient_panel()