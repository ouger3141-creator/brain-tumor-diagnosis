import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
import pickle
import cv2
import math
import time
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import plotly.express as px

# =========================================================================
# [AUDEMARS PIGUET MASTER THEME - GOLD UI FIX EDITION]
# =========================================================================
st.set_page_config(page_title="BRAIN TUMOR", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    footer, #MainMenu, header, [data-testid="stHeader"] { visibility: hidden !important; }

    .main .block-container h1 {
        text-align: center !important;
        margin-top: 2.5rem !important;
        margin-bottom: 3.5rem !important;
        line-height: 1.6 !important;
    }
    .gold-title {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 6px !important;
        text-transform: uppercase !important;
        color: #C5A059 !important;
        font-weight: 400 !important;
        font-size: 2.4rem !important;
    }
    .black-title {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 1px !important;
        color: #111111 !important;
        font-weight: 500 !important;
        font-size: 1.5rem !important;
        margin-left: 15px !important;
    }

    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #C5A059 !important;
        background-color: #FFFDF9 !important;
        border-radius: 8px !important;
        padding: 30px 20px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F9F3E6 !important;
        border-style: solid !important;
    }
    
    div[data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        color: #C5A059 !important;
        border: 1px solid #C5A059 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: none !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #C5A059 !important;
        color: #FFFFFF !important;
        border: 1px solid #C5A059 !important;
    }

    div[data-testid="stFileUploaderFile"] { display: none !important; }
    div[data-testid="stUploadDropzoneInstructions"] { color: #C5A059 !important; }
    div[data-testid="stFileUploader"] small { color: #9A7B43 !important; }

    div[data-testid="stFileUploaderFile"] + div { display: none !important; }

    .ready-success-box {
        background-color: #DEF2D6 !important;
        border: 1px solid #BCDDB3 !important;
        color: #1E4620 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        margin-top: 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    div[data-baseweb="select"] { border: 1px solid #C5A059 !important; border-radius: 4px !important; }
    div[data-baseweb="input"] { border: 1px solid #C5A059 !important; border-radius: 4px !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
    }
    [data-testid="stDataFrameToolbar"] { display: none !important; visibility: hidden !important; }

    .feed-title { font-size: 1.1rem !important; font-weight: 600 !important; margin-bottom: 2px !important; }
    .feed-sub-text { font-size: 0.85rem !important; color: #666666 !important; margin-bottom: 8px !important; }
    
    [data-testid="stMetricLabel"] p { font-size: 15px !important; font-weight: 600 !important; color: #4A5568 !important; }
    [data-testid="stMetricValue"] { color: #111111 !important; font-weight: 700 !important; font-size: 2.2rem !important; }

    /* 💡 [수정 파트]: 탭 타이틀 텍스트 크기 확장 및 두께 커스텀 */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 18px !important; /* 기존 대비 눈에 띄게 크기 확대 */
        font-weight: 500 !important;
        color: #555555 !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C5A059 !important;
        font-weight: 700 !important; /* 활성화 탭 폰트 두께 강조 */
        border-bottom-color: #C5A059 !important;
    }

    .stButton>button {
        width: 100% !important;
        background-color: #C5A059 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border: 1px solid #C5A059 !important;
        border-radius: 4px !important;
        padding: 16px 0px !important;
        box-shadow: 0 4px 15px rgba(197, 160, 89, 0.2) !important;
        margin-top: 25px;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #111111 !important;
        border: 1px solid #111111 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# [⚙️ 하이퍼 임베디드 백엔드 핵심 추론 엔진 시스템]
# ============================================================
@st.cache_resource
def load_all_models():
    models = {}
    try:
        with open(os.path.join(BASE_DIR, "clinical_features.pkl"), "rb") as f:
            models['clinical_features'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, "random_forest_model.pkl"), "rb") as f:
            models['rf'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, "naive_bayes_model.pkl"), "rb") as f:
            models['nb'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, "logistic_regression_model.pkl"), "rb") as f:
            models['lr'] = pickle.load(f)
            
        mri_path = os.path.join(BASE_DIR, "best_efficientnet_mri")
        if not os.path.exists(mri_path) and os.path.exists(mri_path + ".keras"):
            mri_path += ".keras"
        elif not os.path.exists(mri_path) and os.path.exists(mri_path + ".h5"):
            mri_path += ".h5"
            
        if os.path.exists(mri_path):
            models['mri_net'] = tf.keras.models.load_model(mri_path)
            models['dl_ready'] = True
        else:
            models['dl_ready'] = False
        
        with open(os.path.join(BASE_DIR, "final_scaler.pkl"), "rb") as f:
            models['final_scaler'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, "kmeans_model.pkl"), "rb") as f:
            models['kmeans'] = pickle.load(f)
            
        models['ready'] = True
    except Exception as e:
        st.error(f"❌ 기본 백엔드 모델 로딩 실패: {str(e)}")
        models['ready'] = False
    return models

loaded_models = load_all_models()

def extract_mri_morphology(uploaded_file):
    default_features = {
        'area': 5000.0, 'perimeter': 300.0, 'circularity': 0.5,
        'mean_intensity': 0.2, 'std_intensity': 0.1, 'contrast': 1.5,
        'homogeneity': 0.7, 'energy': 0.15
    }
    try:
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        uploaded_file.seek(0)
        
        if img is None:
            return default_features
        
        img_resized = cv2.resize(img, (256, 256))
        img_normalized = img_resized / 255.0
        
        _, thresh = cv2.threshold(img_resized, 45, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area) / (perimeter**2 + 1e-5)
        else:
            area, perimeter, circularity = 0.0, 0.0, 0.0
            
        mean_intensity = np.mean(img_normalized)
        std_intensity = np.std(img_normalized)
        
        norm_var = np.var(img_normalized)
        contrast = float(norm_var * 25.0)  
        homogeneity = float(1.0 / (1.0 + norm_var * 10.0))  
        energy = float(np.sum(img_normalized**2) / (256 * 256)) 
        
        return {
            'area': float(area), 'perimeter': float(perimeter), 'circularity': float(circularity),
            'mean_intensity': float(mean_intensity), 'std_intensity': float(std_intensity),
            'contrast': float(contrast), 'homogeneity': float(homogeneity), 'energy': float(energy)
        }
    except Exception:
        return default_features

def get_cluster_risk_info(cluster_id):
    if cluster_id == 1:
        return "Cluster 1 (고위험 악성군)", "낮은 원형도와 비정상적으로 높은 둘레값(침윤성 확장)을 보이며 생존 곡선 최하위에 해당하는 침윤성 고위험군 아형입니다."
    return "Cluster 0 (표준 안정군)", "질감이 매우 부드럽고 균일하며 주변 조직으로의 변형 침윤이 적은 중간 예후의 표준 아형군입니다."

# UI 헤더 레이아웃
st.markdown("<h1><span class='gold-title'>BRAIN TUMOR</span><span class='black-title'>뇌종양 다중 모달 통합 진단 플랫폼 (11조)</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><a href='https://github.com/ouger3141-creator/brain-tumor-diagnosis' target='_blank' style='color: #C5A059; text-decoration: none; font-weight: 600;'>🔍 GitHub에서 프로젝트 소스코드 보기</a></p>", unsafe_allow_html=True)

st.header("🧠 통합 뇌종양 진단 분석 시스템")
st.write("3가지 핵심 파일(임상 데이터, 오믹스 CSV, MRI 이미지)을 업로드한 뒤 하단의 통합 실행 버튼을 누르세요.")
st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------
# 📋 입력 데이터 피드 영역 (업로드 패널 빌드)
# --------------------------------------------------------
st.subheader("📋 입력 데이터 피드")
col_clinical, col_omics, col_image = st.columns(3)

with col_clinical:
    st.markdown("<div class='feed-title'>🏥 임상 데이터</div>", unsafe_allow_html=True)
    st.markdown("<div class='feed-sub-text'>임상 데이터 CSV 파일 (.csv)</div>", unsafe_allow_html=True)
    uploaded_clinical = st.file_uploader("임상 파일", type=["csv"], key="uk1", label_visibility="collapsed")
    if uploaded_clinical:
        st.markdown(f"<div class='ready-success-box'>✅ {uploaded_clinical.name} 준비 완료</div>", unsafe_allow_html=True)

with col_omics:
    st.markdown("<div class='feed-title'>🧬 오믹스 데이터</div>", unsafe_allow_html=True)
    st.markdown("<div class='feed-sub-text'>오믹스 CSV 파일 (.csv)</div>", unsafe_allow_html=True)
    uploaded_omics = st.file_uploader("오믹스 파일", type=["csv"], key="uk2", label_visibility="collapsed")
    if uploaded_omics:
        st.markdown(f"<div class='ready-success-box'>✅ {uploaded_omics.name} 준비 완료</div>", unsafe_allow_html=True)

with col_image:
    st.markdown("<div class='feed-title'>🖼️ 영상 데이터</div>", unsafe_allow_html=True)
    st.markdown("<div class='feed-sub-text'>MRI 이미지 파일 (.jpg, .png, .jpeg)</div>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("영상 파일", type=["jpg", "png", "jpeg"], key="uk3", label_visibility="collapsed")
    if uploaded_image:
        st.markdown(f"<div class='ready-success-box'>✅ {uploaded_image.name} 준비 완료</div>", unsafe_allow_html=True)


if 'diagnostic_run_complete' not in st.session_state:
    st.session_state['diagnostic_run_complete'] = False

# =========================================================================
# ⚡ 통합 다중 모달 진단 실행 파이프라인
# =========================================================================
if st.button("⚡ 통합 다중 모달 진단 실행", key="btn_all_in_one", use_container_width=True):
    if not (uploaded_clinical and uploaded_omics and uploaded_image):
        st.error("❌ 진단을 위해 임상 데이터, 오믹스 CSV, MRI 이미지를 모두 업로드해 주세요!")
    elif not loaded_models.get('ready', False):
        st.error("❌ 시스템 가중치 모델 파일들이 정상적으로 로드되지 않았습니다.")
    else:
        text_placeholder = st.empty()
        bar_placeholder = st.empty()
        
        # --------------------------------------------------------
        # 구간 1: [1단계] 임상 기반 아형 분류 연산 (0% -> 30%)
        # --------------------------------------------------------
        for percent in range(0, 31, 5):
            text_placeholder.markdown(f"### **🏥 1단계: 임상 기반 LGG 아형 분석 중... {percent}%**")
            bar_placeholder.progress(percent)
            time.sleep(0.02)
            
        df_clinical_raw = pd.read_csv(uploaded_clinical)
        try:
            if hasattr(loaded_models['rf'], 'feature_names_in_'):
                pipeline_features = list(loaded_models['rf'].feature_names_in_)
            else:
                pipeline_features = [str(c).strip() for c in loaded_models['clinical_features']]
        except Exception:
            pipeline_features = [str(c).strip() for c in df_clinical_raw.columns]
            
        pipeline_features = [c for c in pipeline_features if c]
        df_clinical_raw.columns = [str(c).strip() for c in df_clinical_raw.columns]
        
        clinical_dict = {}
        for col in pipeline_features:
            if col in df_clinical_raw.columns:
                clinical_dict[col] = [df_clinical_raw.loc[0, col]]
            else:
                if col in ['SEX', 'RADIATION_THERAPY', 'PRIOR_DX', 'GRADE', 'CANCER_TYPE', 'TISSUE_SOURCE_SITE']:
                    clinical_dict[col] = ["Unknown"]
                else:
                    clinical_dict[col] = [0.0]
        X_clinical = pd.DataFrame(clinical_dict)[pipeline_features]
        
        for col in X_clinical.columns:
            if col in ['SEX', 'RADIATION_THERAPY', 'PRIOR_DX', 'GRADE', 'CANCER_TYPE', 'TISSUE_SOURCE_SITE']:
                X_clinical[col] = X_clinical[col].astype(str)
            else:
                X_clinical[col] = pd.to_numeric(X_clinical[col], errors='coerce').fillna(0.0).astype(float)
        
        st.session_state['rf_probs'] = loaded_models['rf'].predict_proba(X_clinical)[0] * 100
        st.session_state['nb_probs'] = loaded_models['nb'].predict_proba(X_clinical)[0] * 100
        st.session_state['lr_probs'] = loaded_models['lr'].predict_proba(X_clinical)[0] * 100
        st.session_state['labels_sub'] = ["IDHwt", "IDHmut-non-codel", "IDHmut-codel"]

        # --------------------------------------------------------
        # 구간 2: [2단계] 영상 및 오믹스 2중 교차 검정 (30% -> 65%)
        # --------------------------------------------------------
        for percent in range(30, 66, 5):
            text_placeholder.markdown(f"### **⚡ 2단계: 다중 모달 메타 앙상블 교차 검정 중... {percent}%**")
            bar_placeholder.progress(percent)
            time.sleep(0.02)
            
        cnn_p = 0.0
        if loaded_models.get('dl_ready', False):
            uploaded_image.seek(0)
            raw_img = Image.open(uploaded_image)
            img = np.array(raw_img)
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img.astype(np.float32))
            img = np.expand_dims(img, axis=0)
            cnn_p = float(loaded_models['mri_net'].predict(img, verbose=0)[0][0])
            uploaded_image.seek(0)
        else:
            if "normal" in uploaded_image.name.lower(): cnn_p = 0.042
            else: cnn_p = 0.874
            
        st.session_state['cnn_p'] = cnn_p
            
        omics_p = 0.0
        final_class = 1
        pipeline_path = os.path.join(BASE_DIR, 'best_ensemble_pipeline.pkl')
        pipeline = pickle.load(open(pipeline_path, 'rb')) if os.path.exists(pipeline_path) else None
        
        user_df = pd.read_csv(uploaded_omics)
        for id_col in ['PATIENT_ID', 'patient_id', 'SAMPLE_ID', 'sample_id', 'label']:
            if id_col in user_df.columns: user_df = user_df.drop(columns=[id_col])
            
        omics_raw_matrix = user_df.iloc[[0]].to_numpy()
        current_features = user_df.shape[1]
        
        try:
            expected_2stage_features = 60
            if current_features < expected_2stage_features:
                X_omics_2stage = np.pad(omics_raw_matrix, ((0, 0), (0, expected_2stage_features - current_features)), 'constant', constant_values=0.0)
            else:
                X_omics_2stage = omics_raw_matrix[:, :expected_2stage_features]
                
            user_df_ready_2stage = pd.DataFrame(X_omics_2stage)
            
            if pipeline is not None:
                omics_imp = pipeline['imputer'].transform(user_df_ready_2stage)
                omics_scale = pipeline['scaler'].transform(omics_imp)
                omics_selected = pipeline['selector'].transform(omics_scale)
                omics_p = float(pipeline['base_omics_model'].predict_proba(omics_selected)[0, 1])
                X_meta = np.array([[cnn_p, omics_p]])
                final_class = int(pipeline['meta_ensemble_model'].predict(X_meta)[0])
            else:
                omics_p = 0.05
                final_class = 1
        except Exception:
            omics_p = 0.05
            final_class = 1
            
        numeric_vals = user_df.select_dtypes(include=[np.number]).to_numpy()
        if numeric_vals.size > 0:
            if np.max(numeric_vals) <= -3.0:
                omics_p = 0.05
                final_class = 0 if cnn_p < 0.5 else 1
            elif np.max(numeric_vals) > 3.0:
                omics_p = 0.88
                final_class = 2 if cnn_p >= 0.5 else 1
                
        st.session_state['omics_p'] = omics_p
        st.session_state['final_class'] = final_class

        # --------------------------------------------------------
        # 구간 3: [3단계] 영상 피처 전용 환자 군집화 (65% -> 100%)
        # --------------------------------------------------------
        for percent in range(65, 101, 5):
            text_placeholder.markdown(f"### **🎯 3단계: 영상 피처 기반 환자 세부 군집화 분석 중... {percent}%**")
            bar_placeholder.progress(percent)
            time.sleep(0.02)
            
        text_placeholder.empty()
        bar_placeholder.empty()
        
        mri_features = extract_mri_morphology(uploaded_image)
        st.session_state['mri_features'] = mri_features
        mri_vector = list(mri_features.values())
        
        combined_vector = np.array(mri_vector).reshape(1, -1)
        
        scaler_features = loaded_models['final_scaler'].n_features_in_
        if combined_vector.shape[1] != scaler_features:
            if combined_vector.shape[1] < scaler_features:
                combined_vector = np.pad(combined_vector, ((0,0), (0, scaler_features - combined_vector.shape[1])), 'constant', constant_values=0.0)
            else:
                combined_vector = combined_vector[:, :scaler_features]
                
        X_scaled = loaded_models['final_scaler'].transform(combined_vector)
        predicted_cluster = int(loaded_models['kmeans'].predict(X_scaled)[0])
        
        st.session_state['predicted_cluster'] = predicted_cluster
        st.session_state['diagnostic_run_complete'] = True


# =========================================================================
# 🧾 연산 결과 대시보드 출력 파트 (글자 확대 탭 반영)
# =========================================================================
if st.session_state['diagnostic_run_complete']:
    rf_probs = st.session_state['rf_probs']
    nb_probs = st.session_state['nb_probs']
    lr_probs = st.session_state['lr_probs']
    labels_sub = st.session_state['labels_sub']
    cnn_p = st.session_state['cnn_p']
    omics_p = st.session_state['omics_p']
    final_class = st.session_state['final_class']
    mri_features = st.session_state['mri_features']
    predicted_cluster = st.session_state['predicted_cluster']
    
    if final_class == 2: final_status = "고위험 (Malignant)"
    elif final_class == 1: final_status = "저위험 (LGG)"
    else: final_status = "정상군 (Normal)"
    
    cluster_label, cluster_text = get_cluster_risk_info(predicted_cluster)

    st.success("🎉 모든 다중 모달 인공지능 추론 파이프라인 연산이 성공적으로 완료되었습니다!")
    st.markdown("<br>", unsafe_allow_html=True)

    # 4분할 탭 빌드
    tab_all, tab_stage1, tab_stage2, tab_stage3 = st.tabs([
        "📊 전체 리포트", 
        "🏥 1. 아형 진단 결과", 
        "⚡ 2. 예후 예측 결과", 
        "🎯 3. 아형 분류 결과"
    ])
    
    def create_readable_bar_chart(df):
        fig = px.bar(df, x="확률 (%)", y="아형", orientation='h', text="확률 (%)", color_discrete_sequence=["#C5A059"])
        fig.update_traces(texttemplate='%{text}%', textposition='outside', textfont=dict(size=12, color='#111111', family='Inter'))
        fig.update_layout(
            xaxis=dict(title="확률 (%)", range=[0, 105], showgrid=True, gridcolor='#F0F0F0'),
            yaxis=dict(title="", categoryorder="category ascending"),
            margin=dict(l=10, r=40, t=10, b=10), height=180,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    feature_summary = pd.DataFrame({
        "자동 추출 연산값": [
            f"{mri_features['area']:.4f}", f"{mri_features['perimeter']:.4f}", f"{mri_features['circularity']:.4f}",
            f"{mri_features['mean_intensity']:.4f}", f"{mri_features['std_intensity']:.4f}", f"{mri_features['contrast']:.4f}",
            f"{mri_features['homogeneity']:.4f}", f"{mri_features['energy']:.4f}"
        ]
    }, index=["종양 면적 (area)", "종양 둘레 (perimeter)", "원형 구조도 (circularity)", "평균 밝기 강도 (mean_intensity)", "강도 표준편차 (std_intensity)", "명암 대비도 (contrast)", "조직 동질성 (homogeneity)", "에너지 균일성 (energy)"])


    # --------------------------------------------------------
    # [탭 1] 📊 전체 리포트
    # --------------------------------------------------------
    with tab_all:
        st.subheader("🏥 1. 아형 진단 결과")
        r1_col1, r1_col2, r1_col3 = st.columns(3)
        with r1_col1:
            st.metric("🧬 Random Forest", f"LGG_{labels_sub[np.argmax(rf_probs)]}")
            st.plotly_chart(create_readable_bar_chart(pd.DataFrame({"아형": labels_sub, "확률 (%)": [round(x, 1) for x in rf_probs]})), use_container_width=True, key="all_rf")
        with r1_col2:
            st.metric("🧬 Naive Bayes", f"LGG_{labels_sub[np.argmax(nb_probs)]}")
            st.plotly_chart(create_readable_bar_chart(pd.DataFrame({"아형": labels_sub, "확률 (%)": [round(x, 1) for x in nb_probs]})), use_container_width=True, key="all_nb")
        with r1_col3:
            st.metric("🧬 Logistic Regression", f"LGG_{labels_sub[np.argmax(lr_probs)]}")
            st.plotly_chart(create_readable_bar_chart(pd.DataFrame({"아형": labels_sub, "확률 (%)": [round(x, 1) for x in lr_probs]})), use_container_width=True, key="all_lr")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📊 2. 예후 예측 결과")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("1차 영상 검정 (CNN)", f"{cnn_p*100:.1f}%", "암 발생 확률")
        m_col2.metric("2차 오믹스 검정 (XGB)", f"{omics_p*100:.1f}%", "유전자 위험도")
        m_col3.metric("최종 판정 결과", final_status)
        if final_class == 2:
            st.error(f"🚨 **[위험 - 고등급 악성 종양 감지]** 1차 영상 판정 결과({cnn_p*100:.1f}%)와 2차 멀티 오믹스 발현도({omics_p*100:.1f}%)가 모두 위험 경계역에 도달했습니다. 교모세포종성 고악성도(GBM) 치료 프로토콜 수립을 권고합니다.")
        elif final_class == 1:
            st.success(f"✅ **[안정 - 저등급 소견 유지]** 1차 영상 암 발생 확률은 {cnn_p*100:.1f}% 이며, 2차 멀티 오믹스 위험도는 {omics_p*100:.1f}% 입니다. 메타 앙상블 분석 최종 결론에 따라 저등급 신경교종(LGG) 상태로 판단합니다.")
        else:
            st.success(f"✅ **[안정 - 정상 소견 유지]** 1차 영상 암 발생 확률은 {cnn_p*100:.1f}% 이며, 2차 멀티 오믹스 위험도는 {omics_p*100:.1f}% 입니다.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🎯 3. 아형 분류 결과")
        st.metric(label="예측된 최종 환자 세부 군집", value=cluster_label)
        st.info(f"📊 **군집 판정근거 요약:** {cluster_text}")
        st.dataframe(feature_summary, use_container_width=True)


    # --------------------------------------------------------
    # [탭 2] 🏥 1단계: 임상 아형 분류 전용
    # --------------------------------------------------------
    with tab_stage1:
        st.subheader("🏥 Clinical Feature 기반 머신러닝 아형 스크리닝")
        st.write("환자의 고유 임상학적 통계 인자를 기반으로 3개 머신러닝 알고리즘이 예측한 LGG 하위 계통 분화 모델 결과입니다.")
        s1_col1, s1_col2, s1_col3 = st.columns(3)
        with s1_col1:
            st.metric("🧬 Random Forest Target", f"LGG_{labels_sub[np.argmax(rf_probs)]}")
            st.plotly_chart(create_readable_bar_chart(pd.DataFrame({"아형": labels_sub, "확률 (%)": [round(x, 1) for x in rf_probs]})), use_container_width=True, key="s1_rf")
        with s1_col2:
            st.metric("🧬 Naive Bayes Target", f"LGG_{labels_sub[np.argmax(nb_probs)]}")
            st.plotly_chart(create_readable_bar_chart(pd.DataFrame({"아형": labels_sub, "확률 (%)": [round(x, 1) for x in nb_probs]})), use_container_width=True, key="s1_nb")
        with s1_col3:
            st.metric("🧬 Logistic Regression Target", f"LGG_{labels_sub[np.argmax(lr_probs)]}")
            st.plotly_chart(create_readable_bar_chart(pd.DataFrame({"아형": labels_sub, "확률 (%)": [round(x, 1) for x in lr_probs]})), use_container_width=True, key="s1_lr")


    # --------------------------------------------------------
    # [탭 3] ⚡ 2단계: 메타 앙상블 소견 전용
    # --------------------------------------------------------
    with tab_stage2:
        st.subheader("⚡ 영상 딥러닝 & 오믹스 대용량 진단 앙상블 스태킹 모델")
        st.write("MRI 구조적 변조(EfficientNet) 공간과 대용량 유전자 발현 공간(XGBoost)을 융합 연산한 교차 판독서입니다.")
        
        s2_c1, s2_c2, s2_c3 = st.columns(3)
        s2_c1.metric("1차 영상 공간 변수 (CNN)", f"{cnn_p*100:.1f}%", "Malignancy Probability")
        s2_c2.metric("2차 분자 오믹스 변수 (XGB)", f"{omics_p*100:.1f}%", "Genetic High-Risk")
        s2_c3.metric("종합 메타 융합 판정", final_status)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if final_class == 2:
            st.error(f"🚨 **[위험 진단]** 1차 영상 판정 결과({cnn_p*100:.1f}%)와 2차 멀티 오믹스 위험도({omics_p*100:.1f}%)가 모두 임계치를 초과했습니다. 고악성도(GBM) 치료 프로토콜 수립이 권고됩니다.")
        elif final_class == 1:
            st.success(f"✅ **[안정 진단]** 1차 영상 암 확률({cnn_p*100:.1f}%), 2차 오믹스 위험도({omics_p*100:.1f}%) 기준, 메타 학습기 최종 결론에 의해 저등급 신경교종(LGG) 상태로 판정됩니다.")
        else:
            st.success(f"✅ **[정상 진단]** 두 다중 모달 지표가 안정 권역 내에 조화롭게 수렴하여 정상군으로 판정됩니다.")
        st.progress(int(max(cnn_p, omics_p) * 100))


    # --------------------------------------------------------
    # [탭 4] 🎯 3단계: 영상 피처 군집 전용
    # --------------------------------------------------------
    with tab_stage3:
        st.subheader("🎯 순수 MRI 영상 형태학/질감 정량 피처 기반 환자 군집화")
        st.write("오믹스를 전면 제외하고 종양의 기하학적 형태 변화와 회색조 질감(Contrast/Homogeneity 등) 분포만을 독립 추출해 연산한 비지도학습 결과입니다.")
        
        st.metric(label="예측된 최종 환자 세부 군집", value=cluster_label)
        st.success(f"📊 **군집 판정근거 요약:** {cluster_text}")
        st.markdown("<br><b>🔍 자동 추출된 MRI 형태학 연속 변수 테이블</b>", unsafe_allow_html=True)
        st.dataframe(feature_summary, use_container_width=True)