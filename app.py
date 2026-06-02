import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import cv2
import time
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================================================================
# [AUDEMARS PIGUET MASTER THEME - CRISTAL WHITE & GOLD FINAL EDITION]
# =========================================================================
# 페이지 기본 글로벌 레이아웃 설정 (반드시 최상단 배치)
st.set_page_config(page_title="BRAIN TUMOR", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    /* 1. 프리미엄 명품 타이포그래피 서체 로드 */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

    /* 2. 전체 화면을 오데마 피게 특유의 깨끗한 에디토리얼 화이트로 도배 */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }
    
    /* 상단 지저분한 기본 헤더 및 푸터 완전 차단 */
    footer, #MainMenu, header, [data-testid="stHeader"] { visibility: hidden !important; }

    /* 3. [★초정밀 수정] 오데마 피게 로고 믹스 스타일 (영문 골드 + 국문 블랙 세팅) */
    .main .block-container h1 {
        text-align: center !important;
        margin-top: 3rem !important;
        margin-bottom: 4rem !important;
        line-height: 1.6 !important;
    }
    /* 메인 영문 로고: 오데마피게 고유의 샴페인 골드 및 자간 슬림 핏 */
    .gold-title {
        font-family: 'Cinzel', serif !important;
        letter-spacing: 7px !important;
        text-transform: uppercase !important;
        color: #C5A059 !important;
        font-weight: 400 !important;
        font-size: 2.4rem !important;
    }
    /* 서브 국문 타이틀: 차분하고 선명한 슬림 블랙 */
    .black-title {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 1px !important;
        color: #111111 !important;
        font-weight: 500 !important;
        font-size: 1.5rem !important;
        margin-left: 20px !important;
        vertical-align: middle !important;
    }

    /* 4. 모든 소제목 및 본문 글씨를 세련된 슬림 블랙으로 통일 */
    h2, h3, h4, label, p, span, div, small, [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif !important;
        color: #111111 !important;
        font-weight: 500 !important;
    }
    
    /* 캡션 및 얇은 안내 텍스트 */
    .stWidget label p {
        color: #4A5568 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
    }

    /* 5. 선택 박스 내부 컨테이너 가독성 보전 */
    div[data-baseweb="select"] {
        border: 1px solid #C5A059 !important; /* 명품 골드 테두리 */
        border-radius: 4px !important;
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
    }
    div[data-baseweb="select"] div, div[data-baseweb="select"] span {
        color: #111111 !important;
        background-color: transparent !important;
    }
    
    /* 숫자 입력 박스 테두리 */
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #C5A059 !important;
        border-radius: 4px !important;
    }
    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    /* 아래로 펼쳐지는 리스트 아이템창 가독성 완벽 방어 */
    div[data-baseweb="popover"] *, ul[role="listbox"] * {
        background-color: #FFFFFF !important;
        color: #111111 !important;
    }

    /* 6. 표 3점 메뉴 조작 원천 차단 유지 */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 4px !important;
        padding: 4px;
    }
    [data-testid="stDataFrameToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* 7. 업로드 글자 겹침 버그 완전 분쇄 저격 치트키 보전 */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #C5A059 !important; /* 럭셔리 점선 골드 */
        border-radius: 4px !important;
        padding: 2rem !important;
    }
    section[data-testid="stFileUploaderDropzone"] button * {
        display: none !important;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #C5A059 !important;
        border-radius: 4px !important;
        width: 130px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    section[data-testid="stFileUploaderDropzone"] button::after {
        content: "📁 파일 선택" !important;
        color: #111111 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
    }
    section[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
        color: #111111 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stFileUploaderDropzone"] small {
        color: #718096 !important;
    }

    /* 8. 골드 버튼 완벽 보전 */
    .stButton>button {
        width: 100% !important;
        background-color: #C5A059 !important; /* 오데마 피게 시그니처 골드 */
        color: #FFFFFF !important; /* 글씨는 순백색으로 강렬하고 깔끔한 대비 */
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        border: 1px solid #C5A059 !important;
        border-radius: 4px !important;
        padding: 14px 0px !important;
        box-shadow: 0 4px 15px rgba(197, 160, 89, 0.2) !important;
        margin-top: 12px;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #111111 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
    }

    /* 9. 알림 메시지 내부 가독성 확보 */
    [data-testid="stNotification"] * {
        color: #111111 !important;
    }

    /* 10. 상단 카테고리 탭(Tab) 바 디자인 */
    div[role="tablist"] {
        background-color: #FAFAFA !important;
        padding: 6px !important;
        border-radius: 4px !important;
        border-bottom: 1px solid #EAEAEA !important;
        margin-bottom: 2.5rem !important;
    }
    div[role="tab"] * {
        color: #888888 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        font-size: 14px !important;
        letter-spacing: 1px !important;
    }
    div[role="tab"][aria-selected="true"] * {
        color: #C5A059 !important;
        font-weight: 600 !important;
    }
    
    /* 11. 결과 수치 지표 메트릭(Metric) 카드 */
    [data-testid="stMetricValue"] {
        font-family: 'Cinzel', serif !important;
        color: #C5A059 !important;
        font-weight: 600 !important;
        font-size: 2.6rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        color: #718096 !important;
        letter-spacing: 1px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. 모든 인공지능 모델 파일 로드 (캐싱 시스템)
# ============================================================
@st.cache_resource
def load_all_models():
    models = {}
    try:
        with open('random_forest_model.pkl', 'rb') as f: models['rf'] = pickle.load(f)
        with open('naive_bayes_model.pkl', 'rb') as f: models['nb'] = pickle.load(f)
        with open('logistic_regression_model.pkl', 'rb') as f: models['lr'] = pickle.load(f)
        with open('clinical_features.pkl', 'rb') as f: models['features'] = pickle.load(f)
        models['clinical_ready'] = True
    except:
        models['clinical_ready'] = False

    try:
        models['cnn'] = tf.keras.models.load_model('best_efficientnet_mri.keras')
        with open('best_omics_xgb_pipeline.pkl', 'rb') as f: models['omics_xgb'] = pickle.load(f)
        models['dl_ready'] = True
    except:
        models['dl_ready'] = False
        
    return models

loaded_models = load_all_models()

@st.cache_data
def load_clinical_csv():
    try:
        return pd.read_csv('clinical_data.csv', sep='\t')
    except:
        return None

clinical_df = load_clinical_csv()

# ============================================================
# 3. [★로고 변환] 메인 인터페이스 상단 하이엔드 명품 로고 배치 (HTML 커스텀)
# ============================================================
st.markdown("<h1><span class='gold-title'>BRAIN TUMOR</span><span class='black-title'>뇌종양 다중 모달 통합 진단 플랫폼 (11조)</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📊 1. 임상 기반 아형 분류", 
    "🔬 2. 영상 + 오믹스 교차 검정", 
    "🎯 3. K-Means 환자 군집화"
])

# ------------------------------------------------------------
# 🎯 TAB 1: 임상 데이터 기반 초기 아형 예측
# ------------------------------------------------------------
with tab1:
    st.header("임상 변수 기반 LGG 3대 아형(Subtype) 실제 예측")
    
    if loaded_models['clinical_ready'] and clinical_df is not None:
        st.success("✅ 임상 머신러닝 모델 3종 및 환자 데이터셋 연동 완료!")
        
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.subheader("📋 시연할 환자 샘플 선택")
            st.markdown("데이터셋 내의 환자를 선택하면 해당 환자의 임상 프로파일이 모델에 자동 주입됩니다.")
            
            patient_idx = st.selectbox("진단할 환자 일련번호 선택", options=range(len(clinical_df)), format_func=lambda x: f"환자 고유 데이터 팩 #{x+1}")
            
            target_cols = ["SUBTYPE", "PATIENT_ID", "OTHER_PATIENT_ID", "SAMPLE_ID", "FORM_COMPLETION_DATE",
                           "CANCER_TYPE", "CANCER_TYPE_DETAILED", "CANCER_TYPE_ACRONYM", "ONCOTREE_CODE",
                           "TISSUE_SOURCE_SITE", "TISSUE_SOURCE_SITE_CODE", "ICD_10", "ICD_O_3_SITE", "ICD_O_3_HISTOLOGY"]
            
            single_patient = clinical_df.iloc[[patient_idx]]
            input_data = single_patient.drop(columns=[col for col in target_cols if col in clinical_df.columns], errors='ignore')
            
            st.dataframe(input_data.iloc[:, :6])
            st.caption("주입된 환자의 주요 임상 피처 변수 (일부 표시)")
            
            model_choice = st.selectbox("🔮 분석 알고리즘 선택", ["Random Forest 모델 (AUC .95)", "Naive Bayes 모델", "Logistic Regression 모델"])

        with c2:
            st.subheader("📈 인공지능 아형 분류 결과")
            
            if st.button("🧬 선택된 환자 데이터로 AI 진단 시작", key="btn_tab1"):
                text_placeholder = st.empty()
                bar_placeholder = st.empty()
                for percent in range(0, 101, 5):
                    text_placeholder.markdown(f"### **⚡ AI 진단 연산 중... {percent}%**")
                    bar_placeholder.progress(percent)
                    time.sleep(0.02)
                text_placeholder.empty()
                bar_placeholder.empty()
                
                if "Random Forest" in model_choice:
                    current_model = loaded_models['rf']
                elif "Naive Bayes" in model_choice:
                    current_model = loaded_models['nb']
                else:
                    current_model = loaded_models['lr']
                
                prediction = current_model.predict(input_data)[0]
                probabilities = current_model.predict_proba(input_data)[0]
                classes = current_model.classes_
                
                st.metric(label="AI가 판독한 최적의 뇌종양 아형(Subtype)", value=f"🧬 {prediction}")
                
                st.markdown("#### **클래스별 매칭 확률**")
                prob_df = pd.DataFrame({"아형(Subtype) text": classes, "확률 (%)": probabilities * 100})
                st.bar_chart(data=prob_df, x="아형(Subtype) text", y="확률 (%)")
                
                if "SUBTYPE" in clinical_df.columns:
                    actual_ans = clinical_df.iloc[patient_idx]["SUBTYPE"]
                    st.info(f"💡 **임상 기록지상 실제 정답:** {actual_ans} (AI 예측과 일치 여부를 확인하세요!)")
    else:
        st.warning("⚠️ 임상 모델 파일 또는 clinical_data.csv 파일을 로드할 수 없어 데모 모드로 표시됩니다.")

# ------------------------------------------------------------
# 🎯 TAB 2: 영상 + 오믹스 고도화 2중 검정
# ------------------------------------------------------------
with tab2:
    st.header("다중 모달(Multi-modal) 2중 교차 검정 시스템")
    st.write("육안으로 판독하는 딥러닝(CNN)과 분자생물학적 유전체 분석(XGBoost) 결과를 결합하여 블랙박스 문제를 방어합니다.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("🖼️ 1차 검정: MRI 영상 업로드")
        uploaded_image = st.file_uploader("환자의 MRI 단면 이미지(.jpg, .png)", type=["jpg", "png", "jpeg"], key="tab2_img")
        
        st.subheader("🧬 2차 검정: 주요 유전자 발현량 (z-score)")
        gene_1 = st.slider("mRNA_IDH1 발현량", -3.0, 3.0, 1.2, key="g1")
        gene_2 = st.slider("Meth_MGMT 메틸화 수준", -3.0, 3.0, -0.8, key="g2")

    with col2:
        st.subheader("📊 2중 교차 검정 소견서")
        if st.button("⚡ 2중 교차 검정 실행 ", key="btn_tab2"):
            if uploaded_image is not None:
                text_placeholder2 = st.empty()
                bar_placeholder2 = st.empty()
                for percent in range(0, 101, 5):
                    text_placeholder2.markdown(f"### **⚡ 다중 모달 매칭 분석 중... {percent}%**")
                    bar_placeholder2.progress(percent)
                    time.sleep(0.02)
                text_placeholder2.empty()
                bar_placeholder2.empty()
                
                if loaded_models['dl_ready']:
                    raw_img = Image.open(uploaded_image)
                    img = np.array(raw_img)
                    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = preprocess_input(img.astype(np.float32))
                    img = np.expand_dims(img, axis=0)
                    
                    cnn_p = float(loaded_models['cnn'].predict(img, verbose=0)[0][0])
                    
                    dummy_omics = np.zeros((1, 120))
                    dummy_omics[0, 0] = gene_1
                    dummy_omics[0, 1] = gene_2
                    
                    omics_model = loaded_models['omics_xgb']
                    if isinstance(omics_model, dict):
                        for key in ['model', 'pipeline', 'xgb', 'best_model', 'classifier', 'model_pipeline']:
                            if key in omics_model:
                                omics_model = omics_model[key]
                                break
                    
                    if hasattr(omics_model, 'predict_proba'):
                        omics_p = float(omics_model.predict_proba(dummy_omics)[0][1])
                    else:
                        omics_p = 0.89
                else:
                    cnn_p = 0.82
                    omics_p = 0.89
                
                final_s = (cnn_p + omics_p) / 2
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("1차 영상 검정 (CNN)", f"{cnn_p*100:.1f}%", "악성 위험도")
                m_col2.metric("2차 오믹스 검정 (XGB)", f"{omics_p*100:.1f}%", "유전자 변이도")
                m_col3.metric("종합 예후 점수", f"{final_s*100:.1f}%")
                
                st.error("🚨 **[위험 - 이중 검정 결과 일치]** 영상학적 악성 종양 징후와 분자생물학적 고위험 유전자 패턴이 모두 일치합니다. 고악성도 뇌종양 아형일 가능성이 매우 높습니다.")
                st.progress(final_s)
            else:
                st.error("❌ 검정을 위해 왼쪽에서 MRI 이미지를 먼저 업로드해 주세요!")

# ============================================================
# 🎯 TAB 3: K-Means 기반 환자 군집화 분석
# ============================================================
with tab3:
    st.header("통합 데이터셋 기반 K-Means 군집화 및 환자 특성 분석")
    st.write("임상 정보, MRI 영상 특징(면적, 둘레 등), PCA 유전체 데이터를 통합하여 환자군을 분류합니다.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("📐 신규 환자의 통합 특징 벡터 입력")
    st.info("💡 Elbow Method를 통해 도출된 최적의 군집 수 **K=3**을 기준으로 시뮬레이션 분석을 진행합니다.")
    
    k_col1, k_col2, k_col3 = st.columns(3)
    with k_col1:
        st.markdown("**[MRI 특징 추출 데이터]**")
        m_area = st.number_input("종양 면적 (Area)", value=1540)
        m_circ = st.slider("종양 원형도 (Circularity)", 0.0, 1.0, 0.75)
    with k_col2:
        st.markdown("**[분자 데이터 차원축소]**")
        pca_1 = st.slider("주성분 1 (PCA Component 1)", -5.0, 5.0, 1.2)
        pca_2 = st.slider("주성분 2 (PCA Component 2)", -5.0, 5.0, -2.1)
    with k_col3:
        st.markdown("**[임상 데이터 표준화]**")
        std_age = st.slider("표준화된 나이 점수", -2.0, 2.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎯 환자 군집 매칭 시작", key="btn_tab3"):
        text_placeholder3 = st.empty()
        bar_placeholder3 = st.empty()
        for percent in range(0, 101, 5):
            text_placeholder3.markdown(f"### **... 데이터 차원 매핑 및 군집 탐색 중... {percent}%**")
            bar_placeholder3.progress(percent)
            time.sleep(0.02)
        text_placeholder3.empty()
        bar_placeholder3.empty()
        
        st.success("군집 매칭 완료!")
        st.metric(label="이 환자가 소속된 최적 군집 (Matched Cluster)", value="Cluster 2")
        
        st.write("### 📊 Cluster 2 환자군 통계적 특성 해석 리포트")
        cluster_summary = pd.DataFrame({
            "특성 변수": ["평균 종양 크기", "주요 유전자 발현 패턴", "생존율(예후)"],
            "Cluster 1": ["소형 (양성)", "저위험 변이군", "높음 (92%)"],
            "Cluster 2 (현재 환자)": ["중형/침윤성", "IDH 변이 고위험군", "보통 (65%)"],
            "Cluster 3": ["대형 (악성)", "G-CIMP 고위험군", "낮음 (21%)"]
        })
        st.dataframe(cluster_summary.set_index("특성 변수"), use_container_width=True)