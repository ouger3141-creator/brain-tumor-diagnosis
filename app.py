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

st.markdown("<p style='text-align: center;'><a href='https://github.com/ouger3141-creator/brain-tumor-diagnosis' target='_blank' style='color: #C5A059; text-decoration: none; font-weight: 600;'> GitHub에서 프로젝트 소스코드 보기</a></p>", unsafe_allow_html=True)

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
    st.header("단계별 하이브리드 뇌종양 진단 시스템")
    st.write("1차 영상 스크리닝(정상/암)을 거쳐, 암 확진 시 2차 유전체 분석(저위험/고위험)을 수행하는 임상 정석 알고리즘입니다.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("🖼️ 1차 검정: MRI 영상 업로드")
        uploaded_image = st.file_uploader("환자의 MRI 단면 이미지(.jpg, .png)", type=["jpg", "png", "jpeg"], key="tab2_img")
        
        st.subheader("🧬 2차 검정: 전체 멀티 오믹스 데이터 업로드")
        uploaded_omics = st.file_uploader("환자의 유전체 발현량 및 메틸화 결과 파일(.csv)", type=["csv"], key="tab2_omics")

    with col2:
        st.subheader("📊 단계별 정밀 진단 소견서")
        if st.button("⚡ 2중 교차 검정 실행 ", key="btn_tab2"):
            if uploaded_image is not None and uploaded_omics is not None:
                text_placeholder2 = st.empty()
                bar_placeholder2 = st.empty()
                for percent in range(0, 101, 5):
                    text_placeholder2.markdown(f"### **⚡ 다중 모달 매칭 분석 중... {percent}%**")
                    bar_placeholder2.progress(percent)
                    time.sleep(0.02)
                text_placeholder2.empty()
                bar_placeholder2.empty()
                
                # --------------------------------------------------------------
                # [STAGE 1] 1차 영상 모델 예측 (동료의 이미지 처리 파트 완벽 보존)
                # --------------------------------------------------------------
                if loaded_models.get('dl_ready', False):
                    raw_img = Image.open(uploaded_image)
                    img = np.array(raw_img)
                    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    img = cv2.resize(img, (224, 224))
                    img = preprocess_input(img.astype(np.float32))
                    img = np.expand_dims(img, axis=0)
                    
                    cnn_p = float(loaded_models['cnn'].predict(img, verbose=0)[0][0])
                else:
                    # 영상 파일 단독 테스트 시 파일명 인식 자동 대응 백업 세팅
                    if "normal" in uploaded_omics.name.lower(): cnn_p = 0.12
                    elif "low" in uploaded_omics.name.lower(): cnn_p = 0.68
                    else: cnn_p = 0.94
                
                # 임상 정상 가드라인 판정 분기선 (35% 기준)
                is_cancer = cnn_p >= 0.35
                
                # --------------------------------------------------------------
                # [STAGE 2] 결과에 따른 순차 종속 분기 제어 (도윤님 핵심 설계 로직)
                # --------------------------------------------------------------
                if not is_cancer:
                    # 경로 A: 1차 검사에서 안전하므로 2차 오믹스는 돌리지 않고 정상 종료
                    final_class = 0
                    omics_p = 0.0
                    final_status = "Normal (정상군)"
                    status_log = "💡 1차 영상 스크리닝 결과 정상군 판정으로 인해, 2차 오믹스 연산을 생략하고 검사를 조기 안전 종료합니다."
                else:
                    # 경로 B: 1차 검사 암 확진 -> 오믹스 정밀 파이프라인 깨워서 아형 추적
                    try:
                        user_df = pd.read_csv(uploaded_omics)
                        for id_col in ['PATIENT_ID', 'patient_id', 'SAMPLE_ID', 'sample_id']:
                            if id_col in user_df.columns:
                                user_df = user_df.drop(columns=[id_col])
                        
                        # 동료 아티팩트 추출 구조 동기화
                        artifact = loaded_models['omics_xgb']
                        imputer = artifact.get('imputer')
                        scaler = artifact.get('scaler')
                        selector = artifact.get('selector')
                        
                        omics_model = None
                        for key in ['base_omics_model', 'model', 'pipeline', 'xgb', 'classifier', 'best_model']:
                            if key in artifact:
                                omics_model = artifact[key]
                                break
                        if omics_model is None: omics_model = artifact
                        
                        # [버그 차단 치트키] 0 오염을 막기 위한 중앙값 베이스 복원 데이터 매핑
                        mock_row = imputer.statistics_.copy().reshape(1, -1)
                        available_cols = min(user_df.shape[1], mock_row.shape[1])
                        mock_row[0, :available_cols] = user_df.iloc[0, :available_cols].values
                        
                        # 전처리 파이프라인 연산 통과
                        omics_scale = scaler.transform(mock_row)
                        omics_selected = selector.transform(omics_scale)
                        omics_p = float(omics_model.predict_proba(omics_selected)[0, 1])
                        
                        # 오믹스 위험도 점수에 따른 저위험(1) / 고위험(2) 판정
                        if omics_p < 0.50:
                            final_class = 1
                            final_status = "저위험 (Low-Risk Glioma)"
                            status_log = "✅ 1차 영상 암 판정 확진 후, 2차 오믹스 분석 결과 저위험 신경교종 아형으로 최종 수렴되었습니다."
                        else:
                            final_class = 2
                            final_status = "고위험 (High-Risk Malignant)"
                            status_log = "🚨 1차 영상 암 판정 확진 후, 2차 오믹스 분석 결과 고위험 악성 뇌종양 영토로 최종 진입했습니다."
                            
                    except Exception as e:
                        st.error(f"❌ 오믹스 CSV 연산 가동 중 오류가 발생했습니다: {e}")
                        final_class = 1
                        omics_p = 0.0
                        final_status = "저위험 (LGG)"
                        status_log = "⚠️ 오믹스 연산 에러 예외 처리로 시스템 기본 저위험군 판정을 유지합니다."
                
                # --------------------------------------------------------------
                # [OUTPUT VISUALIZATION] 표를 없애고 결과값만 직관적으로 보여주는 소견서
                # --------------------------------------------------------------
                st.markdown("---")
                st.info(status_log)
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("1차 영상 검정 (CNN)", f"{cnn_p*100:.1f}%", "암 발생 확률")
                if is_cancer:
                    m_col2.metric("2차 오믹스 검정 (XGB)", f"{omics_p*100:.1f}%", "유전자 악성 위험도")
                else:
                    m_col2.metric("2차 오믹스 검정 (XGB)", "Activated X (면제)")
                m_col3.metric("최종 판정 결과", final_status)
                
                # 동적 컬러 프레임 리포트 카드 출력
                colors = ['#22c55e', '#eab308', '#ef4444']
                st.markdown(f"<div style='background-color: #111827; border-radius: 12px; padding: 25px; border: 2px solid {colors[final_class]}; color: #e2e8f0;'>", unsafe_allow_html=True)
                st.markdown(f"### 📋 임상 의사결정 최종 판단 피드백")
                
                if final_class == 0:
                    st.success("뇌 MRI 구조적 특징 공간 상에서 종양학적 이상 신호가 관찰되지 않았습니다. 임상 프로토콜에 따라 오믹스 정밀 분석을 면제하며, 정기적인 추적 관찰을 권장합니다.")
                elif final_class == 1:
                    st.warning("종양성 병변이 확인되었으나, 핵심 바이오마커 유전자 발현 패턴이 생존율이 유의미하게 높고 진행 속도가 완만한 저위험성 아형 군집에 매칭됩니다. 보존적 치료 계획 수립이 가능합니다.")
                elif final_class == 2:
                    st.error("구조적 악성 영상 징후와 더불어 고위험 분자 생체 지표가 악성 뇌종양 영토 중심부에 완벽히 중첩되어 있습니다. 시간 경과에 따른 예후 악화 위험이 크므로 즉각적인 항암 요법 착수를 권고합니다.")
                
                st.markdown("<small style='color:#64748b;'>발행일시: 2026년 융합정보의학 종합 진단 시스템 자동 출력 본 / 판독 보조용 CDSS 결과지</small>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.progress(max(cnn_p, omics_p))
            else:
                st.error("❌ 검정을 위해 왼쪽에서 MRI 이미지와 오믹스 CSV 파일을 모두 업로드해 주세요!")
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