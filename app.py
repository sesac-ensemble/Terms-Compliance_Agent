import streamlit as st
import traceback
from datetime import datetime
import graphviz
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
# from langchain_core.tracers.context import tracing_v2_enabled
# 랭채인 트래킹 설정 끄기 -> .env 파일에서 LANGCHAIN_TRACING_V2=false

# 인증 관리자
import auth_manager 

# 모듈화된 설정, 그래프, PDF 모듈 로드
from config2 import SIMILARITY_THRESHOLD, MAX_ITERATIONS, SHOW_RETRIEVED_CASES
from langgraph_components import load_app_safe
from langgraph.types import Command
from ui_modules import run_pdf_batch_mode

def run_chatbot_mode(app, current_threshold_value):
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "hitl_pending" not in st.session_state:
        st.session_state.hitl_pending = False
    if "current_state" not in st.session_state:
        st.session_state.current_state = {}
    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = None
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": """### 안녕하세요, 법률 약관 검토 챗봇입니다👋\n
새로운 약관 조항의 공정성 검토를 도와드리겠습니다. 분석을 원하는 **약관 조항**만 아래 채팅창에 입력해 주세요.
            
        [입력 예시]
        회원이 본 카드의 발급 목적과 다르게 이용한다고 카드사가 판단하거나, 
        기타 이에 준하는 중대한 사유가 발생하여 계약 유지가 곤란하다고 인정되는 경우, 카드사는 본 계약을 해지할 수 있습니다.
<- 더 궁금한 점이 있으시다면, 왼쪽 사이드바의 `도움말`을 확인하세요.
        """
        })

    # 1. 채팅 메시지 기록을 먼저 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 2. RAG 결과(유사 사례)가 state에 존재할 경우, expander를 출력
    # (RAG 실행 전에는 'cases'가 None이므로 이 블록은 건너뜀)
    cases = st.session_state.current_state.get('retrieved_cases_metadata', None)
    
    if SHOW_RETRIEVED_CASES and cases is not None:
        used_threshold = st.session_state.current_state.get('similarity_threshold', SIMILARITY_THRESHOLD)
        
        with st.expander("참고한 유사 사례 보기", expanded=False):
            if cases:
                st.caption(f"총 {len(cases)}개 사례 (유사도 {used_threshold:.0%} 이상)")
                
                for case in cases:
                    similarity = case['similarity']
                    
                    if similarity >= 0.7:
                        color = "🟢"
                    elif similarity >= 0.5:
                        color = "🟡"
                    else:
                        color = "🟠"
                    
                    st.markdown(f"### {color} 사례 {case['index']} - 유사도: {similarity:.1%}")
                    st.caption(f"📅 {case['date']} | 유형: {case['case_type']}")
                    
                    with st.container():
                        st.markdown("**불공정 약관 조항:**")
                        st.info(case['content'].split('결론:')[0].replace('약관: ', '').strip())
                        
                        if case['explanation']:
                            st.markdown("**시정 요청 사유:**")
                            st.warning(case['explanation'])
                            
                        if case['conclusion']:
                            st.markdown("**최종 결론:**")
                            st.success(case['conclusion'])
                        
                        if case['related_law']:
                            st.caption(f"🔗 관련법: {case['related_law']}")
                    
                    st.divider()
            else:
                st.warning("검색된 사례가 없습니다.")
    
    # 3. 피드백 대기 상태(hitl_pending)인 경우, 피드백 UI 출력
    if st.session_state.hitl_pending:
        current_iteration = st.session_state.current_state.get('iteration', 1)
        
        # --- [UI 상태 관리 변수 초기화] ---
        if "show_modify_input" not in st.session_state:
            st.session_state.show_modify_input = False

        st.info(f"개선안 (반복 {current_iteration}/{MAX_ITERATIONS})에 대한 피드백을 주세요.")

        # ============================================================
        # [화면 A] 기본 버튼 선택 화면 (입력창 숨김 상태)
        # ============================================================
        if not st.session_state.show_modify_input:
            col1, col2, col3 = st.columns(3)
            
            # 1. 수락 버튼
            with col1:
                if st.button("현재 개선안 수락 (Approve)", use_container_width=True, type="primary"):
                    st.session_state.pending_feedback = {
                        "user_feedback": "approved",
                        "modify_reason": "",
                        "retry_action": ""
                    }
                    st.session_state.hitl_pending = False
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "[피드백] 개선안을 수락합니다 (완료)."
                    })
                    st.rerun()

            # 2. 수정/재생성 버튼 (누르면 입력창 열림)
            with col2:
                if st.button("다른 개선안 생성 (Modify)", use_container_width=True):
                    st.session_state.show_modify_input = True  # 상태 변경
                    st.rerun()

            # 3. 폐기 버튼
            with col3:
                if st.button("현재 개선안 폐기 (Discard)", use_container_width=True):
                    st.session_state.pending_feedback = {
                        "user_feedback": "rejected",
                        "retry_action": "discard",
                        "modify_reason": ""
                    }
                    st.session_state.hitl_pending = False
                    st.session_state.messages.append({
                        "role": "user", 
                        "content": "[피드백] 거절 (검토 폐기)."
                    })
                    st.rerun()

        # ============================================================
        # [화면 B] 수정 사유 입력 화면 (버튼 누른 후)
        # ============================================================
        else:
            st.markdown("### 📝 수정 요청 사항 입력")
            st.caption("구체적으로 적어주실수록 더 정확한 개선안이 나옵니다.")
            
            # 반복 횟수 제한 체크
            if current_iteration >= MAX_ITERATIONS:
                st.warning(f"⚠️ 반복 횟수 제한({MAX_ITERATIONS}회)에 도달하여 더 이상 수정할 수 없습니다.")
                if st.button("돌아가기", use_container_width=True):
                    st.session_state.show_modify_input = False
                    st.rerun()
            else:
                modify_reason = st.text_area(
                    "수정 요청 사유:", 
                    key="modify_reason_input",
                    height=150,
                    placeholder="예) 위약금 비율을 조금 더 낮춰줘\n예) 해지 사유를 더 구체적으로 명시해줘"
                )

                b_col1, b_col2 = st.columns([1, 1])
                
                with b_col1:
                    if st.button("취소 (이전으로)", use_container_width=True):
                        st.session_state.show_modify_input = False
                        st.rerun()
                        
                with b_col2:
                    if st.button("제출하기", type="primary", use_container_width=True):
                        if not modify_reason.strip():
                            st.error("수정 요청 사유를 입력해주세요.")
                        else:
                            # 제출 로직
                            st.session_state.pending_feedback = {
                                "user_feedback": "modify",
                                "modify_reason": modify_reason.strip(),
                                "retry_action": ""
                            }
                            st.session_state.hitl_pending = False
                            st.session_state.show_modify_input = False # 상태 초기화
                            
                            st.session_state.messages.append({
                                "role": "user", 
                                "content": f"[피드백] 수정 요청:\n{modify_reason.strip()}"
                            })
                            st.rerun()
                            
        st.chat_input("피드백을 먼저 완료해주세요.", disabled=True)

    # 4. 피드백 대기 상태가 아닌 경우, 채팅 입력창 활성화
    else:
        # 4-1. 보류 중인 피드백이 있다면 먼저 처리
        if st.session_state.pending_feedback is not None:
            feedback_input = st.session_state.pending_feedback
            st.session_state.pending_feedback = None
            
            # 피드백 입력(invoke) 시, 현재 사이드바의 임계값을 다시 주입(overwrite)합니다.
            feedback_input["similarity_threshold"] = current_threshold_value  # 10/16 추가
            
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            with st.chat_message("assistant"):
                with st.spinner("피드백을 반영하여 처리 중..."):
                    try:
                        output = app.invoke(
                            Command(resume=feedback_input), 
                            config=config
                        )
                        st.session_state.current_state = output
                        
                        # output(결과)이 아닌 feedback_input(입력 의도)을 확인합니다.
                        # output 상태값이 유실되더라도, 사용자가 누른 버튼 정보는 확실하기 때문
                        sent_feedback = feedback_input.get('user_feedback', '')
                        sent_retry = feedback_input.get('retry_action', '')

                        if sent_feedback == "approved" or (sent_feedback == "rejected" and sent_retry == "discard"):
                            st.markdown("### 검토 완료\n검토가 최종 완료되었습니다.")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "검토가 완료되었습니다."
                            })
                            # 상태 초기화 (중복 실행 방지)
                            st.session_state.hitl_pending = False
                            st.session_state.thread_id = None
                            st.rerun()
                        else:
                            st.markdown(f"### 🔄 새로운 개선안 (반복 {output.get('iteration', '?')}/{MAX_ITERATIONS})")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"피드백 처리 중 오류 발생: {e}")
                        print(f"Error details: {traceback.format_exc()}") # 자세한 에러 로그 출력
                        st.session_state.hitl_pending = False
                        st.session_state.thread_id = None

        # 4-2. 새 프롬프트(쿼리)를 받음
        elif prompt := st.chat_input("검토할 약관 조항을 입력하세요..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("약관 조항을 분석 중입니다..."):
                    try:
                        st.session_state.thread_id = f"session_{datetime.now().timestamp()}"
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        
                        # log를 위한 로그인 정보 가져오기 (없으면 unknown 처리)
                        user_email = st.session_state.get("username", "unknown")
                        user_name = st.session_state.get("name", "unknown")
                        
                        initial_state = {
                            "clause": prompt,
                            "iteration": 1,
                            "session_id": st.session_state.thread_id,
                            "validation_failed": False,
                            "retrieved_cases_metadata": [],
                            "retrieved_laws_metadata": [],
                            "similarity_threshold": current_threshold_value,
                            "user_email": user_email,
                            "user_name": user_name
                        }
                        
                        # with tracing_v2_enabled():
                        output = app.invoke(initial_state, config=config)
                        
                        if output.get('validation_failed', False):
                            error_msg = f"입력 오류: {output.get('cleaned_text', '알 수 없는 오류')}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            st.session_state.thread_id = None
                        # --- 수정 11/15---
                        # '공정'일 때와 '불공정'일 때를 분리
                        elif output.get('fairness_label') == "공정":
                            st.session_state.current_state = output
                            # '공정'일 경우 (generate_fair_report_node 경유)
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            # '공정'이므로 피드백 대기(HITL) 없이 완료
                            st.session_state.hitl_pending = False 
                            st.session_state.thread_id = None # 세션 종료
                            st.rerun()
                        else:
                            st.session_state.current_state = output
                            # '불공정'일 경우 (generate_proposal_node 경유)
                            st.markdown("### 제안 (첫 번째 개선안)")
                            st.markdown(output['improvement_proposal'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": output['improvement_proposal']
                            })
                            # '불공정'이므로 피드백 대기(HITL)
                            st.session_state.hitl_pending = True
                            st.rerun()

                    except Exception as e:
                        st.error(f"약관 검토 중 오류 발생: {e}")
                        st.exception(traceback.format_exc())
                        st.session_state.thread_id = None
                        st.session_state.hitl_pending = False

def draw_user_guide():
    st.title("사용 가이드 및 지원 범위")
    st.markdown("#### AI와 함께하는 약관 심사, 이렇게 진행하세요.")
    
    st.write("") # 여백

    # 1. 탭을 사용하여 정보 구조화
    tab1, tab2 = st.tabs(["이용 절차 (Workflow)", "지원 범위"])
    
    # --- [탭 1] 이용 절차 (시각화 중심) ---
    with tab1:
        st.subheader("1. 업무 흐름도")
        st.caption("입력부터 최종 확정까지 4단계로 진행됩니다.")

        # Graphviz를 이용한 깔끔한 프로세스 다이어그램
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', size='12,4') # 가로 방향
        graph.attr('node', shape='rect', style='filled', 
                   fillcolor='#f0f2f6', fontname='Malgun Gothic', 
                   fontsize='14', margin='0.2')
        graph.attr('edge', fontsize='12')

        # 노드 정의
        graph.node('Input', '1. 조항 입력\n(검토 요청)', fillcolor='#e3f2fd') # 강조색
        graph.node('AI', '2. AI 정밀 진단\n(불공정/유사사례)')
        graph.node('Draft', '3. 개선안 제안\n(Redline)')
        graph.node('Human', '4. 전문가 검토\n(수락/수정/폐기)', fillcolor='#fff3e0') # 강조색

        # 엣지(화살표) 연결
        graph.edge('Input', 'AI')
        graph.edge('AI', 'Draft')
        graph.edge('Draft', 'Human', label=' 제안 확인')
        graph.edge('Human', 'Draft', label=' 💬 수정 요청\n(Feedback)', style='dashed', color='red')
        graph.edge('Human', 'Human', label=' ✅ 최종 확정', color='green')

        st.graphviz_chart(graph)
        
        st.divider()

        # 꿀팁 섹션
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **수정 요청(Feedback) 꿀팁**")
            st.markdown("""
            AI의 제안이 마음에 들지 않으면 채팅하듯 편하게 명령하세요.
            - *"위약금을 5%로 낮춰줘"*
            - *"문구를 좀 더 부드럽게 다듬어줘"*
            - *"법적 근거를 더 보강해줘"*
            """)
            st.caption("법적 근거를 해치지 않는 선에서 적용됩니다.")
        with col2:
            st.success("✨ **검토 완료 기준**")
            st.markdown("""
            - **[피드백]** 버튼을 눌러야 해당 조항 검토가 완료됩니다.
            - 완료된 내용은 자동으로 로그에 저장됩니다.
            """)

    # --- [탭 2] 분석 범위 (카드 UI 중심) ---
    with tab2:
        st.subheader("AI가 할 수 있는 것 vs 없는 것")
        st.markdown("본 시스템은 **'개별 조항(Clause)' 단위의 공정성 심사**에 특화되어 있습니다.")
        
        st.write("") 

        # 3단 컬럼으로 O/X 명확히 구분
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.container(border=True)
            st.success("✅ 문장/표현 레벨")
            st.markdown("""
            **[지원 기능]**
            - **오타 및 비문** 교정
            - **모호한 표현** 감지
            - **독소 조항** 키워드 식별
            """)
            
        with c2:
            st.container(border=True)
            st.info("✅ 법적 효력 레벨")
            st.markdown("""
            **[핵심 기능]**
            - **불공정 유형(8개)** 판별
            - **관련 법령** 매칭
            - **유사 심결례** 비교 검색
            """)
            
        with c3:
            st.container(border=True)
            st.warning("⚠️ 문서 구조 레벨")
            st.markdown("""
            **[지원 제한]**
            - **문서 서식/디자인**
            """)
            
        st.divider()
        st.caption("※ AI의 분석 결과는 법적 효력을 갖지 않으며, 최종 판단 책임은 담당자에게 있습니다.")

def draw_analysis_scope():
    st.header("데이터 구조")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.subheader("참고하는 데이터-링크 추가")
            st.caption("최신 데이터를 참고하였습니다.")
            st.markdown("""
            - 약관법
            - 약관심사지침
            - 금융소비자법 시행령
            - 금융소비자 보호에 관한 감독 규정
            - 공정위에서 발표한 불공정 약관 시정 사례
            """)
        
    with col2:
        with st.container(border=True):
            st.subheader("데이터 구조")
            st.markdown("""
            
            """)
        
    with col3:
        with st.container(border=True):
            st.subheader("데이터 어쩌구")
            st.markdown("""
            
            """)

    st.divider()
    
    st.header("판단 기준 보기")
    
    col4, col5 = st.columns(2)
    
    with col4: 
        st.subheader("법령 검색 방법")
        st.markdown("""
        1. 사용자의 질문과 가상 유사한 법령 5개 검색
        2. 유사한 사례 5개 중, 유사도가 50% 이하는 삭제
                    """)
        st.markdown("""
        | 우선순위 | 법령명 | 메모 |
        | :--- | :--- | :--- |
        | **1** | 약관의 규제에 관한 법률 | 특별법, 최우선 적용 |
        | **2** | 전자금융거래법 | 분야별 개별법 |
        | **3** | 금융소비자보호에 관한 법률 | 일반법 |
        | **4** | 금융소비자법 시행령 | 대통령령 (하위 규범) |
        | **5** | 감독규정 | 금융감독원 내부 규정, 행정규칙 |
        | **6** | 약관심사지침 | 법률 아님, 법적 효력 없음 (행정해석) |
        """)
        st.caption("감독 규정을 쪼갬, 쪼갠 이유:")
        
        st.subheader("사례 검색 방법")
        st.markdown("""
        1. 사용자의 질문과 가상 유사한 사례 5개 검색
        2. 유사한 사례 5개 중, 유사도가 50% 이하는 삭제
        3. 남은 사례 중, 가장 최신 사례 1개만 선택
                    """)

    with col5:
        st.subheader("개선안 생성 방법")
        st.markdown("""
        1. 검색된 사례 + 법령 + 사용자의 질문을 반영하여 개선안 생성
                    """)
        
@st.cache_resource
def get_cached_app():
    """
    LangGraph 앱과 VectorStore를 캐싱하여
    Streamlit이 Rerun 되어도 메모리(Checkpoint)가 유지되도록 합니다.
    """
    return load_app_safe()

def main_chatbot_ui():
    st.set_page_config(page_title="약관 검토 챗봇", layout="wide")
    
    # --- [1] 상태 초기화 ---
    if "show_guide" not in st.session_state:
        st.session_state.show_guide = False
    if "show_scope" not in st.session_state:
        st.session_state.show_scope = False

    # 현재 '도움말 모드'인지 확인 (가이드나 범위 화면 중 하나라도 켜져 있으면 True)
    is_help_mode = st.session_state.show_guide or st.session_state.show_scope
    
    # ---------------------------------------------------------
    # [사이드바 영역]
    # ---------------------------------------------------------
    with st.sidebar:
       # 1. 검색 설정 (가이드나 범위 화면이 아닐 때만 활성화)
        disabled_status = st.session_state.show_guide or st.session_state.show_scope
        
        st.subheader("검색 옵션")
        similarity_threshold_percent = st.slider(
            "유사도 임계값 (%)",
            min_value=0,
            max_value=100,
            value=int(SIMILARITY_THRESHOLD * 100),
            step=5,
            format="%d%%",
            disabled=is_help_mode
        )
        current_threshold_value = similarity_threshold_percent / 100.0
        
        if not is_help_mode:
            st.caption(f"현재 설정: {similarity_threshold_percent}% 이상 유사 사례 검색")
        
        st.divider()
            
        st.header("도움말")
        
        # 2. 화면 전환 버튼 로직 (가이드 보기 / 분석 범위 / 돌아가기)
        # 2-1. 가이드 버튼 (보고 있으면 '닫기', 안 보고 있으면 '열기')
        if st.session_state.show_guide:
            # 현재 가이드를 보고 있는 상태 -> '돌아가기' 버튼으로 표시
            if st.button("**⬅️ 돌아가기**", use_container_width=True):
                st.session_state.show_guide = False
                st.rerun()
        else:
            # 가이드를 안 보고 있는 상태 -> '가이드 보기' 버튼으로 표시
            if st.button("사용 가이드 보기", use_container_width=True):
                st.session_state.show_guide = True
                st.session_state.show_scope = False # 다른 창은 닫음
                st.rerun()

        # 2-2. 분석 범위 버튼 (보고 있으면 '닫기', 안 보고 있으면 '열기')
        if st.session_state.show_scope:
            # 현재 분석 범위를 보고 있는 상태 -> '돌아가기' 버튼으로 표시
            if st.button("**⬅️ 돌아가기**", use_container_width=True, key="btn_close_scope"):
                st.session_state.show_scope = False
                st.rerun()
        else:
            # 분석 범위를 안 보고 있는 상태 -> '범위 보기' 버튼으로 표시
            if st.button("데이터 구조 / 판단 기준 보기", use_container_width=True):
                st.session_state.show_scope = True
                st.session_state.show_guide = False # 다른 창은 닫음
                st.rerun()
    
        st.caption("2025.11 약관 분석 모듈 v1.0")

    # ---------------------------------------------------------
    # [메인 화면 영역]
    # ---------------------------------------------------------
    
    # [A] 가이드 보기 모드일 때 -> 가이드 함수 호출
    if st.session_state.show_guide:
        draw_user_guide()
    
    # [B] 분석 범위 보기 모드
    elif st.session_state.show_scope:
        draw_analysis_scope()
    
    # [C] 검토 모드일 때 -> 기존 탭(Radio) 화면 표시
    else:
        st.title("약관 검토 챗봇")
        st.caption("본 서비스는 법무팀의 신규 약관 작성을 지원하는 내부용 도구입니다. AI 분석은 법적 해석을 대체하지 않으며, 최종 검토·판단 책임은 법무팀 담당자에게 있습니다.")

        
        # 앱 로드
        app, vectorstore = load_app_safe()
        if not app or not vectorstore:
            st.error("앱 초기화 실패")
            return

        # --- 기존의 Radio 탭 유지 ---
        tab_options = ["💬 챗봇 (단일 조항 검토)", "📄 PDF (전체 문서 검토)"]
        
        # 탭 상태 유지
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = tab_options[0]

        active_tab = st.radio(
            "모드 선택",
            tab_options,
            key="active_tab", # session_state와 자동 연동
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.divider()

        if active_tab == "💬 챗봇 (단일 조항 검토)":
            run_chatbot_mode(app, current_threshold_value)
            
        elif active_tab == "📄 PDF (전체 문서 검토)":
            run_pdf_batch_mode(app, vectorstore, current_threshold_value)
        

def main():
    # 1. 인증 관리자로부터 객체 가져오기
    authenticator = auth_manager.get_authenticator()

    # 2. 로그인 상태 확인 및 처리 (이 함수가 로그인 창 표시부터 검증까지 다 함)
    if auth_manager.check_login_status(authenticator):
        # 3. 로그인 성공 시 메인 UI 실행
        main_chatbot_ui()

if __name__ == "__main__":
    main()