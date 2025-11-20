import streamlit as st
import graphviz

def draw_user_guide():
    st.header("사용 가이드 및 지원 범위")
    st.markdown("#### AI와 함께하는 약관 심사, 이렇게 진행하세요.")
    
    st.write("") # 여백

    # 1. 탭을 사용하여 정보 구조화
    tab1, tab2 = st.tabs(["이용 절차 (Workflow)", "지원 범위"])
    
    # --- [탭 1] 이용 절차 (시각화 중심) ---
    with tab1:
        st.markdown("#### 1. 업무 흐름도")
        st.caption("입력부터 최종 확정까지 4단계로 진행됩니다.")

        # Graphviz를 이용한 깔끔한 프로세스 다이어그램
        graph = graphviz.Digraph()
        graph.attr(rankdir='LR', size='12,4') # 가로 방향
        graph.attr('node', shape='rect', style='filled', 
                   fillcolor='#f0f2f6', fontname='Malgun Gothic', 
                   fontsize='14', margin='0.2')
        graph.attr('edge', fontsize='12')

        # 노드 정의
        graph.node('Input', '1. 조항 입력\n(분석 요청)', fillcolor='#e3f2fd') # 강조색
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
        
        st.write("") # 여백
        
        st.markdown("#### 2. 효과적인 사용을 위한 3단계 가이드")
        
        step1, step2, step3 = st.columns(3)
        
        with step1:
            with st.container(border=True):
                st.markdown("### 1️⃣ 입력")
                st.markdown("**'한 번에 하나씩'**")
                st.caption("""
                약관 전체를 붙여넣기보다,
                **분석이 필요한 '개별 조항' 단위**로
                입력해야 AI가 가장 정확하게 분석합니다.
                
                약관 전체를 넣고 싶다면, `PDF(전체 문서 분석)` 기능을 이용해주세요.
                """)
        
        with step2:
            with st.container(border=True):
                st.markdown("### 2️⃣ 검토")
                st.markdown("**'빨간 펜 선생님처럼'**")
                st.caption("""
                AI가 제안한 개선안에서
                **변경된 문구(Redline) 위주**로 확인하세요.
                법적 뉘앙스가 유지되었는지 체크합니다.
                """)
                st.write("")
                st.write("")
                st.write("")
                
        with step3:
            with st.container(border=True):
                st.markdown("### 3️⃣ 피드백")
                st.markdown("**'동료에게 말하듯이'**")
                st.caption("""
                결과가 아쉽다면 **구체적으로 명령**하세요.
                *"~부분은 빼줘", "좀 더 강한 어조로"* 처럼
                대화하듯 요청하면 즉시 수정합니다.
                """)
                st.write("")
                st.write("")
        
        st.divider()

        # 꿀팁 섹션
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 **수정 요청(Feedback) 꿀팁**")
            st.markdown("""
            AI의 제안이 마음에 들지 않으면 채팅하듯 편하게 명령하세요.
            - *"2, 3번 참고 법령과 1번 유사 사례의 시정 취지를 반영하여, 입력된 조항의 불공정성을 해소하는 방향으로 다시 작성해줘"*
            - *"위약금을 5%로 낮춰줘"*
            - *"문구를 좀 더 부드럽게 다듬어줘"*
            - *"법적 근거를 더 보강해줘"*
            """)
            st.caption("법적 허용 범위 내에서 적용됩니다.")
        with col2:
            st.success("✨ **분석 완료 기준**")
            st.markdown("""
            - **[피드백]** 버튼을 눌러야 해당 조항 분석이 완료됩니다.
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
    pass

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
    pass