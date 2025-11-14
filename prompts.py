"""프롬프트 중앙 관리"""

FAIRNESS_PROMPT = """다음 약관 조항이 공정한지, 불공정한지 판단하세요:

{clause}

출력 형식: "공정" 또는 "불공정" 중 하나만 출력하세요.
중요: 반드시 "공정" 또는 "불공정" 단어만 출력하고, 절대로 부연 설명이나 이유를 추가하지 마세요."""



UNFAIR_TYPE_PROMPT = """다음 약관 조항의 불공정 유형을 판단하세요:

{clause}

유형:
1. 서비스 일방적 변경·중단
2. 기한의 이익 상실
3. 고객 권리 제한
4. 통지·고지 부적절
5. 계약 해지·변경 사유 포괄적
6. 비용 과다 부과·환급 제한
7. 면책·책임 전가
8. 기타 불공정 약관

**출력 규칙:**
- 가장 주된 불공정 유형 1개만 선택
- "불공정(숫자. 유형명)" 형식으로만 출력
- 판단 근거, 설명, 기타 텍스트 절대 포함 금지

출력 예시: 불공정(2. 기한의 이익 상실)"""


IMPROVEMENT_PROMPT_V1 = """당신은 법률 전문가입니다.

[원본 약관 조항]
{original_clause}

[불공정 유형]
{unfair_type}

[관련 시정 사례]
{related_cases}

[관련 법령]
{related_laws}

{feedback_context}

[작업]
위 정보를 바탕으로 이 약관 조항을 공정한 약관으로 개선하세요.

[중요 규칙]
- 법 근거는 위의 "관련 시정 사례 및 법령"에 명시된 것만 사용하세요.
- 근거 없는 법령이나 조항, 특정 기간을 포함하지 마세요.

[출력 형식]
1. 개선된 약관 조항
2. 개선 사유
3. 핵심 변경 사항"""

GEVAL_PROMPT = """당신은 금융 분야 약관 심사 전문가입니다.

[평가 대상]
원본: {original_clause}
유형: {unfair_type}
법령: {predicted_laws}
사례: {similar_cases}
개선안: {improvement_proposal}

[정답]
법령: {expected_laws}
사례: {reference_case}

[평가 항목 - 각 10점]

1. 법령 선택 적절성
2. 사례 활용도
3. 개선안 구체성

[출력]
법령 적절성: [점수]/10
사례 활용도: [점수]/10
개선안 구체성: [점수]/10
총점: [점수]/30

평가 근거: [각 항목별 2-3문장]"""



IMPROVEMENT_EVAL_PROMPT = """당신은 약관 심사 실무 담당자입니다.

[평가 대상]
원본 약관: {original_clause}
개선안: {improvement_proposal}
적용 법령: {laws}
개선 사유: {reason}

[평가 항목]
다음 2가지 항목을 평가하세요.

1. 개선안 실효성 (10점 만점)
- 평가 기준: 이 개선안이 불공정성을 실제로 해소하는가?
- 10점: 불공정성 완전 해소, 법령 요건 완벽 충족
- 7-9점: 불공정성 대부분 해소, 일부 보완 필요
- 4-6점: 부분적 해소, 핵심 문제 일부 미해결
- 1-3점: 형식적 수정, 실질적 개선 미흡
- 0점: 불공정성 해소 안 됨

2. 실무 채택 가능성 (YES/NO)
- 평가 기준: 수정 없이 바로 사용 가능한가?
- YES 조건 (모두 충족):
  * 법령 근거가 명확히 제시됨
  * 구체적인 기한/절차/방법이 명시됨
  * 문구가 명확하고 모호하지 않음
  * 실무에서 즉시 적용 가능
- NO 조건 (하나라도 해당):
  * 법령 근거 불명확
  * 추상적이거나 모호한 표현
  * 실무 적용을 위해 추가 수정 필요

[출력 형식]
반드시 아래 JSON 형식으로만 답변하세요.
{{
  "effectiveness": {{
    "score": 0-10 사이 정수,
    "reason": "100자 이내 평가 근거"
  }},
  "adoptability": {{
    "decision": "YES" 또는 "NO",
    "reason": "100자 이내 판단 근거"
  }}
}}
"""

# 현재 사용 프롬프트
ACTIVE_FAIRNESS_PROMPT = FAIRNESS_PROMPT
ACTIVE_UNFAIR_TYPE_PROMPT = UNFAIR_TYPE_PROMPT
ACTIVE_IMPROVEMENT_PROMPT = IMPROVEMENT_PROMPT_V1
ACTIVE_GEVAL_PROMPT = GEVAL_PROMPT
ACTIVE_IMPROVEMENT_EVAL_PROMPT = IMPROVEMENT_EVAL_PROMPT