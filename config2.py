# config.py
"""설정 파일"""

# 벡터 검색 설정
SIMILARITY_THRESHOLD = 0.5  # 유사도 임계점 (0.0 ~ 1.0)
SEARCH_TOP_K_CASES = 10     # 초기 검색 개수 (필터링 전)
SEARCH_TOP_K_LAWS = 10
MAX_DISPLAY_CASES = 5       # 최종 표시 개수
MAX_DISPLAY_LAWS = 5

# 워크플로우 설정
MAX_ITERATIONS = 3

# 모델 설정
EMBEDDING_MODEL = "solar-embedding-1-large-passage"
LLM_MODEL = "solar-pro2"

# UI 설정
SHOW_SIMILARITY_SCORES = True   # 유사도 점수 표시 여부
SHOW_RETRIEVED_CASES = True     # 검색된 사례 표시 여부