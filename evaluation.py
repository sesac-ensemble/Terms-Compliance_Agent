from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from langchain_upstage import ChatUpstage
from langchain_core.tracers.context import tracing_v2_enabled
from prompts import ACTIVE_IMPROVEMENT_EVAL_PROMPT
from dotenv import load_dotenv
import re

load_dotenv()

llm = ChatUpstage(model="solar-pro2")

def evaluate_improvement_with_llm(original_clause: str, improvement_proposal: str, laws: str, reason: str, langsmith_project: str) -> dict:
    prompt = ACTIVE_IMPROVEMENT_EVAL_PROMPT.format(
        original_clause=original_clause,
        improvement_proposal=improvement_proposal,
        laws=laws,
        reason=reason
    )
    
    with tracing_v2_enabled(
        project_name=langsmith_project,
        tags=["improvement_evaluation", "llm_judge"]
    ):
        result = llm.invoke(prompt).content.strip()
    
    try:
        eval_result = json.loads(result)
        return eval_result
    except json.JSONDecodeError:
        return {
            "effectiveness": {"score": 0, "reason": "JSON 파싱 실패"},
            "adoptability": {"decision": "NO", "reason": "JSON 파싱 실패"}
        }

def evaluate_classification(langsmith_project: str, eval_file='batch_test_results.jsonl'):

    eval_data = []
    
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    
    df = pd.DataFrame(eval_data)
    df_labeled = df.dropna(subset=['label', 'predicted_fairness'])
    
    if len(df_labeled) == 0:
        print("오류: Ground truth 라벨 없음")
        return
    
    print(f"전체: {len(df)}개, 평가 가능: {len(df_labeled)}개\n")
    
    print("=" * 60)
    print("1단계: 공정/불공정 분류 평가")
    print("=" * 60)
    
    y_true_fairness = df_labeled['label'].map({'fair': '공정', 'unfair': '불공정'})
    y_pred_fairness = df_labeled['predicted_fairness']
    
    print(classification_report(y_true_fairness, y_pred_fairness))
    
    cm_fairness = confusion_matrix(y_true_fairness, y_pred_fairness, labels=['공정', '불공정'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_fairness, annot=True, fmt='d', cmap='Blues',
                xticklabels=['fair', 'unfair'], yticklabels=['fair', 'unfair'])
    plt.title(f'fair/unfair Confusion Matrix (n={len(df_labeled)})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('fairness_confusion_matrix.png', dpi=300)
    print("\n✓ 저장: fairness_confusion_matrix.png\n")
    
    
    print("=" * 60)
    print("2단계: 불공정 유형 분류 평가")
    print("=" * 60)
    
    df_unfair = df_labeled[df_labeled['label'] == 'unfair']
    
    if len(df_unfair) == 0:
        print("불공정 데이터 없음")
        return
    
    print(f"불공정 데이터: {len(df_unfair)}개\n")
    
    # 1. 유형 영문 매핑
    label_mapping = {
        "1. 서비스 일방적 변경·중단": "1. Unilateral Change/Discontinuation of Service",
        "2. 기한의 이익 상실": "2. Loss of Benefit of Term",
        "3. 고객 권리 제한": "3. Restriction of Customer Rights",
        "4. 통지·고지 부적절": "4. Improper Notification",
        "5. 계약 해지·변경 사유 포괄적": "5. Broad Reasons for Contract Termination/Change",
        "6. 비용 과다 부과·환급 제한": "6. Excessive Charges/Limited Refund",
        "7. 면책·책임 전가": "7. Exemption/Transfer of Liability",
        "8. 기타 불공정 약관": "8. Other Unfair Clauses"
    }

    # 2. predicted_type에서 괄호 안 내용 추출 함수
    def extract_type(pred_str):
        if pd.isna(pred_str):
            return None
        match = re.search(r'\((.+?)\)', str(pred_str))
        if match:
            return match.group(1)
        return pred_str

    # 3. 데이터 추출 및 전처리
    y_true_type = df_unfair['ground_truth_type']
    y_pred_type = df_unfair['predicted_type'].apply(extract_type)

    # 4. 영문 라벨 리스트 준비(고정 8종)
    labels_kr = [
        "1. 서비스 일방적 변경·중단",
        "2. 기한의 이익 상실",
        "3. 고객 권리 제한",
        "4. 통지·고지 부적절",
        "5. 계약 해지·변경 사유 포괄적",
        "6. 비용 과다 부과·환급 제한",
        "7. 면책·책임 전가",
        "8. 기타 불공정 약관"
    ]
    labels_en = [label_mapping[k] for k in labels_kr]

    # 5. 한국어→영어 변환
    y_true_type_en = [label_mapping.get(label, label) for label in y_true_type]
    y_pred_type_en = [label_mapping.get(label, label) for label in y_pred_type]

    # 6. 혼동행렬 계산 및 시각화
    cm_type = confusion_matrix(y_true_type_en, y_pred_type_en, labels=labels_en)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=labels_en, yticklabels=labels_en)
    plt.title(f'Unfair Type Confusion Matrix (n={len(df_unfair)})')
    plt.ylabel('Actual Type')
    plt.xlabel('Predicted Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('unfair_type_confusion_matrix.png', dpi=300)
    print("\n✓ Saved: unfair_type_confusion_matrix.png\n")
    
    print("=" * 60)
    print("3단계: 개선안 실효성 평가 (LLM Judge)")
    print("=" * 60)
    
    # 실제로 개선안이 생성된 불공정 조항만 평가 대상으로 추출
    df_need_eval = df_unfair[
        (df_unfair['predicted_improvement'].notna()) & 
        (df_unfair['predicted_improvement'] != "This clause is considered fair and does not require improvement.")   # ok
    ].copy()
    
    if len(df_need_eval) == 0:
        print("개선안 평가 대상 없음")
    else:
        print(f"개선안 평가 대상: {len(df_need_eval)}개")
        print("LLM으로 개선안 평가 중...\n")
        
        effectiveness_scores = []
        adoptability_decisions = []
        
        # 각 조항을 순회하면 evaluate_improvement_with_llm 함수 호출
        for idx, row in df_need_eval.iterrows():
            laws_text = "\n".join(row.get('retrieved_laws', []))
            eval_result = evaluate_improvement_with_llm(
                original_clause=row['cleaned_text'],
                improvement_proposal=row['predicted_improvement'],
                laws=laws_text,
                reason=row['predicted_type'],
                langsmith_project=langsmith_project
            )
            
            effectiveness_scores.append(eval_result.get('effectiveness', {}).get('score', 0))
            adoptability_decisions.append(eval_result.get('adoptability', {}).get('decision', 'NO'))
        
        if effectiveness_scores:
            print(f"개선안 실효성 평균 점수: {np.mean(effectiveness_scores):.2f}/10")
            print(f"개선안 실효성 중앙값: {np.median(effectiveness_scores):.2f}/10")
            print(f"개선안 실효성 표준편차: {np.std(effectiveness_scores):.2f}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(effectiveness_scores, bins=11, range=(0, 10), edgecolor='black', alpha=0.7)
            plt.xlabel('Effectiveness Score')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Improvement Effectiveness Scores (n={len(effectiveness_scores)})')
            # plt.xlabel('실효성 점수')
            # plt.ylabel('빈도')
            # plt.title(f'개선안 실효성 점수 분포 (n={len(effectiveness_scores)})')
            plt.tight_layout()
            plt.savefig('improvement_effectiveness_distribution.png', dpi=300)
            print("\n✓ 저장: improvement_effectiveness_distribution.png\n")
        
        if adoptability_decisions:
            yes_count = adoptability_decisions.count('YES')
            no_count = adoptability_decisions.count('NO')
            yes_ratio = yes_count / len(adoptability_decisions) * 100
            
            print(f"\n실무 채택 가능성:")
            print(f"  YES: {yes_count}개 ({yes_ratio:.1f}%)")
            print(f"  NO: {no_count}개 ({100-yes_ratio:.1f}%)")
            
            plt.figure(figsize=(8, 6))
            plt.bar(['YES', 'NO'], [yes_count, no_count], color=['green', 'red'], alpha=0.7)
            # plt.xlabel('채택 가능성')
            # plt.ylabel('빈도')
            # plt.title(f'실무 채택 가능성 분포 (n={len(adoptability_decisions)})')
            plt.xlabel('Adoptability')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Adoptability Decisions (n={len(adoptability_decisions)})')
   
            plt.tight_layout()
            plt.savefig('improvement_adoptability_distribution.png', dpi=300)
            print("\n✓ 저장: improvement_adoptability_distribution.png\n")
    
    print("=" * 60)
    print("4단계: RAG 품질 평가")
    print("=" * 60)
    
    df_with_retrieval = df_unfair[
        (df_unfair['retrieved_cases_count'].notna()) & 
        (df_unfair['retrieved_laws_count'].notna())
    ]
    
    if len(df_with_retrieval) == 0:
        print("RAG 검색 데이터 없음")
    else:
        avg_cases = df_with_retrieval['retrieved_cases_count'].mean()
        avg_laws = df_with_retrieval['retrieved_laws_count'].mean()
        
        print(f"평균 유사사례 검색 수: {avg_cases:.2f}")
        print(f"평균 관련 법령 검색 수: {avg_laws:.2f}")
        
        zero_cases = (df_with_retrieval['retrieved_cases_count'] == 0).sum()
        zero_laws = (df_with_retrieval['retrieved_laws_count'] == 0).sum()
        
        print(f"\n유사사례 미검색 건수: {zero_cases}개 ({zero_cases/len(df_with_retrieval)*100:.1f}%)")
        print(f"관련 법령 미검색 건수: {zero_laws}개 ({zero_laws/len(df_with_retrieval)*100:.1f}%)")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].hist(df_with_retrieval['retrieved_cases_count'], bins=10, edgecolor='black', alpha=0.7, color='skyblue')
    
    # 한글  ----------------------
        # axes[0].set_xlabel('검색된 유사사례 수')
        # axes[0].set_ylabel('빈도')
        # axes[0].set_title('유사사례 검색 분포')
        
        # axes[1].hist(df_with_retrieval['retrieved_laws_count'], bins=10, edgecolor='black', alpha=0.7, color='salmon')
        # axes[1].set_xlabel('검색된 관련 법령 수')
        # axes[1].set_ylabel('빈도')
        # axes[1].set_title('관련 법령 검색 분포')


        axes[0].set_xlabel('Number of Retrieved Similar Cases')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Retrieved Similar Cases')

        axes[1].hist(df_with_retrieval['retrieved_laws_count'], bins=10, edgecolor='black', alpha=0.7, color='salmon')
        axes[1].set_xlabel('Number of Retrieved Relevant Laws')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Retrieved Relevant Laws')
                
        plt.tight_layout()
        plt.savefig('rag_retrieval_distribution.png', dpi=300)
        print("\n✓ 저장: rag_retrieval_distribution.png\n")

if __name__ == "__main__":
    import sys
    project_name = sys.argv[1] if len(sys.argv) > 1 else "contract-review-eval"
    evaluate_classification(langsmith_project=project_name, eval_file='batch_test_results.jsonl')

# python evaluation.py evaluation