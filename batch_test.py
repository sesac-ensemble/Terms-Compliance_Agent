import pandas as pd
pd.set_option('display.max_columns', None)  # 모든 컬럼 표시
import json
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langchain_core.tracers.context import tracing_v2_enabled
from config2 import *
from prompts import (
    ACTIVE_FAIRNESS_PROMPT,
    ACTIVE_UNFAIR_TYPE_PROMPT,
    ACTIVE_IMPROVEMENT_PROMPT
)
import re
from tqdm import tqdm

load_dotenv()

embeddings = UpstageEmbeddings(model=EMBEDDING_MODEL)
llm = ChatUpstage(model=LLM_MODEL)

def load_vectordb():
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="./chroma_db",
        collection_name="contract_laws",
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def clean_text(text: str) -> str:
    cleaned = re.sub(r'^[\s•\-\*]+', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]\s*', '', cleaned)
    cleaned = re.sub(r'\(\d+\)\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def prepare_full_clause(clause: str, context: str = None) -> str:
    if context and pd.notna(context) and context.strip():
        return f"[참조 조항]\n{context}\n\n[본 조항]\n{clause}"
    return clause

def safe_get_value(row, key, default="N/A"):
    val = row.get(key, None)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    if isinstance(val, str) and val.strip() == "":
        return default
    return val

def classify_fairness(clause: str, langsmith_project: str) -> str:
    prompt = ACTIVE_FAIRNESS_PROMPT.format(clause=clause)
    with tracing_v2_enabled(
        project_name=langsmith_project, # "contract-review-batch",
        tags=["fairness_classification", "batch_test"]
    ):
        result = llm.invoke(prompt).content.strip()
        #print(f"classify_fairness result:{result}")
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # 첫 줄만 추출
        result = result.split('\n')[0].strip()
        #print(f"22222 classify_fairness result:{result}")
        #print("22222222222222222222222222222222222222222!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # '공정' 또는 '불공정'만 추출
        if '공정' in result and '불공정' not in result:
            return '공정'
        elif '불공정' in result:
            return '불공정'
        
    return result


def classify_unfair_type(clause: str, fairness: str, langsmith_project: str) -> str:
    # fairness가 부연 설명 포함한 경우 첫 줄만 추출
    fairness_line = fairness.split('\n')[0].strip()
    
    # print("*************  1 fairness_line ")
    # print(f"fairness_line: {fairness_line}")
    # print("************* 1  fairness_line ")
    
    if '공정' in fairness_line and '불공정' not in fairness_line:
        return "N/A"
    
    prompt = ACTIVE_UNFAIR_TYPE_PROMPT.format(clause=clause)
    
    with tracing_v2_enabled(
        project_name=langsmith_project,
        tags=["unfair_type_classification", "batch_test"]
    ):
        result = llm.invoke(prompt).content.strip()
        
        # print("2222222222222222")
        # print(f"result: {result}")
        # print("2222222222222222")
        
        # 여러 유형이 출력된 경우 첫 번째만 추출
        # 패턴: "불공정(숫자. 유형명)"을 찾아서 첫 번째만 반환
        import re
        pattern = r'불공정\(\d+\.\s*[^)]+\)'
        matches = re.findall(pattern, result)
        
        if matches:
            print("33333333333333333333333")
            print(f"matches[0]: {matches[0]}")
            print("33333333333333333333333")
            return matches[0]  # 첫 번째 유형만 반환
        
        # 패턴이 없으면 첫 줄만 반환
        
        # print("44444444444444")
        # print(result.split('\n')[0].strip())
        # # print(f"result.split('\n')[0].strip(): {result.split('\n')[0].strip()}")
        # print("44444444444444444")
        return result.split('\n')[0].strip()


def retrieve_cases_and_laws(vectorstore, clause: str, unfair_type: str, threshold: float = SIMILARITY_THRESHOLD):
    if unfair_type == "N/A":
        return [], []
    search_query = f"{unfair_type} {clause}"
    results_cases = vectorstore.similarity_search_with_relevance_scores(
        search_query, k=SEARCH_TOP_K_CASES, filter={"source_type": "case"}
    )
    filtered_cases = [
        {"similarity": score, "content": doc.page_content, "metadata": doc.metadata}
        for doc, score in results_cases if score >= threshold
    ][:MAX_DISPLAY_CASES]
    law_query = " ".join([c['metadata'].get('related_law', '') for c in filtered_cases if c['metadata'].get('related_law')])
    if not law_query:
        law_query = search_query
    results_laws = vectorstore.similarity_search_with_relevance_scores(
        law_query, k=SEARCH_TOP_K_LAWS, filter={"source_type": "law"}
    )
    filtered_laws = [
        {"similarity": score, "content": doc.page_content}
        for doc, score in results_laws if score >= threshold
    ][:MAX_DISPLAY_LAWS]
    return filtered_cases, filtered_laws

def generate_improvement_proposal(clause: str, unfair_type: str, cases: List[Dict], laws: List[Dict], langsmith_project: str) -> str:
    if unfair_type == "N/A":
        return "This clause is considered fair and does not require improvement."
    cases_text = "\n".join([f"- {c['content']}" for c in cases])
    laws_text = "\n".join([f"- {l['content']}" for l in laws])
    prompt = ACTIVE_IMPROVEMENT_PROMPT.format(
        original_clause=clause,
        unfair_type=unfair_type,
        related_cases=cases_text,
        related_laws=laws_text,
        feedback_context=""
    )
    with tracing_v2_enabled(
        project_name=langsmith_project, # "contract-review-batch",
        tags=["improvement_generation", "batch_test"]
    ):
        result = llm.invoke(prompt).content
    return result

def batch_process_test_data(csv_file: str, output_file: str = 'batch_test_results.jsonl', langsmith_project: str = "contract-review-batch"):
    print(f"Loading test data: {csv_file}")
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"Total records: {len(df)}")
    print(f"Columns detected: {df.columns.tolist()}\n")
    if 'clause' not in df.columns:
        print("'clause' column missing. Please check CSV.")
        return
    has_context = 'context' in df.columns
    print(f"Context column present: {has_context}\n")
    vectorstore = load_vectordb()
    results = []
    print(f"LangSmith project: {langsmith_project}")
    print("Starting batch processing...\n")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            clause_raw = row['clause']
            context = row.get('context', None) if has_context else None
            full_clause = prepare_full_clause(clause_raw, context)
            cleaned = clean_text(full_clause)
            sample_metadata = {
                "batch_index": int(idx),
                "csv_file": csv_file,
                "has_context": bool(context and pd.notna(context)),
                "label": safe_get_value(row, 'label', default="N/A"),
                "ground_truth_type": safe_get_value(row, 'ground_truth_type', default="N/A"),
                "ground_truth_improvement": safe_get_value(row, 'ground_truth_improvement', default="N/A")
            }
            fairness = classify_fairness(cleaned, langsmith_project)
            ####
            print("**********")
            print(f"Index: {idx}, fairness: {fairness!r}")
            if fairness == '공정':
                unfair_type = "N/A"   
                improvement = None
                cases = []
                laws = []             
            else:
                unfair_type = classify_unfair_type(cleaned, fairness, langsmith_project)
                print(f"unfair_type: {unfair_type} *****************")
                cases, laws = retrieve_cases_and_laws(vectorstore, cleaned, unfair_type)
                improvement = generate_improvement_proposal(cleaned, unfair_type, cases, laws, langsmith_project)
                
                
###                
            print(f"Index: {idx}, cases count: {len(cases)}, laws count: {len(laws)}")
            print("**********")
    
            result = {
                "timestamp": datetime.now().isoformat(),
                "index": int(idx),
                "original_clause": clause_raw,
                "context": context if (context and pd.notna(context)) else None,
                "full_clause": full_clause,
                "cleaned_text": cleaned,
                "predicted_fairness": fairness,
                "predicted_type": unfair_type,
                "predicted_improvement": improvement,
                "label": sample_metadata["label"], 
                "ground_truth_type": sample_metadata["ground_truth_type"],
                "ground_truth_improvement": sample_metadata["ground_truth_improvement"],
                "retrieved_cases_count": len(cases),
                "retrieved_laws_count": len(laws),
                "retrieved_cases": [c['content'] for c in cases],
                "retrieved_laws": [l['content'] for l in laws]
            }
            results.append(result)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"\nError processing index {idx}: {e}")
            error_result = {
                "timestamp": datetime.now().isoformat(),
                "index": int(idx),
                "original_clause": clause_raw,
                "error": str(e),
                "predicted_fairness": None,
                "predicted_type": None,
                "predicted_improvement": None,
                "label": "N/A",
                "ground_truth_type": "N/A",
                "ground_truth_improvement": "N/A"
            }
            results.append(error_result)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
    print(f"\nBatch processing finished! Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python batch_test.py <test_csv_file> [langsmith_project_name]")
        sys.exit(1) # 종료
    csv_file = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else "contract-review-batch"
    batch_process_test_data(csv_file, langsmith_project=project_name)
    
    
    # python batch_test.py test.csv batch-test
    # test.csv : 테스트셋
    # batch-test : 랭스미스 트래킹 시 사용할 프로젝트명