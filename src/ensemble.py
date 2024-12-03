import pandas as pd
from collections import Counter

def ensemble_predictions(csv_files, method='majority', weights=None):
    predictions = []
    for file in csv_files:
        df = pd.read_csv(file)
        predictions.append(df.set_index('id')['answer'])
    
    result_df = pd.DataFrame(index=predictions[0].index)
    
    if method == 'majority':
        for idx in result_df.index:
            votes = [pred[idx] for pred in predictions]
            result_df.loc[idx, 'answer'] = Counter(votes).most_common(1)[0][0]
            
    elif method == 'weighted':
        if weights is None:
            weights = [1/len(predictions)] * len(predictions)

        for idx in result_df.index:
            votes = [pred[idx] for pred in predictions]
            weighted_votes = {k: 0 for k in set(votes)}
            for vote, weight in zip(votes, weights):
                weighted_votes[vote] += weight
            result_df.loc[idx, 'answer'] = max(weighted_votes.items(), key=lambda x: x[1])[0]
    
    # 신뢰도 점수 추가
    result_df['confidence'] = 0.0
    for idx in result_df.index:
        votes = [pred[idx] for pred in predictions]
        majority_count = Counter(votes).most_common(1)[0][1]
        result_df.loc[idx, 'confidence'] = majority_count / len(predictions)
    
    return result_df.reset_index()

def process_ensemble_file(df, output_file: str):
    """결과를 DataFrame에서 바로 처리"""
    # confidence 열 제거 및 answer를 int로 변환
    final_df = df[['id', 'answer']].copy()
    final_df['answer'] = final_df['answer'].astype(int)
        
    final_df.to_csv(output_file, index=False)
    print(f"\nProcessed results saved to: {output_file}")
        
    # 참고: 정답 개수 확인
    print("\n정답 개수 분포:")
    print(final_df['answer'].value_counts().sort_index())

def main():
    # csv 파일 이름 작성
    csv_files = [
        '8041_qw32_19.csv',
        '8065_qw32_5.csv',
        '8065_qw32_13.csv',
        '8088_qw32_7.csv',
        '8088_qw32_12.csv',
        '8157_qw32_23.csv',
        '8180_qw32_24.csv'
    ]
    
    # 다수결 앙상블 -> 앙상블 파일 저장이름 작성
    majority_results = ensemble_predictions(csv_files, method='majority')
    process_ensemble_file(
        df=majority_results,
        output_file='final_ensemble_majority.csv'
    )

    # 가중 평균 앙상블 -> 가중치, 앙상블 파일 저장이름 작성
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]  
    weighted_results = ensemble_predictions(csv_files, method='weighted', weights=weights)
    process_ensemble_file(
        df=weighted_results,
        output_file='final_ensemble_weighted.csv'
    )

if __name__ == "__main__":
    main()