import pandas as pd
import os
#alpha = 0.1
def calculate_score(row, total_attributes, alpha, beta1=0.5, beta2=0.5):
    """
    Calculate the score for a row using two components of I:
    - I1: Based on the number of attributes used in split
    - I2: Based on the number of partition conditions
    """
    I1 = 1 - (len(row['LMT'].strip("()").split(", ")) / total_attributes)
    I2 = 1 - (row['num_pcts'] / total_attributes)
    Interpretability = I1 * (beta1) + I2 * (beta2)
    Accuracy = row['Coverage']
    Score = ((1 - alpha) * Interpretability) + (Accuracy * alpha)
    return Score, Interpretability

def score_and_sort_metrics(result_path_root, transformation_index, total_attributes, alpha):
    metrics_file_path = f'{result_path_root}/dataset_{transformation_index}/metrics_summary_{transformation_index}.csv'
    sorted_metrics_file_path = f'{result_path_root}/dataset_{transformation_index}/metrics_summary_sorted_{transformation_index}.csv'
    reduced_metrics_file_path = f'{result_path_root}/dataset_{transformation_index}/metrics_summary_sorted_scores_CTs{transformation_index}.csv'
    # Read the metrics file
    metrics_df = pd.read_csv(metrics_file_path)
    # Calculate the score for each row
    metrics_df[['Score', 'Interpretability']] = metrics_df.apply(
        lambda row: pd.Series(calculate_score(row, total_attributes, alpha, beta1 = 0.5, beta2 = 0.5)),
        axis=1
    )

    # Sort the DataFrame by score in descending order
    sorted_metrics_df = metrics_df.sort_values(by='Score', ascending=False)
    sorted_metrics_df = sorted_metrics_df.rename(columns={'Coverage': 'Accuracy'})

    # Convert Accuracy, Interpretabilityability, and Score to percentage values with no decimals
    sorted_metrics_df['Accuracy'] = (sorted_metrics_df['Accuracy'] * 100).round(0).astype(int)
    sorted_metrics_df['Interpretability'] = (sorted_metrics_df['Interpretability'] * 100).round(0).astype(int)
    sorted_metrics_df['Score'] = (sorted_metrics_df['Score'] * 100).round(0).astype(int)

    reduced_metrics_df = sorted_metrics_df[['LMT','Score', 'Accuracy', 'Interpretability', 'cts', 'cardinality']]
    reduced_metrics_df = reduced_metrics_df.drop_duplicates(subset=['cts'], keep='first')
    # Write the sorted DataFrame to a new CSV file
    sorted_metrics_df.to_csv(sorted_metrics_file_path, index=False)
    reduced_metrics_df.to_csv(reduced_metrics_file_path, index=False)



# def collect_top_scores(result_path_root, sorted_metrics_paths):
#     top_scores = []
#     for dataset_path in sorted_metrics_paths:
#         if dataset_path.startswith('dataset_'):
#             transformation_index = int(dataset_path.split('_')[-1])

#         sorted_metrics_file_path = f'{result_path_root}/dataset_{transformation_index}/metrics_summary_sorted_{transformation_index}.csv'
    
#         sorted_metrics_df = pd.read_csv(sorted_metrics_file_path)
#         if not sorted_metrics_df.empty:
#             top_row = sorted_metrics_df.iloc[0]
#             top_row['transformation_index'] = transformation_index
#             top_scores.append(top_row)

#     top_scores_df = pd.DataFrame(top_scores)
#     top_scores_df = top_scores_df.sort_values(by = 'transformation_index',ascending=True)
#     top_scores_df.to_csv(f'{result_path_root}/top_scores_summary.csv', index=False)



def collect_top_scores(result_path_root, sorted_metrics_paths):
    top_scores = []
    for dataset_path in sorted_metrics_paths:
        if dataset_path.startswith('dataset_'):
            transformation_index = int(dataset_path.split('_')[-1])
            sorted_metrics_file_path = f'{result_path_root}/dataset_{transformation_index}/metrics_summary_sorted_{transformation_index}.csv'
        
            sorted_metrics_df = pd.read_csv(sorted_metrics_file_path)
            if not sorted_metrics_df.empty:
                top_row = sorted_metrics_df.iloc[0]
                top_row['transformation_index'] = transformation_index
                top_scores.append(top_row)

    top_scores_df = pd.DataFrame(top_scores)
    top_scores_df = top_scores_df.sort_values(by='transformation_index', ascending=True)
    top_scores_df.to_csv(f'{result_path_root}/top_scores_summary.csv', index=False)


