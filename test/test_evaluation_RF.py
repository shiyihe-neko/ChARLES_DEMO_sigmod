# import os
# import shutil
# from Evaluation.RF_Evaluator import Evaluator
# from Evaluation.Util import Approach
# from Util.Visualization import Visualizer
# from Util.Util import is_categorical, maybe_primary_key
# from pandas.api.types import is_numeric_dtype
# from Evaluation.scoring import score_and_sort_metrics,collect_top_scores
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import sys

# # 设置默认参数
# n = 3
# m = 3
# alpha = 0.1

# # 如果从命令行传入参数，就使用它们
# if len(sys.argv) >= 4:
#     n = int(sys.argv[1])
#     m = int(sys.argv[2])
#     alpha = float(sys.argv[3])


# class Test_Evaluator:        
#     def test_evaluator(self):       
#         approach = Approach.LINEAR_TREE
#         source_data_path = 'c_0.csv'
#         target_data_path_root = 'c_1.csv'     
#         result_path_root = 'ensemble_test/temp_results_RF_' + Approach.get_name(approach) # root folder where to save results
#         alpha  = 0.1
#         if os.path.exists(result_path_root):
#             shutil.rmtree(result_path_root)        
#         os.makedirs(result_path_root)              
#         evaluator = Evaluator(approach_name = Approach.get_name(approach), 
#                             source_data_path = source_data_path, 
#                             target_data_path_root = target_data_path_root, 
#                             result_path_root = result_path_root)
                            
#         for dataset_path in evaluator.target_dataset_paths:
#             transformation_index = int(dataset_path.split('_')[-1].split('.')[0])
#             dataset_folder = os.path.join(result_path_root, f'dataset_{transformation_index}')
#             os.makedirs(dataset_folder, exist_ok=True) 

#         evaluator.evaluate()

#         for dataset_path in evaluator.target_dataset_paths:
#             transformation_index = int(dataset_path.split('_')[-1].split('.')[0])
#             dataset_folder = os.path.join(result_path_root, f'dataset_{transformation_index}')
#             assert os.path.exists(dataset_folder), f"Dataset folder missing: {dataset_folder}"
#             assert os.path.exists(os.path.join(dataset_folder, f'metrics_summary_{transformation_index}.csv')), "Metrics CSV missing"

#         source_data = pd.read_csv(source_data_path)

#         categorical_attrs = [attr for attr in source_data.columns if is_categorical(source_data[attr])]

#         numerical_attrs = [attr for attr in source_data.columns if 
#                        not is_categorical(source_data[attr]) and
#                        not maybe_primary_key(source_data[attr]) and
#                        is_numeric_dtype(source_data[attr]) 
#                        and attr != 'base_bonus']
#         total_attributes = len(categorical_attrs) + len(numerical_attrs)
        
#         for dataset_path in evaluator.target_dataset_paths:
#             transformation_index = int(dataset_path.split('_')[-1].split('.')[0])
#             score_and_sort_metrics(result_path_root, transformation_index, total_attributes, alpha)

#         sorted_metrics_paths = f'{result_path_root}'
#         sorted_paths = [f for f in os.listdir(sorted_metrics_paths)]
#         collect_top_scores(result_path_root,sorted_paths)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from Evaluation.RF_Evaluator import Evaluator
from Evaluation.Util import Approach
from Util.Visualization import Visualizer
from Util.Util import is_categorical, maybe_primary_key
from pandas.api.types import is_numeric_dtype
from Evaluation.scoring import score_and_sort_metrics, collect_top_scores
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt


n = 3
m = 3
alpha = 0.1

if len(sys.argv) >= 4:
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    alpha = float(sys.argv[3])

class Test_Evaluator:        
    def test_evaluator(self):       
        approach = Approach.LINEAR_TREE
        source_data_path = 'c_0.csv'
        target_data_path_root = 'c_1.csv'     
        result_path_root = 'ensemble_test/temp_results_RF_' + Approach.get_name(approach)

        if os.path.exists(result_path_root):
            shutil.rmtree(result_path_root)        
        os.makedirs(result_path_root)              

        evaluator = Evaluator(
            approach_name=Approach.get_name(approach), 
            source_data_path=source_data_path, 
            target_data_path_root=target_data_path_root, 
            result_path_root=result_path_root
        )
                            
        for dataset_path in evaluator.target_dataset_paths:
            transformation_index = int(dataset_path.split('_')[-1].split('.')[0])
            dataset_folder = os.path.join(result_path_root, f'dataset_{transformation_index}')
            os.makedirs(dataset_folder, exist_ok=True)

        evaluator.evaluate(n=n, m=m)

        for dataset_path in evaluator.target_dataset_paths:
            transformation_index = int(dataset_path.split('_')[-1].split('.')[0])
            dataset_folder = os.path.join(result_path_root, f'dataset_{transformation_index}')
            assert os.path.exists(dataset_folder), f"Dataset folder missing: {dataset_folder}"
            assert os.path.exists(os.path.join(dataset_folder, f'metrics_summary_{transformation_index}.csv')), "Metrics CSV missing"

        source_data = pd.read_csv(source_data_path)

        categorical_attrs = [attr for attr in source_data.columns if is_categorical(source_data[attr])]

        numerical_attrs = [
            attr for attr in source_data.columns 
            if not is_categorical(source_data[attr]) and
               not maybe_primary_key(source_data[attr]) and
               is_numeric_dtype(source_data[attr]) and
               attr != 'base_bonus'
        ]

        total_attributes = len(categorical_attrs) + len(numerical_attrs)
        
        for dataset_path in evaluator.target_dataset_paths:
            transformation_index = int(dataset_path.split('_')[-1].split('.')[0])
            score_and_sort_metrics(result_path_root, transformation_index, total_attributes, alpha)

        sorted_metrics_paths = result_path_root
        sorted_paths = [f for f in os.listdir(sorted_metrics_paths)]
        collect_top_scores(result_path_root, sorted_paths)

if __name__ == "__main__":
    evaluator = Test_Evaluator()
    evaluator.test_evaluator()