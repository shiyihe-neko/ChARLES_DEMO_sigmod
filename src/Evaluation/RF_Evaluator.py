import os
import pandas as pd
import pickle
import re
import copy
import numpy as np
from Evaluation.Util import Metric, Approach
from Util.Util import is_categorical, maybe_primary_key
from pandas.api.types import is_numeric_dtype
from LinearTree.RF_LinearTree import LinearTree, generate_attribute_combinations
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from collections import Counter

etol = 0.01
error_tolerances = [0.0, 0.01, 0.05, 0.10, 0.20]

class Evaluator:
    def __init__(self, approach_name, source_data_path, target_data_path_root, result_path_root):
        self.approach_name = approach_name
        #self.max_depth = max_depth
        self.source_data_path = source_data_path                      # Source dataset
        self.target_data_path_root = target_data_path_root            # Root folder for the transformed datasets 
        self.result_path_root = result_path_root   

                        # Root folder where to save results
        #self.result_template = result_template        
        #self.target_dataset_paths = [f for f in os.listdir(self.target_data_path_root) if f.endswith('csv')]

        if os.path.isfile(self.target_data_path_root) and self.target_data_path_root.endswith('csv'):
            self.target_dataset_paths = [self.target_data_path_root]
        else:
            self.target_dataset_paths = [f for f in os.listdir(self.target_data_path_root) if f.endswith('csv')]

        self.mse_loss = dict() 
        self.mse = dict()
        self.best_depths_per_LMT = {} 
        self.relative_error = 0
        self.coverage = 0
        self.mse = 0
        self.Part_Prec_Tup_TranE = 0
        self.runtime = {}
        self.runtime_best_depth = 0
        if not hasattr(self, "metrics_store"):  # Initialize only once
            self.metrics_store = {}
    
    def _normalize_condition(self, attr, op, val, df):
        """Normalize condition based on attribute type."""
        # for numerical attributes we need to check:
        ####  y > 7 should be considered same as y >=8; 
        #### also y < 6 should be considered same as y <= 5
        if is_numeric_dtype(df[attr]):
            if op == '>':
                return (attr, '>=', val + 1)
            elif op == '<':
                return (attr, '<=', val - 1)
            return (attr, op, val)
        else:
            # For categorical with 2 values
            ## also if a categorical attribute has only two distinct values, 
            ## then for example y != x1 is same as y = x2
            unique_vals = df[attr].unique()
            if len(unique_vals) == 2 and op == '!=':
                other_val = [v for v in unique_vals if v != val][0]
                return (attr, '==', other_val)
            return (attr, op, val)
    
    def _get_ct_text(self, ct, source_df):
        """Get text representation of conditional transformation"""
        conditions = []
        for attr, op, val in ct.partition.conditions:
           ## conditions.append(f"{attr} {op} {val}")
###### new trial condition check
            norm_cond = self._normalize_condition(attr, op, val, source_df)
            conditions.append(f"{norm_cond[0]} {norm_cond[1]} {norm_cond[2]}")
#########################

        return " AND ".join(sorted(conditions))
    
    def get_transformation_function(self, ct):
        """Get transformation function details from CT."""
        transform = ct.single_transformation
        target = transform.target_attribute
        indep_vars = transform.independent_attributes
        coeffs = transform.coefficients
        
        # Format: target = coeff1*var1 + coeff2*var2 + ... + intercept
        terms = [f"{coeff}*{var}" for coeff, var in zip(coeffs[:-1], indep_vars)]
        function_str = f"{target} = {' + '.join(terms)} + {coeffs[-1]}"
        
        return {
            'target': target,
            'variables': indep_vars,
            'coefficients': coeffs,
            'function_string': function_str
        }
    
    # matching transformation function
    def _functions_match(self, pct, gct):
        """Compare transformation functions between predicted and ground truth CTs."""
        p_func = self.get_transformation_function(pct)
        g_func = self.get_transformation_function(gct)

        return (p_func['target'] == g_func['target'] and
                set(p_func['variables']) == set(g_func['variables']) and
                np.allclose(p_func['coefficients'], g_func['coefficients'], rtol=1e-5))
    
    ##∣actual−predicted∣≤(atol+rtol×∣predicted∣)
     ###np.allclose allows some variation based on the scale of the values.

    # comparing actual tuple values
    def _values_match(self, pct_indices, actual_values, predicted_values, rtol=1e-2):
        """Compare actual and predicted values for given indices."""
        actual_subset = actual_values[pct_indices]
        predicted_subset = predicted_values[pct_indices]
        relative_errs = np.where(actual_subset !=0,
                                 np.abs((predicted_subset - actual_subset) / actual_subset),
                                 np.abs(predicted_subset))
        return np.all(relative_errs <= rtol)
    
    def evaluate(self,n = 3, m = 3):
        if self.approach_name == Approach.get_name(Approach.LINEAR_TREE): 
            start_time = time.time()
            source_df = pd.read_csv(self.source_data_path)

            # Debugging: Print loaded target dataset paths
            print("Target Dataset Paths:", self.target_dataset_paths)

            for dataset_path in self.target_dataset_paths:
                transformation_index = int(dataset_path.split('_')[-1].split('.')[0])

                dataset_folder = os.path.join(self.result_path_root, f'dataset_{transformation_index}')
                metric_data = []
                start_time = time.time()
                metrics_group = {}
                ### REMOVE THE LINE BELOW AS THIS LEAKS GROUND TRUTH TO THE LINEAR TREE MODEL ###
                # potential_transformation_attributes = pickle.load(open(os.path.join('temp_transformations', 'trans_{}.pkl'.format(transformation_index)),'rb'))[0].single_transformation.independent_attributes
                ### REMOVE THE LINE ABOVE AS THIS LEAKS GROUND TRUTH TO THE LINEAR TREE MODEL ###
                #target_df = pd.read_csv(os.path.join(self.target_data_path_root, dataset_path)) 
                target_df = pd.read_csv(dataset_path) 
                attribute_combos = generate_attribute_combinations(source_df,target_df, n, m, transformation_index)
                self.best_depths_per_LMT[transformation_index] = {}

                for attributes in attribute_combos:
                    best_model = None # RF
                    prev_loss = float('inf') #adjustable depth
                    best_depth = None #adjustable depth

                    for depth in range(3,4):
                        linear_tree = LinearTree(source_df = source_df,
                                                target_df = target_df,
                                                max_depth = depth,
                                                transformation_index=transformation_index,
                                                random_state= 0,
                                                potential_split_attributes=[attr for attr in attributes if is_categorical(source_df[attr])],
                                                potential_transformation_attributes=[attr for attr in attributes if 
                                                                                     not is_categorical(source_df[attr]) and
                                                                                     not maybe_primary_key(source_df[attr]) and
                                                                                     is_numeric_dtype(source_df[attr]) and
                                                                                     attr != 'base_bonus'
                                                                                     ],
                                                loss_criterion = Metric.MEAN_SQUARED_ERROR)
                        if linear_tree.loss < prev_loss:
                            prev_loss = linear_tree.loss
                            best_model = linear_tree
                            best_depth = depth
                        else:
                            break  # Stop if increasing depth does not improve performance

                    # Store the best depth for this LMT under this dataset
                    self.best_depths_per_LMT[transformation_index][tuple(attributes)] = best_depth

                    num_pcts = len(best_model.transformation.conditional_transformations)

            #end_time = time.time()
            #self.runtime_best_depth = end_time - start_time
                  
                    file_prefix = f'approach_linear_tree_trans_{transformation_index}_LMT_{"_".join(attributes)}'
                    best_model.store(dataset_folder, f"{file_prefix}.txt")
                    txt_file_path = os.path.join(dataset_folder, f"{file_prefix}.txt")
                    with open(os.path.join(dataset_folder, f"{file_prefix}.pkl"), 'wb') as f:
                        pickle.dump(best_model, f)

                    # Parse the .txt file to extract conditions and transformations
                    with open(txt_file_path, 'r') as f:
                        txt_content = f.read()

                    # Extract conditions and transformations
                    partitions = re.findall(
                    r'Partition condition:\s*(.*?)\s*Partition transformation function:\s*(.*?)\s*(?=Partition|$)',
                    txt_content,
                    flags=re.IGNORECASE | re.DOTALL
                        )
                    combined_conditions = []
                    for condition, transformation in partitions:
                        combined_conditions.append(f"{condition} -> {transformation}")

                    # Combine all conditions and transformations into a single string, separated by new lines
                    combined_value = "\n".join(combined_conditions)

                    # Extract cardinalities
                    cardinality_list = re.findall(
                            r"Partition cardinality:\s*(\d+)", txt_content
                            )
                    
                    cardinality = ", ".join(cardinality_list)

                    self.mse_loss[(transformation_index,tuple(attributes))] = best_model.loss
                    self.runtime[(transformation_index,tuple(attributes))] = time.time() - start_time

                    actual_values = copy.deepcopy(target_df[best_model.target_col]).astype(float)
                    predicated_values = copy.deepcopy(source_df[best_model.target_col]).astype(float)

                    predicted_cts = best_model.transformation.conditional_transformations
                    #num_partitions_pred_cts = len(predicted_cts)
                    #ground_truth_cts = pickle.load(open(os.path.join(self.target_data_path_root, 'trans_{}.pkl'.format(transformation_index)), 'rb')).conditional_transformations
                    #num_partitions_gt_cts = len(ground_truth_cts)

                    tough_match_value = 0 # track conditional matches, # tuples match (cardinality) and indices match and exact tuple values
                    
                    for pct in predicted_cts:
                        #predicted_cts_tuples += pct.partition.cardinality
                        predicated_values[pct.partition.tuple_indices] = pct.single_transformation.apply(source_df.iloc[pct.partition.tuple_indices])
                        '''
                        for gct in ground_truth_cts: ### this can your 1.4
                            ##gct_text = self._get_ct_text(gct,source_df)
                            ##if pct_text == gct_text:
                            if pct.partition.cardinality == gct.partition.cardinality:
                                if set(pct.partition.tuple_indices) == set(gct.partition.tuple_indices):
                                    if self._values_match(pct.partition.tuple_indices,actual_values,predicated_values):
                                        tough_match_value += 1
                                        break
                        '''
                        ### have to write 1.5 from scratch
                    mse = mean_squared_error(predicated_values, actual_values)
                    # assert(np.abs(mse - linear_tree.loss) < 1e-1)
                    
                    # mean relative error
                    predicated_values = np.array(predicated_values)
                    actual_values = np.array(actual_values)
                    relative_errors = np.abs((predicated_values - actual_values)/actual_values)

                    coverage = np.count_nonzero(relative_errors <= etol) / source_df.shape[0]
                   #part_prec_tup_tranE = tough_match_value / len(predicted_cts) if predicted_cts else 0
                    # Store metrics per LMT variant
    
                    metrics_group[tuple(attributes)] = {
                    "coverage": coverage,
                    "relative_error": np.mean(relative_errors),
                    "mse": mse,
                    #"Part_Prec_Tup_TranE": part_prec_tup_tranE,
                    "best_depth": best_depth,
                    "num_pcts": num_pcts,  # Add the number of PCTs
                    "cts": combined_value,
                    "cardinality" : cardinality
                    }
                    # Prepare metric data for CSV export
                    #metric_data = []

                for attributes, metrics in metrics_group.items():
                    metric_data.append([
                        attributes,
                        metrics["relative_error"],
                        metrics["coverage"],
                        metrics["mse"],
                        #metrics["Part_Prec_Tup_TranE"],
                        metrics["best_depth"],
                        metrics["num_pcts"],  # Include the number of PCTs
                        metrics["cts"],
                        metrics["cardinality"]
                    ])

                    #self.coverage[(transformation_index, tuple(attributes))] = coverage
                    #self.coverage = coverage
                    #self.relative_error[(transformation_index, tuple(attributes))] = np.mean(relative_errors)
                    #self.relative_error = np.mean(relative_errors)
                    #self.Part_Prec_Tup_TranE = part_prec_tup_tranE    
                    #metric_data.append(["_".join(attributes),self.relative_error, self.coverage, self.Part_Prec_Tup_TranE])
                    
                df = pd.DataFrame(metric_data, columns=['LMT', 'Rel_Error', 'Coverage','MSE','best_depth','num_pcts','cts','cardinality'])
                df.to_csv(os.path.join(dataset_folder, f'metrics_summary_{transformation_index}.csv'), index=False)                    
                
        else:
            raise Exception("The specified model is  not supported yet.")                       
'''
    def plot_metrics(self):
        # Load the combined results
        result_df = pd.read_csv(os.path.join(self.result_path_root, 'result_summary_combined.csv'))
        
        # Define bins and labels
        bins = [0, 0.3, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]
        labels = ['0-30%', '30-50%', '50-75%', '75-85%', '85-90%', '90-95%', '95-100%']
        
        # Metrics to plot
        #metrics = ['Part_Prec','Part_Prec_Cond','Part_Prec_Tup', 'Part_Prec_Tup_TranF', 'Part_Prec_Tup_TranE', 'Coverage','Tup_Prec_Part']
        metrics = ['Part_Prec_Tup_TranE', 'Coverage']

        # Create a figure with subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            # Bin the data
            result_df[f'{metric}_bin'] = pd.cut(result_df[metric], bins=bins, labels=labels, include_lowest=True)
            
            # Calculate the percentage of rows in each bin
            bin_counts = result_df[f'{metric}_bin'].value_counts(normalize=True) * 100
            
            # Sort the bin counts by bin order
            bin_counts = bin_counts.reindex(labels).fillna(0)
            
            # Plot the bar chart in the corresponding subplot
            ax = bin_counts.plot(kind='bar', width=0.95, color='skyblue', ax=axes[i])
            ax.set_xlabel(f'Range of {metric}', fontsize=8)
            ax.set_ylabel('Percentage of Transformed datasets',fontsize=8)
            #ax.set_title(f'Distribution of {metric}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45,fontsize=8)
            
            # Label each bar with its value
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', 
                            xytext=(0, 9), 
                            textcoords='offset points') 
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path_root, 'metrics_distribution.png'))
        plt.close()

        total_runtime = result_df['RunTime'].sum() + self.runtime_best_depth
         # Create a DataFrame for the total runtime
        total_runtime_df = pd.DataFrame({'Model': ['LMT_Adj_dep'], 'Runtime': [total_runtime]})

        # Plot the bar chart
        plt.figure(figsize=(5, 4))
        ax = total_runtime_df.plot(kind='bar',width =0.1, x='Model', y='Runtime', legend=False, color='skyblue')
        plt.ylabel('Runtime')
        plt.title('Total Runtime', loc = 'left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        # Add a label to the bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())} s', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', xytext=(0, 11), 
                        textcoords='offset points')
            
        plt.savefig(os.path.join(self.result_path_root, 'runtime_distribution.png'))
        plt.close()

        ########## if want to create separate plots
      
    '''

