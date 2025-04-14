import numpy as np
import random
import pandas as pd
from pandas.api.types import is_numeric_dtype
from Evaluation.Util import Metric
from Util.Transformation import ConditionalTransformation, Partition, SingleTransformation, Transformation
from Util.Util import is_categorical, maybe_primary_key, RELATIONAL_OPERATORS, flip
from Util.attr_search_spac_red import filter_attributes
from sklearn.linear_model import LinearRegression # linear regression
from sklearn.metrics import mean_squared_error
from itertools import combinations
import os
import time

EPS = 1.0               # Very small loss and if loss is this small, can assume perfect fit
MIN_PARTITION_SIZE = 2  # minimum size of a valid partition

def generate_attribute_combinations(source_df,target_df, n, m, transformation_index):
    # MAY BE ADD TARGET_DF HERE AS PARAMETER
    """
    Generate all valid (n, m) combinations of categorical and numerical attributes.
    """
    categorical_attrs = [attr for attr in source_df.columns if is_categorical(source_df[attr])]
    # MAY BE USE THE FUNCTION FROM NEW_CHECK.PY ? 

    numerical_attrs = [attr for attr in source_df.columns if 
                       not is_categorical(source_df[attr]) and
                       not maybe_primary_key(source_df[attr]) and
                       is_numeric_dtype(source_df[attr]) 
                       and attr != 'base_bonus'
                        ]
    print(f"Numerical Attributes: {numerical_attrs}")
    print(f"Categorical Attributes: {categorical_attrs}")
    filtered_categorical_attributes, filtered_numerical_attributes = filter_attributes(target_df, transformation_index,numerical_attrs,categorical_attrs)
    print(f"Filtered Categorical Attributes: {filtered_categorical_attributes}")
    print(f"Filtered Numerical Attributes: {filtered_numerical_attributes}")
    
    #filtered_categorical_attributes = categorical_attrs.copy()
    #filtered_numerical_attributes = numerical_attrs.copy()
    # Generate all possible selections of categorical (split) and numerical (transform) attributes
    cat_combos = [list(comb) for i in range(1, n + 1) for comb in combinations(filtered_categorical_attributes, i)]
    num_combos = [list(comb) for i in range(1, m + 1) for comb in combinations(filtered_numerical_attributes, i)]
    
    # Combine categorical and numerical attribute subsets
    attr_combinations = []
    for cat in cat_combos:
        for num in num_combos:
            attr_combinations.append(cat + num)
    return attr_combinations

class LinearTree:
    def __init__(self, 
                 source_df: pd.DataFrame, 
                 target_df: pd.DataFrame, 
                 transformation_index = 0,
                 target_col = None, 
                 potential_split_attributes = None,
                 potential_transformation_attributes = None,
                 random_state = 0,n =1, m = 1,             
                 max_depth = 5, 
                 loss_criterion = Metric.MEAN_SQUARED_ERROR, 
                 conditions = []):
        self.source_df = source_df
        self.target_df = target_df
        '''
        if self.target_col is None:
            changed_cols = [col for col in source_df.columns if not np.array_equal(np.array(source_df[col]), np.array(target_df[col]))]
            assert len(changed_cols) == 1, "Currently change in exactly one target attribute is supported"
            self.target_col = changed_cols[0]
        '''
        self.target_col = target_col
        if self.target_col is None:
            changed_cols = [col for col in source_df.columns if col in target_df.columns and 
                    not np.array_equal(np.array(source_df[col]), np.array(target_df[col]))]

            if not changed_cols:
                raise ValueError("No changed column detected between source and target data. Check input files.")

            assert len(changed_cols) == 1, f"Expected exactly one target attribute to change, but found {len(changed_cols)}: {changed_cols}"
            self.target_col = changed_cols[0]

# Debugging print statements (remove them after debugging)
        print(f"Detected Target Column: {self.target_col}")

        self.attribute_combinations = []
        if potential_split_attributes is None or potential_transformation_attributes is None:
            attribute_combos = generate_attribute_combinations(self.source_df,self.target_df, n, m, transformation_index)
    
    # Store only valid attribute combinations
            for combo in attribute_combos:
                split_attrs = [attr for attr in combo[0] if is_categorical(self.source_df[attr])]
                trans_attrs = [attr for attr in combo[1] if is_numeric_dtype(self.source_df[attr]) 
                                                             and attr != self.target_col 
                                                            # and not is_categorical(self.source_df[attr]) and
                                                            #not maybe_primary_key(self.source_df[attr])
                                                            ]
                if trans_attrs:  # Ensure transformation attributes exist
                    self.attribute_combinations.append((split_attrs, trans_attrs))
        else:
            self.attribute_combinations = [(potential_split_attributes, potential_transformation_attributes)]
        
        self.max_depth = max_depth
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.loss_criterion = loss_criterion
        self.conditions = conditions            # conditions passed from parent, for root node, it is empty
        assert(self.loss_criterion == Metric.MEAN_SQUARED_ERROR)

        self.split_condition = None             # how to best split this node
        self.left_tree = None                   # Unless leaf, will be none
        self.right_tree = None                  # Unless leaf, will be none
        self.model = None                       # Unless leaf, will be none
        self.transformation_attributes = None   # Unless leaf, will be none
        self.loss = 1e100
        self.transformation = None
        self.fit()

        merge_happened = True

        while merge_happened:
            merge_happened = False
            new_cts = []
            invalids = []

            for i in range(len(self.transformation.conditional_transformations) - 1):
                for j in range(i + 1, len(self.transformation.conditional_transformations)):
                    ct1 = self.transformation.conditional_transformations[i]
                    ct2 = self.transformation.conditional_transformations[j]
                    
                    if ct1.single_transformation.matches(ct2.single_transformation):
                        merged_partition = ct1.partition.merge(ct2.partition)
                        if merged_partition is not None:
                            merged_partition.process(self.source_df)
                            merged_ct = ConditionalTransformation(
                                 partition=merged_partition, 
                                 single_transformation=ct1.single_transformation)
                            new_cts.append(merged_ct)
                            invalids.append(i)
                            invalids.append(j)
                            merge_happened = True
            
            for i in range(len(self.transformation.conditional_transformations)):
                if i not in invalids:
                     new_cts.append(self.transformation.conditional_transformations[i])
            
            self.transformation = Transformation(conditional_transformations=new_cts)

        # Initialize runtime attribute
        #self.runtime = 0

    def fit(self):
        if np.array_equal(self.target_df[self.target_col], self.source_df[self.target_col]):
            self.model = None
            self.loss = 0
            self.losses = [self.loss]
            self.transformation = Transformation(conditional_transformations=[])
            return
        
        #attribute_combos = generate_attribute_combinations(self.source_df,self.target_df, n, m)

        for split_attrs, trans_attrs in self.attribute_combinations:
            if not trans_attrs:
                continue  # Skip if no numerical attributes are available

            model = LinearRegression()
            print(f"Target Column: {self.target_col}")
            print(f"Transformation Attributes: {self.transformation_attributes}")
            model.fit(X=self.source_df[trans_attrs], y=self.target_df[self.target_col])
            predicted_values = model.predict(X=self.source_df[trans_attrs])
            cur_loss = mean_squared_error(y_true=self.target_df[self.target_col], y_pred=predicted_values)

            if cur_loss < self.loss:
                self.loss = cur_loss
                self.model = model
                #self.potential_transformation_attributes = trans_attrs
                self.transformation_attributes = trans_attrs
                self.potential_split_attributes = split_attrs  

        if self.max_depth == 0 or self.loss < EPS:  
            cur_partition = Partition(conditions=self.conditions)
            cur_partition.process(self.source_df)
            self.transformation = Transformation(conditional_transformations=[ConditionalTransformation(
                partition=cur_partition,
                single_transformation=SingleTransformation(
                     target_attribute=self.target_col,
                     independent_attributes=self.transformation_attributes,
                     coefficients=list(self.model.coef_) + [self.model.intercept_])
            )])
            self.losses = [self.loss]
            return

        # Ensure at least one categorical attribute exists before splitting
        if not self.potential_split_attributes:
            return

        min_gain = 0
        best_split_parameters = None
        for attr in self.potential_split_attributes:
            potential_operators = RELATIONAL_OPERATORS if is_numeric_dtype(self.source_df[attr]) else RELATIONAL_OPERATORS[:2]
            for op in potential_operators:
                for val in np.unique(self.source_df[attr]):
                    left_partition = Partition(conditions=[(attr, op, val)])
                    relevant_source_df = left_partition.apply(self.source_df)
                    relevant_target_df = left_partition.apply(self.target_df)
                    if relevant_source_df.shape[0] < MIN_PARTITION_SIZE:
                        continue

                    model.fit(X=relevant_source_df[self.transformation_attributes], y=relevant_target_df[self.target_col])
                    predicted_values = model.predict(X=relevant_source_df[self.transformation_attributes])
                    total_loss_left = mean_squared_error(y_true=relevant_target_df[self.target_col], y_pred=predicted_values) * relevant_source_df.shape[0]

                    right_partition = Partition(conditions=[flip((attr, op, val))])
                    relevant_source_df = right_partition.apply(self.source_df)
                    if relevant_source_df.shape[0] < MIN_PARTITION_SIZE:
                        continue

                    relevant_target_df = right_partition.apply(self.target_df)
                    model.fit(X=relevant_source_df[self.transformation_attributes], y=relevant_target_df[self.target_col])
                    predicted_values = model.predict(X=relevant_source_df[self.transformation_attributes])
                    total_loss_right = mean_squared_error(y_true=relevant_target_df[self.target_col], y_pred=predicted_values) * relevant_source_df.shape[0]

                    gain = self.loss * self.source_df.shape[0] - (total_loss_left + total_loss_right)
                    if gain > min_gain:
                        min_gain = gain
                        best_split_parameters = (attr, op, val)

        
        if best_split_parameters is not None:
            cur_left_tree = LinearTree(source_df = Partition(conditions = [best_split_parameters]).apply(self.source_df),
                                        target_df = Partition(conditions = [best_split_parameters]).apply(self.target_df),
                                        potential_split_attributes = self.potential_split_attributes,
                                        #potential_transformation_attributes = self.potential_transformation_attributes,
                                        potential_transformation_attributes = self.transformation_attributes,
                                        target_col = self.target_col, 
                                        max_depth = self.max_depth - 1, 
                                        random_state = self.random_state,
                                        loss_criterion = self.loss_criterion,
                                        conditions  = self.conditions + [best_split_parameters])
            cur_right_tree = LinearTree(source_df = Partition(conditions = [flip(best_split_parameters)]).apply(self.source_df),
                                        target_df = Partition(conditions = [flip(best_split_parameters)]).apply(self.target_df),
                                        potential_split_attributes = self.potential_split_attributes,
                                        #potential_transformation_attributes = self.potential_transformation_attributes,
                                        potential_transformation_attributes = self.transformation_attributes,
                                        target_col = self.target_col, 
                                        max_depth = self.max_depth - 1, 
                                        random_state = self.random_state,
                                        loss_criterion = self.loss_criterion,
                                        conditions  = self.conditions + [flip(best_split_parameters)])
                
            if cur_left_tree.loss * cur_left_tree.source_df.shape[0] + cur_right_tree.loss * cur_right_tree.source_df.shape[0] < self.loss * self.source_df.shape[0]:
                self.loss = (cur_left_tree.loss * cur_left_tree.source_df.shape[0] + cur_right_tree.loss * cur_right_tree.source_df.shape[0])/self.source_df.shape[0]
                self.left_tree = cur_left_tree
                self.right_tree = cur_right_tree
                self.split_condition = best_split_parameters
                self.model = None
                self.transformation_attributes = None
                self.transformation = Transformation(conditional_transformations= cur_left_tree.transformation.conditional_transformations + cur_right_tree.transformation.conditional_transformations)
                self.losses = cur_left_tree.losses + cur_right_tree.losses
        
        if self.model is not None:  # It is best as a leaf
            cur_partition = Partition(conditions = self.conditions)
            cur_partition.process(self.source_df)
            self.transformation = Transformation(conditional_transformations=[ConditionalTransformation(
                partition=cur_partition,
                single_transformation = SingleTransformation(
                     target_attribute = self.target_col,
                     independent_attributes = self.transformation_attributes,
                     coefficients = list(self.model.coef_) + [self.model.intercept_]))])
            self.losses = [self.loss]

        #end_time = time.time()  # End timing
       # self.runtime = (end_time - start_time)*1000000  # Calculate runtime

    def store(self, result_path_root, file_name):
        self.transformation.store(
             df = self.source_df,
             detailed_text_path=os.path.join(result_path_root, file_name),
             pickle_dump_path=os.path.join(result_path_root, file_name.replace('txt', 'pkl')))
