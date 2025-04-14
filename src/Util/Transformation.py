import copy
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import pickle

from Util.Util import flip

EPS = 1e-3

class Transformation:
    def __init__(self, conditional_transformations, noise_level = 0.0):
        self.conditional_transformations = conditional_transformations
        self.number_of_partitions = len(self.conditional_transformations)

        if self.number_of_partitions == 0:
            return
        
        self.noise_level = noise_level        
        self.number_of_split_attributes = len(set([str(cond[0]) for cond in np.concatenate([ct.partition.conditions for ct in self.conditional_transformations])]))
        self.number_of_transformation_attributes = len(set(np.concatenate([ct.single_transformation.independent_attributes for ct in self.conditional_transformations])))
        self.max_number_of_conditions_in_a_partition = max([len([cond[0] for cond in ct.partition.conditions]) for ct in conditional_transformations])
        self.number_of_tuples_transformed = sum([c.partition.cardinality for c in self.conditional_transformations])
        self.plottable = self.number_of_transformation_attributes == 1 and self.number_of_partitions <= 20

    def store(self, df, detailed_text_path, pickle_dump_path, transformed_data_path = None, plot_path = None, create_plot = False):
        f = open(detailed_text_path, "w")        
        transformed_data = df.copy()

        f.write('Number of partitions: ' + str(self.number_of_partitions)+ '\n')
        f.write('Number of attributes used in split: ' + str(self.number_of_split_attributes)+ '\n')
        f.write('Number of transformation attributes: ' + str(self.number_of_transformation_attributes)+ '\n')        
        f.write('Max number of conditions in a partition: ' + str(self.max_number_of_conditions_in_a_partition)+ '\n')
        f.write('Fraction of tuples transformed: ' + str(self.number_of_tuples_transformed/df.shape[0])+ '\n')
        
        f.write('---------------------------------------------------------------------------\n\n')
        num_altered_tuples = 0
        for ct in self.conditional_transformations:            
            f.write('Partition cardinality: ' + str(ct.partition.cardinality) + '\n')
            f.write('Partition condition: ' + str(ct.partition) + '\n')
            f.write('Partition transformation function: ' + str(ct.single_transformation) + '\n\n')
            num_altered_tuples += ct.partition.cardinality

            relevant_data = ct.partition.apply(df)
            transformed_values = ct.single_transformation.apply(relevant_data, self.noise_level)
            target_column_name = ct.single_transformation.target_attribute
            transformed_data.loc[relevant_data.index, target_column_name] = transformed_values
        
        f.close()
        self.fraction_of_tuples_transformed = num_altered_tuples/df.shape[0]

        # Save the transformation details as a pickle object
        with open(pickle_dump_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        if transformed_data_path is not None:
            # Save transformed data to a new CSV file
            transformed_data.to_csv(transformed_data_path, index=False)

        if create_plot and self.plottable:
            # Plotting only makes sense under this condition
            plt.figure(figsize=(12, 8)) 
            
            cmap = plt.get_cmap('tab20', self.number_of_partitions)
            colors = [cmap(i) for i in range(self.number_of_partitions)]
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'X', 'd', '|', '_', 'P', '+', '1', '2']

            i = 0
            for ct in self.conditional_transformations:  
                relevant_data = ct.partition.apply(transformed_data)
                x_values = relevant_data[ct.single_transformation.independent_attributes[0]]
                y_values = relevant_data[ct.single_transformation.target_attribute]                    

                plt.scatter(x = x_values, 
                            y = y_values, 
                            color = colors[i],
                            marker = markers[i],
                            s = 4)
                plt.xlabel(ct.single_transformation.independent_attributes[0])
                plt.ylabel(ct.single_transformation.target_attribute)
                i += 1
        
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        

class Partition:
    def __init__(self, conditions):
        self.conditions = conditions
        self.cardinality = None
        self.tuple_indices = None
    
    def __str__(self):
         return ' AND '.join([str(a[0]) + ' ' + str(a[1]) + ' ' + str(a[2]) for a in self.conditions])

    def process(self, df):
        self.tuple_indices = self.apply(df).index.tolist()
        self.cardinality = len(self.tuple_indices)

    def apply(self, df):
        cur_df = copy.deepcopy(df)
        for condition in self.conditions:
            cur_df = self.filter_dataframe(cur_df, condition[0], condition[1], condition[2])
        return cur_df

    def filter_dataframe(self, df, a, op, v):
        if is_numeric_dtype(df[a]):
            try:
                v = float(v)
            except:
                raise('Invalid data type')
        if op == '==':
            return df[df[a] == v]
        elif op == '!=':
            return df[df[a] != v]
        elif op == '<':
            return df[df[a] < v]
        elif op == '>':
            return df[df[a] > v]
        elif op == '<=':
            return df[df[a] <= v]
        elif op == '>=':
            return df[df[a] >= v]
        else:
            return df
    
    def merge(self, other):
        to_remove_1 = []
        to_remove_2 = []
        for c in self.conditions:
            for o in other.conditions:
                if c == flip(o):
                    to_remove_1.append(c)
                    to_remove_2.append(o)
        
        left = [c for c in self.conditions if c not in to_remove_1]
        right = [c for c in other.conditions if c not in to_remove_2]

        left.sort()
        right.sort()

        if left == right:
            return Partition(conditions=left)
        
        return None

class SingleTransformation:
    def __init__(self, target_attribute, independent_attributes, coefficients):
        self.target_attribute = target_attribute
        self.independent_attributes = []
        self.coefficients = []

        # Remove noise
        for i in range(len(coefficients) - 1):
            if np.abs(coefficients[i]) > EPS:
                self.coefficients.append(coefficients[i])
                self.independent_attributes.append(independent_attributes[i])     
        self.coefficients.append(coefficients[-1])              

    def __str__(self):
        return self.target_attribute + ' = ' + ' + '.join(
            [str(round(self.coefficients[c], 2)) + ' * ' + str(self.independent_attributes[c]) 
             for c in range(len(self.coefficients) - 1)]) + ' + ' + str(round(self.coefficients[-1], 2))

    def matches(self, other):
        if self.independent_attributes == other.independent_attributes and np.all(np.isclose(self.coefficients, other.coefficients, atol=EPS)):
            return True
    
    def apply(self, df, noise = 0):
        augmented_df = np.hstack((df[self.independent_attributes], np.ones((df[self.independent_attributes].shape[0], 1))))
        val = np.dot(augmented_df, self.coefficients)
        noise_to_add = np.random.uniform(low = -noise, high = noise, size = val.shape) 
        return val + noise_to_add * np.mean(val)

class ConditionalTransformation:
    def __init__(self, partition: Partition, single_transformation: SingleTransformation):
        self.partition = partition
        self.single_transformation = single_transformation