import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle

class Visualizer:
    def __init__(self, source_data_path, target_data_path_root, target_template, result_path_root, result_template, plots_path_root, detailed_plot_template, approach_name):
        self.source_data_path = source_data_path
        self.target_data_path_root = target_data_path_root
        self.target_template = target_template
        self.result_path_root = result_path_root
        self.result_template = result_template
        self.plots_path_root = plots_path_root
        self.detailed_plot_template = detailed_plot_template
        self.approach_name = approach_name
        self.source_df = pd.read_csv(self.source_data_path)
        

    def plot_summary(self, metric):
        results_df = pd.read_csv(os.path.join(self.result_path_root, 'result_summary_{}.csv'.format(metric)))
        metric = results_df.columns[-1]
        transformations = results_df['Transformation #'].apply(lambda x: f'Tr_{int(x)}')
        values = results_df[metric]
        
        plt.figure(figsize=(12, 8))    
        plt.bar(transformations, values, color='black')
        plt.title('Performance of in terms of ' + metric)
        plt.xlabel('Transformation #')
        plt.ylabel(metric)        
        plt.tight_layout()

        plt.savefig(os.path.join(self.plots_path_root, self.approach_name + '_' + metric+'_results.pdf'))
        plt.close()


    def plot_details(self):
        cmap = plt.get_cmap('tab20', 20)
        colors = [cmap(i) for i in range(20)]        
        
        num_transformations = len([f for f in os.listdir(self.target_data_path_root) if f.endswith('.pkl')])  
        for trans_id in range(1, num_transformations + 1):
            plt.figure(figsize=(12, 8))  
            independent_attribute = None
            target_attribute = None

            # Ground truth
            gt_transformation = pickle.load(open(os.path.join(self.target_data_path_root, self.target_template.format(trans_id)),'rb'))            
            independent_attribute = gt_transformation.conditional_transformations[0].single_transformation.independent_attributes[0]
            target_attribute = gt_transformation.conditional_transformations[0].single_transformation.target_attribute

            # Prediction from approach
            predicated_transformation = pickle.load(open(os.path.join(self.result_path_root, self.result_template.replace('txt', 'pkl').format(self.approach_name, trans_id)),'rb'))

            if not gt_transformation.plottable or not predicated_transformation.plottable:
                continue

            i = 0
            
            for ct in gt_transformation.conditional_transformations:
                relevant_data = ct.partition.apply(self.source_df)
                x_values = relevant_data[independent_attribute]
                y_values = ct.single_transformation.apply(relevant_data)
                
                plt.scatter(x_values, 
                         y_values, 
                         color = colors[i%20],
                         marker = 's',
                         label = 'GT => ('+ str(len(x_values)) +') | Cond:' + str(ct.partition) + ' | Trans: ' + str(ct.single_transformation),
                         s = 4)
                i += 1              

            i = 0
            for ct in predicated_transformation.conditional_transformations:
                if ct.single_transformation.independent_attributes[0] != independent_attribute:
                    continue

                relevant_data = ct.partition.apply(self.source_df)
                x_values = relevant_data[independent_attribute]
                y_values = ct.single_transformation.apply(relevant_data)                
                               
                plt.plot(x_values,
                         y_values, 
                         '-',
                         color = colors[i%20],
                         label = 'LT => ('+ str(len(x_values)) +') | Cond:' + str(ct.partition) + ' | Trans: ' + str(ct.single_transformation),
                         markersize = 4,
                         alpha = 0.5)
                i += 1            

            plt.legend()
            plt.xlabel(independent_attribute)
            plt.ylabel(target_attribute)  
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path_root, self.detailed_plot_template.format(self.approach_name, trans_id)))
            plt.close()


   

