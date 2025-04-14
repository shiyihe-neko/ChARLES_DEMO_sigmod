import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
import os
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np

# Get all CSV files in the temp_transformations folder
#csv_files = glob.glob('../temp_transformations/target_data_*.csv')

def filter_attributes(target_df,transformation_index, numerical_attrs,categorical_attrs):

    # Extract the index from the filename
    #file_name = os.path.basename(file_path)
    #index = file_name.split('_')[-1].split('.')[0]

    # Load the data
    data = target_df.copy()

    # Create the output directory if it doesn't exist
    par_dir = 'ensemble/feature_selection/'
    output_dir = os.path.join(par_dir,f'Charles_Supplementary_folder_{transformation_index}')
    os.makedirs(output_dir, exist_ok=True)

    # A) Identify Numerical Attributes and Calculate Pearson Correlation
    numerical_attributes = numerical_attrs.copy()
    target_variable = 'base_bonus' ##################

    # Calculate Pearson Correlation
    correlations = data[numerical_attributes + [target_variable]].corr()[target_variable].drop(target_variable)
    filtered_numerical_attributes = correlations[abs(correlations) >= 0.10].index.tolist()
    # If no attribute has |correlation| >= 0.05, pick the attribute with the highest correlation value
    if not filtered_numerical_attributes:
        filtered_numerical_attributes = [correlations.abs().idxmax()]
    # Sort the filtered attributes by absolute correlation value in descending order
    filtered_numerical_attributes = sorted(filtered_numerical_attributes, key=lambda x: abs(correlations[x]), reverse=True)

# Select the top 50% of the numerical attributes
    #num_attributes_to_select = max(1, len(numerical_attributes)//2) # RELAXED AS OF NOW
    num_attributes_to_select = max(1, len(filtered_numerical_attributes))
    filtered_numerical_attributes = filtered_numerical_attributes[:num_attributes_to_select]

    # Save the correlation results
    #correlations.to_csv(f'{output_dir}/numerical_correlations.csv')
    filtered_numerical_attributes_df = pd.DataFrame({
    'Filtered_Numerical_Attributes': filtered_numerical_attributes,
    'Correlation_Value': correlations[filtered_numerical_attributes].values})

    filtered_numerical_attributes_df.to_csv(f'{output_dir}/filtered_numerical_attributes_{transformation_index}.csv', index=False)

    categorical_attributes = categorical_attrs.copy()
   # B ANOVA for categorical indep var and numerical dep var
    anova_results = []
    for cat_attr in categorical_attributes:
        groups = [data[data[cat_attr] == category][target_variable].dropna() for category in data[cat_attr].unique()]
        f_val, p_val = f_oneway(*groups)
        anova_results.append({'Attribute': cat_attr, 'F-value': f_val, 'p-value': p_val})

    # Save the ANOVA test results
    anova_results_df = pd.DataFrame(anova_results)
    anova_results_df = anova_results_df.sort_values(by='p-value')
    anova_results_df.to_csv(f'{output_dir}/anova_results_{transformation_index}.csv', index=False)

    categ_attributes_to_select = max(1, len(categorical_attributes) // 2)
    # Select attributes based on ANOVA and Random Forest
    selected_attributes = set(anova_results_df[anova_results_df['p-value'] <= 0.10]['Attribute'][:categ_attributes_to_select])
    
    '''
    # C) Calculate Value Counts and Filter Rare Categories
    threshold_percentage = 5
    total_records = len(data)

    for cat_attr in categorical_attributes:
        value_counts = data[cat_attr].value_counts(normalize=True) * 100
        rare_categories = value_counts[value_counts < threshold_percentage].index.tolist()
        data[cat_attr] = data[cat_attr].apply(lambda x: 'Other' if x in rare_categories else x)

    # Save the modified data with grouped rare categories
    #data.to_csv(f'{output_dir}/modified_data_with_grouped_rare_categories.csv', index=False)
    '''

# Encode categorical variables
    label_encoders = {}
    for cat_attr in categorical_attributes:
        le = LabelEncoder()
        data[cat_attr] = le.fit_transform(data[cat_attr])
        label_encoders[cat_attr] = le

    # Prepare data for Random Forest
    X = data[categorical_attributes]
    y = data[target_variable]

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    feature_importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': categorical_attributes,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    #print(data['Length_of_Service'].value_counts())

    # Calculate Root Mean Squared Error (RMSE)
    y_pred = rf.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    rmse_df = pd.DataFrame({'Feature': ['RMSE'], 'Importance': [rmse]})
    feature_importance_df = pd.concat([feature_importance_df, rmse_df], ignore_index=True)
    # Save feature importance
    feature_importance_df.to_csv(f'{output_dir}/feature_importance_{transformation_index}.csv', index=False)

    # Select top 1 features from Random Forest
    top_rf_features = set(feature_importance_df.head(1)['Feature'])

    # Combine selected attributes from ANOVA and Random Forest
    final_selected_attributes = selected_attributes.union(top_rf_features)

    fin_split_attr_df = pd.DataFrame({'split_attributes':list(final_selected_attributes)})
    fin_split_attr_df.to_csv(f'{output_dir}/fin_split_attr_{transformation_index}.csv', index=False)

    return list(final_selected_attributes), filtered_numerical_attributes