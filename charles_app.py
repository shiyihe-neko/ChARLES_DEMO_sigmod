import streamlit as st
import pandas as pd
import random
from streamlit.components.v1 import html
import plotly.express as px
import subprocess
import pandas as pd
import os
import re
import plotly.express as px
from textwrap import wrap
import sys


st.set_page_config(layout="wide")
st.markdown("""
<style>
    .stMain .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    /* element margin */
    .stButton, .stSlider, div[data-testid="stFileUploader"] {
        margin-bottom: 0.5rem;
    }
    /* top title margin */
    h1, h2, h3 {
        margin-top: 0.5rem !important;
    }
    /* file loader size */
    div[data-testid="stFileUploader"] {
        padding: 0.2rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* upload button size and margin */
    div[data-testid="stFileUploader"] button {
        padding: 0.1rem 0.5rem !important;
        font-size: 0.8rem !important;
        height: auto !important;
        min-height: 0 !important;
    }

    /* file loader element margin */
    div[data-testid="stFileUploader"] > div {
        gap: 0.2rem !important;
        padding: 0.2rem !important;
    }

    /* file loader label size */
    div[data-testid="stFileUploader"] label {
        font-size: 0.9rem !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    /* file info size */
    div[data-testid="stFileUploader"] p {
        font-size: 0.8rem !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] {
        padding-top: 0 !important;
    }

    /* padding */
    section[data-testid="stFileUploaderDropzone"] {
        padding: 0.5rem !important;
    }

    /* file loader background and frame  */
    div[data-testid="stFileUploader"] > div:first-child {
        border: 1px dashed #ccc !important;
        border-radius: 4px !important;
        background-color: #f8f9fa !important;
    }

    /* data editor margin */
    div[data-testid="stDataFrame"] {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }
            
    /* col1 height */
    div[data-testid="column"]:nth-of-type(1) > div {
        max-height: 250px;
        overflow-y: auto;
        padding-right: 8px;
    }
            
</style>
""", unsafe_allow_html=True)


if 'is_confirmed' not in st.session_state:
    st.session_state.is_confirmed = False
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False

def on_slider_change():
    st.session_state.is_confirmed = False
    st.session_state.show_comparison = False

st.title("ChARLES App")

col1, col2, col3 = st.columns([4, 3, 3])


def format_summary(summary_text):
    parts = summary_text.split('→')
    formatted_blocks = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if 'base_bonus =' in part:
            condition, equation = part.split('base_bonus =', 1)
            formatted_blocks.append(
                f"""
                <div style="margin-bottom: 0.8em; font-family: monospace; font-size: 0.9rem;">
                    <div style="margin-left: 0.5em;">{condition.strip()}</div>
                    <div style="margin-left: 2em;">base_bonus = {equation.strip()}</div>
                </div>
                """
            )
        else:
            formatted_blocks.append(
                f"""
                <div style="margin-bottom: 0.8em; font-family: monospace; font-size: 0.9rem;">
                    <div style="margin-left: 0.5em;">{part}</div>
                </div>
                """
            )

    return "<div>" + "".join(formatted_blocks) + "</div>"


def render_summary_row(idx, row, colors):
    cols = st.columns([0.5, 3, 0.8, 1, 1, 1])
    bg_color = "#f9f9f9" if idx % 2 == 0 else "#ffffff"

    with cols[0]:
        st.markdown(f"**{idx + 1}**")

    with cols[1]:
        lines = row['cts'].split('\n')
        for i, line in enumerate(lines):
            color = colors[i % len(colors)]
            st.markdown(f":{color}[{line}]")

    with cols[2]:
        if st.button("Details", key=f"details_{idx}"):
            st.session_state.selected_cardinality = row["cardinality"]
            st.session_state.selected_summary = row["cts"]
            st.session_state.selected_row = row

    with cols[3]:
        st.write(row["Score"])

    with cols[4]:
        st.write(row["Accuracy"])

    with cols[5]:
        st.write(row["Interpretability"])


def load_metrics_summary():
    base_dir = "ensemble_test/temp_results_RF_LINEAR_TREE"
    dataset_dirs = [d for d in os.listdir(base_dir) if d.startswith("dataset_")]
    summaries = []
    for dataset in dataset_dirs:
        file_path = os.path.join(base_dir, dataset, f"metrics_summary_sorted_scores_CTs{dataset.split('_')[-1]}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            summaries.append(df)
    if summaries:
        return pd.concat(summaries, ignore_index=True)
    else:
        raise FileNotFoundError("No metrics_summary_sorted_scores_CTsX.csv files found.")

def run_test_script(n, m, alpha):

    script_path = os.path.join("test", "test_evaluation_RF.py")   
    st.write("Current working directory:", os.getcwd()) 

    result = subprocess.run(
        [sys.executable, script_path, str(n), str(m), str(alpha)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        st.error(f"run test_evaluation_RF.py wrong:\n{result.stderr}")
    else:
        st.success("Model evaluation completed successfully.")


def filter_attributes(target_df, numerical_attrs, categorical_attrs, target_variable='base_bonus'):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import f_oneway

    data = target_df.dropna(subset=[target_variable]).copy()

    # Pearson
    correlations = data[numerical_attrs + [target_variable]].corr()[target_variable].drop(target_variable)
    numerical_scores = correlations.abs().dropna().sort_values(ascending=False)

    # ANOVA + RF
    anova_scores = {}
    for cat in categorical_attrs:
        try:
            groups = [data[data[cat] == v][target_variable] for v in data[cat].dropna().unique()]
            if len(groups) > 1:
                f_val, p_val = f_oneway(*groups)
                anova_scores[cat] = 1 - p_val  
        except:
            continue

    # Random Forest for categorical
    rf_df = data[categorical_attrs].copy()
    label_encoders = {}
    for col in rf_df.columns:
        le = LabelEncoder()
        try:
            rf_df[col] = le.fit_transform(rf_df[col])
        except:
            rf_df[col] = 0

    rf_scores = {}
    try:
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(rf_df, data[target_variable])
        for i, col in enumerate(rf_df.columns):
            rf_scores[col] = rf.feature_importances_[i]
    except:
        rf_scores = {}

    categorical_scores = {}
    for attr in set(anova_scores) | set(rf_scores):
        categorical_scores[attr] = max(anova_scores.get(attr, 0), rf_scores.get(attr, 0))

    return numerical_scores, categorical_scores



with col1:
    upload_col1, upload_col2 = st.columns(2)

    with upload_col1:
        source_file = st.file_uploader("Source CSV", type=["csv"], key="source_file_uploader")
    
    with upload_col2:
        target_file = st.file_uploader("Target CSV", type=["csv"], key="target_file_uploader")

    if source_file and target_file:
        source_df = pd.read_csv(source_file)
        target_df = pd.read_csv(target_file)
        exclude_columns = ['Employee_id'] 
        filtered_columns = sorted(
            [col for col in source_df.columns if col not in exclude_columns],
            key=lambda x: (
                (0 if x[0].isdigit() else 1, x[0].upper() if x[0].isalpha() else x[0])
            )
        )
        source_df_filtered = source_df[filtered_columns]
        numberic_columns = sorted(
            [col for col in source_df_filtered.select_dtypes(include=["number"]).columns],
            key=lambda x: (
                (0 if x[0].isdigit() else 1, x[0].upper() if x[0].isalpha() else x[0])
            )
        )
        total_columns = len(filtered_columns)
        total_numeric_columns = len(numberic_columns)
        print(f"Total columns: {total_columns}, Numeric columns: {total_numeric_columns}")

        if set(source_df.columns) == set(target_df.columns):
            changed_attributes = [
                col for col in source_df.columns
                if not source_df[col].equals(target_df[col])
            ]

            if changed_attributes:
                st.markdown("#### Select Target Attributes")

                df_changed = pd.DataFrame({
                    "Attribute": changed_attributes,
                    "Selected": [False] * len(changed_attributes)
                })

                edited_changed_df = st.data_editor(
                    df_changed,
                    num_rows="dynamic",
                    height=min(len(df_changed) * 35 + 40, 300),  
                    use_container_width=True
                )

                slider_col1, slider_col2, button_col = st.columns([4, 4, 2])

                with slider_col1:
                    condition_max = st.session_state.get('total_condition', total_columns)
                    condition = st.slider("Condition", min_value=0, max_value=condition_max, value=min(3, condition_max), key="condition_slider", on_change=on_slider_change)

                with slider_col2:
                    transformation_max = st.session_state.get('total_transformation', total_numeric_columns)
                    transformation = st.slider("Transformation", min_value=0, max_value=transformation_max, value=min(3, transformation_max), key="transformation_slider", on_change=on_slider_change)

                with button_col:
                    st.markdown("""
                    <style>
                    div[data-testid="stButton"] > button {
                        margin-top: 10px;
                        width: 100%;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                if st.button("Confirm", key="confirm_button"):
                    selected_cols = edited_changed_df[edited_changed_df["Selected"] == True]["Attribute"].tolist()
                    st.session_state.is_confirmed = True

                    target_variable = 'base_bonus'

                    numerical_attrs = [col for col in source_df.select_dtypes(include=["number"]).columns if col != target_variable]
                    categorical_attrs = [col for col in source_df.select_dtypes(include=["object", "category"]).columns]


                    numerical_scores, categorical_scores = filter_attributes(
                        target_df=target_df,
                        numerical_attrs=numerical_attrs,
                        categorical_attrs=categorical_attrs,
                        target_variable=target_variable
                    )

                    all_attr_scores = {}
                    all_attr_scores.update(numerical_scores.to_dict())
                    all_attr_scores.update(categorical_scores)

                    sorted_all_attrs = sorted(all_attr_scores, key=lambda x: all_attr_scores[x], reverse=True)

                    sorted_numerical = list(numerical_scores.index)

                    condition_attrs = sorted_all_attrs[:3]
                    transform_attrs = sorted_numerical[:3]

                    st.session_state.total_condition = len(sorted_all_attrs)
                    st.session_state.total_transformation = len(sorted_numerical)

                    condition_n = st.session_state.get("condition_slider", 3)
                    transformation_n = st.session_state.get("transformation_slider", 3)

                    st.session_state.total_condition = len(sorted_all_attrs)
                    st.session_state.total_transformation = len(sorted_numerical)


                    st.session_state.condition_df = pd.DataFrame({
                        "Attribute": sorted_all_attrs,
                        "Selected": [True] * len(sorted_all_attrs)
                    })

                    st.session_state.transform_df = pd.DataFrame({
                        "Attribute": sorted_numerical,
                        "Selected": [True] * len(sorted_numerical)
                    })


            else:
                st.info("no change in attributes between source and target CSV files.")
        else:
            st.error("The source and target CSV files have different columns. Please check the files.")

with col2:
    if st.session_state.is_confirmed:
        if 'condition_df' not in st.session_state or not isinstance(st.session_state.condition_df, pd.DataFrame):
            condition_attrs = filtered_columns[:min(condition, len(filtered_columns))]
            st.session_state.condition_df = pd.DataFrame({
                "Attribute": filtered_columns,
                "Selected": [attr in condition_attrs for attr in filtered_columns]
            })

        if 'transform_df' not in st.session_state or not isinstance(st.session_state.transform_df, pd.DataFrame):
            transform_attrs = numberic_columns[:min(transformation, len(numberic_columns))]
            st.session_state.transform_df = pd.DataFrame({
                "Attribute": numberic_columns,
                "Selected": [attr in transform_attrs for attr in numberic_columns]
            })

        def update_condition_df():
            if isinstance(st.session_state.condition_editor, pd.DataFrame):
                st.session_state.condition_df = st.session_state.condition_editor

        def update_transform_df():
            if isinstance(st.session_state.transform_editor, pd.DataFrame):
                st.session_state.transform_df = st.session_state.transform_editor

        st.write("**Attributes for Condition:**")
        try:
            edited_condition_df = st.data_editor(
                st.session_state.condition_df,
                num_rows=len(filtered_columns),
                height=210,
                use_container_width=True,
                key="condition_editor",
                on_change=update_condition_df
            )
        except Exception as e:
            st.error(f"Error displaying condition editor: {e}")

        st.write("**Attributes for Transformation:**")
        try:
            edited_transform_df = st.data_editor(
                st.session_state.transform_df,
                num_rows=len(numberic_columns),
                height=210,
                use_container_width=True,
                key="transform_editor",
                on_change=update_transform_df
            )
        except Exception as e:
            st.error(f"Error displaying transform editor: {e}")

        if isinstance(edited_condition_df, pd.DataFrame):
            st.session_state.selected_condition_attrs = edited_condition_df[edited_condition_df["Selected"]]["Attribute"].tolist()
        else:
            st.session_state.selected_condition_attrs = []

        if isinstance(edited_transform_df, pd.DataFrame):
            st.session_state.selected_transform_attrs = edited_transform_df[edited_transform_df["Selected"]]["Attribute"].tolist()
        else:
            st.session_state.selected_transform_attrs = []

with col3:
    if st.session_state.is_confirmed:
        st.markdown("### Set Parameter Setting")

        weight = st.slider("Weight of Accuracy(α)", min_value=0.0, max_value=1.0, value=0.5, key="weight_slider")

        if st.button("Compare", key="compare_button"):
            st.session_state.show_comparison = True

            n = st.session_state.condition_slider
            m = st.session_state.transformation_slider
            alpha = st.session_state.weight_slider

            run_test_script(n, m, alpha)

if st.session_state.show_comparison:
    st.markdown("---")
    st.markdown("### Result Summary")

    result_col1, result_col2 = st.columns([6, 4])
    with result_col1:
        summary_df = load_metrics_summary()

        max_rows = len(summary_df)
        num_rows_to_show = st.selectbox(
            "Show Top Summaries",
            options=list(range(1, max_rows + 1)),
            index=min(9, max_rows - 1),
            key="num_rows_selector"
        )
        st.markdown("""
        <style>
        #scroll-container {
            max-height: 420px;
            overflow-y: auto;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-bottom: 8px;
        }
        .summary-header, .summary-row {
            display: flex;
            padding: 6px 8px;
            font-size: 0.85rem;
        }
        .summary-header {
            font-weight: bold;
            background-color: #f0f0f0;
            border-bottom: 1px solid #ccc;
        }
        .summary-row:nth-child(even) {
            background-color: #fafafa;
        }
        .summary-row:nth-child(odd) {
            background-color: #ffffff;
        }
        .cell {
            flex: 1;
            padding: 4px;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        .button-cell {
            flex: 0.7;
            display: flex;
            align-items: center;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="summary-header">
            <div class="cell" style="flex: 0.5;">No.</div>
            <div class="cell" style="flex: 3;">Summary</div>
            <div class="cell" style="flex: 0.8;">Details</div>
            <div class="cell">Score</div>
            <div class="cell">Accuracy</div>
            <div class="cell">Interpretability</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div id="scroll-container">', unsafe_allow_html=True)
        with st.container():
            for idx, row in summary_df.head(num_rows_to_show).iterrows():
                cols = st.columns([0.5, 3, 0.8, 1, 1, 1])
                with cols[0]:
                    st.markdown(f"{idx + 1}")
                with cols[1]:
                    lines = row['cts'].split('\n')
                    colors = ['red', 'blue', 'green', 'orange', 'purple']
                    for i, line in enumerate(lines):
                        color = colors[i % len(colors)]
                        st.markdown(f"<div class='cell' style='color: {color};'>{line}</div>", unsafe_allow_html=True)
                with cols[2]:
                    if st.button("Details", key=f"details_{idx}"):
                        st.session_state.selected_cardinality = row["cardinality"]
                        st.session_state.selected_summary = row["cts"]
                        st.session_state.selected_row = row
                with cols[3]:
                    st.markdown(f"{row['Score']}")
                with cols[4]:
                    st.markdown(f"{row['Accuracy']}")
                with cols[5]:
                    st.markdown(f"{row['Interpretability']}")
                st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

        query_params = st.experimental_get_query_params()
        if "details" in query_params:
            selected_idx = int(query_params["details"][0])
            selected_row = summary_df.iloc[selected_idx]
            st.session_state.selected_cardinality = selected_row['cardinality']
            st.session_state.selected_summary = selected_row['Summary']
            st.session_state.selected_row = selected_row


    with result_col2:
        if "selected_cardinality" in st.session_state:
            st.markdown("### Details of Selected Summary")
            st.markdown("")

            cardinalities = list(map(int, str(st.session_state.selected_cardinality).split(',')))
            labels = [f"Group {i+1}" for i in range(len(cardinalities))]
            color_sequence = ['red', 'blue', 'green', 'orange', 'purple']
            group_colors = color_sequence[:len(labels)]

            summary_lines = st.session_state.selected_summary.strip().split('\n')

            if len(summary_lines) < len(cardinalities):
                summary_lines += [''] * (len(cardinalities) - len(summary_lines))
            else:
                summary_lines = summary_lines[:len(cardinalities)]

            def highlight_keywords(text):
                import re
                keywords = ['->']
                for kw in sorted(keywords, key=len, reverse=True):
                    text = re.sub(fr'(?<!\w)({re.escape(kw)})(?!\w)', r"<b>\1</b>", text)
                return text

            summary_highlighted = [highlight_keywords(line) for line in summary_lines]

            hover_data = [[line, c] for line, c in zip(summary_highlighted, cardinalities)]

            import pandas as pd
            df_treemap = pd.DataFrame({
                'Label': labels,
                'Parent': ["" for _ in labels],
                'Value': cardinalities,
                'ColorGroup': labels
            })

            import plotly.express as px
            fig = px.treemap(
                df_treemap,
                names='Label',
                parents='Parent',
                values='Value',
                color='ColorGroup',
                color_discrete_sequence=group_colors,
            )

            fig.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br><br>"
                    "<span style='white-space:pre-wrap; font-size:0.75rem;'>"
                    "<b>Summary:</b><br>%{customdata[0]}</span><br><br>"
                    "<span style='font-size:0.75rem;'><b>Cardinality:</b> %{customdata[1]}</span>"
                    "<extra></extra>"
                ),
                customdata=hover_data,
                textinfo="label+percent entry",
                root_color="lightgray",
                branchvalues="total",
                pathbar_visible=False,
                level="",           
                maxdepth=-1,
                insidetextfont=dict(size=14)  
            )


            fig.update_layout(
                margin=dict(t=10, l=0, r=0, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                uniformtext=dict(minsize=10, mode='hide')
            )
            st.plotly_chart(fig, use_container_width=True)
