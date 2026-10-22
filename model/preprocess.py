import os
import pandas as pd
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.transformer import DataTransformer
import pickle
import numpy as np

def preprocess_data(raw_path='~~~',
        categorical_columns=[
            'purpose', 'home_ownership', 'sub_grade',
            'term_months', 'emp_length'
        ],
        log_columns=['annual_inc', 'revol_bal'],
        mixed_columns={  
            'dti': [0.0],
            'revol_util': [0.0],
            'bc_util': [0.0],
        },
        
        single_gaussian_columns=[],

        skew_multi_mode_columns=[
            
        ],
        integer_columns=['credit_history_years', 'fico_range_high'],
        problem_type={"Classification": 'loan_status'},
        test_ratio=0.2,
        save_path='./preprocess/preprocessed.csv',
        label_save_path='./preprocess/risk_labels.npy'):

    GENERATION_FEATURES = [
        "dti",
        "annual_inc",
        "installment",
        "revol_util",
        "mo_sin_old_rev_tl_op",
        "int_rate",
        "revol_bal",
        "bc_util",
        "credit_history_years",
        "fico_range_high",
        "sub_grade",
        "purpose",
        "home_ownership",
        "term_months",
        "emp_length"
    ]

    print("Loading and processing raw dataset")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Original train size: {len(df):,}")

    MODEL_COLUMNS = GENERATION_FEATURES + ["loan_status"]
    df = df[MODEL_COLUMNS].copy()

    print(f"Generation shape: {df.shape}")
    
    mixed_columns_combined = mixed_columns.copy()
    for col in skew_multi_mode_columns + single_gaussian_columns:
        mixed_columns_combined[col] = [0.0]  # mode candidate
        
    prep = DataPrep(
        raw_df=df,
        categorical=categorical_columns,
        log=log_columns,
        mixed=mixed_columns,
        integer=integer_columns,
        type=problem_type,
        test_ratio=test_ratio,
        skew_columns=skew_multi_mode_columns,
        single_gaussian_columns=single_gaussian_columns
    )

    transformed_df = prep.df

    transformer = DataTransformer(train_data=transformed_df,
                                  categorical_list=prep.column_types['categorical'],
                                  mixed_dict=prep.column_types['mixed'],
                                  skewed_list=prep.column_types['skewed'],
                                  gaussian_list=prep.column_types['gaussian'])
    transformer.fit()
    transformed = transformer.transform(transformed_df.values)

    risk_labels = np.asarray(prep.labels, dtype=np.float32).reshape(-1)
    if len(risk_labels) != len(transformed):
        raise ValueError(
            "Transformed feature rows and risk labels are not aligned: "
            f"{len(transformed)} != {len(risk_labels)}"
        )
    unique_labels = set(np.unique(risk_labels).tolist())
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            "loan_status must contain binary labels encoded as 0 (Fully Paid) "
            f"and 1 (Default); found {sorted(unique_labels)}"
        )

    for item in transformer.output_info:
        print(item)
        
    processed_dir = os.path.dirname(save_path)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)
    pd.DataFrame(transformed).to_csv(save_path, index=False)
    print(f"Saved processed data to {save_path}")

    label_dir = os.path.dirname(label_save_path)
    if label_dir:
        os.makedirs(label_dir, exist_ok=True)
    np.save(label_save_path, risk_labels)
    print(f"Saved risk labels to {label_save_path}")

    os.makedirs('./preprocess/transformer', exist_ok=True)
    with open('./preprocess/transformer/transformer.pkl', 'wb') as f:
        pickle.dump(transformer, f)
    print("Saved transformer to ./preprocess/transformer/transformer.pkl")

    os.makedirs('./preprocess/dataprep', exist_ok=True)
    with open('./preprocess/dataprep/dataprep.pkl', 'wb') as f:
        pickle.dump(prep, f)
    print("Saved DataPrep object to ./preprocess/dataprep.pkl")
