# Risk-VAE-CTABGAN

Risk-VAE-CTABGAN is a tabular data synthesis framework that combines risk-aware Variational Autoencoder (VAE) representation learning with CTABGAN-based synthetic minority generation. A supervised **Risk head** learns repayment-risk information from the encoder's posterior mean using both Fully Paid and Default borrowers. Latent vectors sampled from actual Default borrowers are then used to generate synthetic Default observations.

The framework shares the CTABGAN generator as the VAE decoder and supports continuous, categorical, and mixed-type borrower attributes. This repository implements **data preprocessing**, **Risk-VAE warm-up**, **joint adversarial training**, and **synthetic Default sample generation** for imbalanced credit-risk learning.

---

## Project Structure

```text
.
├── .gitignore
├── README.md
├── main.py
├── requirements.txt
└── model/
    ├── __init__.py
    ├── preprocess.py
    ├── train_loop.py
    ├── model.py
    ├── sampler.py
    ├── condvec.py
    ├── pipeline/
    │   ├── data_utils.py
    │   ├── data_preparation.py
    │   └── inverse_transform.py
    └── synthesizer/
        ├── __init__.py
        └── transformer.py
```

## Installation

Create a Python 3.12 environment and install the packages used by the implementation:

```bash
conda create -n risk-vae-ctabgan python=3.12 -y
conda activate risk-vae-ctabgan
python -m pip install -r requirements.txt
```

Dependency versions are pinned in `requirements.txt`, including PyTorch 2.3.1 and torchvision 0.18.1. The paper reports Python 3.12 and PyTorch 2.3.1.

Training displays progress bars and an epoch summary of losses and repayment-risk metrics in the terminal.

Run all subsequent commands from the project root. CUDA is selected automatically when available; otherwise, the implementation uses the CPU.

## Usage

```bash
python main.py --mode {preprocess,train,generate,only_train}
```

The repository supports four execution modes:

- `preprocess`: preprocess a training CSV and save features, labels, and transformation objects.
- `train`: run preprocessing and then train a new model.
- `only_train`: train a new model using existing preprocessing artifacts.
- `generate`: load a trained checkpoint and generate synthetic Default observations.

Before using the short commands below, configure `--dataset_path` for preprocessing/training and `--sample_dir` for output, or update their defaults in `main.py`. Personal paths have been replaced with `~~~` placeholders; replace them with your own paths or pass the arguments explicitly. The [Quick Start](#quick-start) provides examples with relative paths.

### Mode 1: Preprocess Only

Use this mode to preprocess the input training dataset without training a model:

```bash
python main.py --mode preprocess --dataset_path dataset/train.csv
```

This step saves the transformed feature CSV, the separate repayment-risk labels, and the fitted transformer and data-preparation objects. Keep these artifacts together because both training and generation use them.

### Mode 2: Train

Use this mode to preprocess the dataset and train Risk-VAE-CTABGAN:

```bash
python main.py --mode train --latent_dim 64 --batch_size 512 --epochs 150 --encoder_freeze_epoch 50
```

The first 50 epochs perform Risk-VAE warm-up. The remaining epochs perform adversarial Default generation with Risk-VAE fine-tuning. Append `--dataset_path dataset/train.csv` to select your training CSV.

### Mode 3: Only Train

Use this mode when preprocessing has already been completed:

```bash
python main.py --mode only_train --latent_dim 64 --batch_size 512 --epochs 150 --encoder_freeze_epoch 50
```

The following artifacts must already exist:

```text
preprocess/preprocessed.csv
preprocess/risk_labels.npy
preprocess/transformer/transformer.pkl
preprocess/dataprep/dataprep.pkl
```

`only_train` starts a new training run using these artifacts; it does not resume a saved checkpoint.

### Mode 4: Generate Synthetic Samples

Use this mode to generate synthetic Default observations from the trained encoder and generator:

```bash
python main.py --mode generate --latent_dim 64 --batch_size 512 --checkpoint_dir checkpoints
```

By default, this loads `checkpoints/risk_vae_ctabgan.pt` and generates **100 observations**. Set `--num_samples` to the required augmentation size and `--sample_dir` to the output directory:

```bash
python main.py --mode generate --latent_dim 64 --batch_size 512 --checkpoint_dir checkpoints --save_name risk_vae_ctabgan.pt --num_samples 478041 --sample_dir samples
```

The generation mode also requires the original preprocessing artifacts: it selects actual Default borrowers from the saved features and labels as latent seeds. Use the same `--latent_dim` and transformation artifacts as the training run.

## Quick Start

A typical workflow separates preprocessing, training, and generation:

```bash
# 1. Preprocess the training CSV containing both repayment classes.
python main.py --mode preprocess --dataset_path dataset/train.csv --sample_dir samples

# 2. Train Risk-VAE-CTABGAN using the saved artifacts.
python main.py --mode only_train --latent_dim 64 --batch_size 512 --epochs 150 --encoder_freeze_epoch 50 --sample_dir samples

# 3. Generate synthetic Default observations from the final checkpoint.
python main.py --mode generate --latent_dim 64 --batch_size 512 --checkpoint_dir checkpoints --num_samples 478041 --sample_dir samples
```

Alternatively, combine preprocessing and training:

```bash
python main.py --mode train --dataset_path dataset/train.csv --latent_dim 64 --batch_size 512 --epochs 150 --encoder_freeze_epoch 50 --sample_dir samples
```

The example augmentation size, 478,041, corresponds to the paper's original training counts of 597,611 Fully Paid and 119,570 Default loans. For another dataset, choose the synthetic count from the class counts of the original training set used by the downstream classifier.

## Input and Output Artifacts

The input is a prepared LendingClub training CSV with the 15 features listed below and a binary `loan_status` column:

- `0`: Fully Paid.
- `1`: Default, including Charged Off loans mapped to this class during source-data preparation.

The CSV must contain both classes for supervised Risk-VAE learning. The script expects numeric 0/1 labels and does not convert raw LendingClub repayment-status strings automatically.

With the default preprocessing/checkpoint paths and `--sample_dir samples`, the output structure is:

```text
preprocess/
├── preprocessed.csv
├── risk_labels.npy
├── transformer/
│   └── transformer.pkl
└── dataprep/
    └── dataprep.pkl

checkpoints/
├── risk_vae_ctabgan.pt
└── final/
    └── risk_vae_ctabgan_epoch150.pth

samples/
└── risk-vae-ctabgan.csv
```

| Artifact | Description |
|---|---|
| `preprocessed.csv` | Transformed borrower features for the retained training observations; excludes `loan_status`. |
| `risk_labels.npy` | Aligned 0/1 labels used by the Risk head and to select Default seeds. |
| `transformer.pkl` | Fitted tabular transformer used for feature encoding and inverse transformation. |
| `dataprep.pkl` | Feature names, categorical encoders, and other preprocessing metadata. |
| `risk_vae_ctabgan.pt` | Final checkpoint loaded by the default generation command. |
| `risk_vae_ctabgan_epoch150.pth` | Epoch-stamped final checkpoint; the epoch number follows `--epochs`. |
| `risk-vae-ctabgan.csv` | Generated borrower features with `loan_status=1` appended. |

The final checkpoint stores the encoder, Risk head, generator, and discriminator state dictionaries. Generation uses the encoder and generator.

The implementation also writes periodic `risk_vae_ctabgan_epoch_{epoch}.pt` checkpoints to `--checkpoint_dir`, starting at epoch 60 and then every 20 epochs. These use the same model state-dictionary keys as the final checkpoint and include the epoch number. To generate from a periodic checkpoint, set its filename with `--save_name`, for example `--save_name risk_vae_ctabgan_epoch_100.pt`.

Preprocessing currently saves `transformer.pkl` and `dataprep.pkl` to the fixed paths shown above. Keep these defaults for the workflow above; changing the corresponding CLI loading paths does not change where preprocessing writes those two files.

## Important Notes on Data Leakage

Prepare the training/test split before fitting the generative model. Fit preprocessing and Risk-VAE-CTABGAN on training observations, and use actual held-out loans for screening and investment evaluation.

The generator uses borrower information and loan terms available at origination. Post-outcome variables should be excluded from predictive inputs, including:

- `total_pymnt` and `total_pymnt_inv`.
- `last_fico_*`.
- `debt_settlement_flag`.
- `recoveries` and `collection_recovery_fee`.
- `last_pymnt_d`.
- `out_prncp` and `out_prncp_inv`.
- Variables directly derived from realized repayment outcomes.

`model/preprocess.py` selects the 15 input features listed in [Data Description](#data-description) before fitting the transformer. `debt_settlement_flag` is absent from this feature list, so it is discarded even when the input CSV contains that column. Its mention above documents an excluded variable.

`loan_status` is kept separately as the supervised target for the Risk head. It is excluded from encoder input features and appended as `1` to generated Default observations. CTABGAN's conditional vectors are constructed from transformed feature categories and mode indicators.

The current `preprocess_data()` default applies an additional stratified split with `test_ratio=0.2` and retains 80% of its input. If an externally prepared training set must be used in full, change this setting to `0.0` in `model/preprocess.py` before preprocessing; it is not exposed as a CLI argument. Perform source-data filtering and missing-row removal before this step when required by the experimental protocol.

## Downstream Use in the Paper

In the study, synthetic Default observations augment the training data for the credit-risk screening classifier. The profitability model is trained separately on the original training data. Candidate-set formation and investment evaluation use actual test loans.

```text
Risk-VAE-CTABGAN synthetic Default generation
        ↓
Original training loans + synthetic Default observations
        ↓
Credit-risk screening classifier
        ↓
Actual test loans predicted as Fully Paid
        ↓
Common profitability model trained on original loans
        ↓
Top 10%, 20%, and 30% investment selection
```

The generator implementation in this repository does not include the downstream screening, profitability, or investment-evaluation scripts.

## Dataset

The experiments use **LendingClub loan records** for imbalanced credit-risk learning and synthetic tabular data generation.

- Dataset source: LendingClub loan-level records.
- Public dataset reference: [Lending Club 2007–2020Q3 on Kaggle](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1).
- Label convention: Fully Paid = 0; Default/Charged Off = 1.
- Task: risk-aware synthetic Default generation for subsequent credit-risk screening.

Prepare an experiment-specific training CSV from the source records, including the required repayment-status mapping, observation filters, derived features, and training/test split. The model's preprocessing command expects this prepared CSV rather than the unmodified source download.

### Data Description

<img src="https://github.com/user-attachments/assets/4a0f4c30-bf83-495c-aa3d-78bfdba779df"  width="851" height="812" alt="Image"/>
The selected inputs consist of 10 continuous variables and 5 categorical variables. They describe borrower creditworthiness, repayment capacity, and loan terms available at origination.

| Type | Feature | Description |
|---|---|---|
| Continuous | `dti` | Debt-to-income ratio. |
| Continuous | `annual_inc` | Annual income. |
| Continuous | `installment` | Monthly installment amount. |
| Continuous | `revol_util` | Revolving credit utilization. |
| Continuous | `mo_sin_old_rev_tl_op` | Months since the oldest revolving account was opened. |
| Continuous | `int_rate` | Loan interest rate. |
| Continuous | `revol_bal` | Revolving credit balance. |
| Continuous | `bc_util` | Bankcard credit utilization. |
| Continuous | `credit_history_years` | Credit history length in years. |
| Continuous | `fico_range_high` | Upper bound of the FICO score range. |
| Categorical | `sub_grade` | LendingClub loan subgrade. |
| Categorical | `purpose` | Loan purpose. |
| Categorical | `home_ownership` | Home ownership status. |
| Categorical | `term_months` | Loan term in months. |
| Categorical | `emp_length` | Employment length. |

The implementation log-transforms `annual_inc` and `revol_bal`, handles zero-valued components of `dti`, `revol_util`, and `bc_util` as mixed modes, and rounds generated `credit_history_years` and `fico_range_high` values to integers during recovery. The transformer uses mixture-based normalization and mode indicators for numeric attributes, and one-hot spans for categorical attributes, including `emp_length`.

## Model Architecture
<img width="893" height="555" alt="Image" src="https://github.com/user-attachments/assets/ca76303f-fc42-4eba-ab77-4b8eacb39c05" />
Risk-VAE-CTABGAN consists of feature transformation, Stage 1 Risk-VAE warm-up, and Stage 2 risk-aware adversarial minority generation.

```text
Prepared training loans: Fully Paid + Default
        ↓
CTABGAN feature transformation
        ↓
VAE encoder → posterior mean μ and log-variance
        ├── μ → Risk head → repayment-risk supervision
        └── z = μ + σ · ε
                ↓
         Generator + feature condition
                ↓
         Reconstructed borrower features

After warm-up:
Actual Default borrowers → encoder posterior → sampled zDefault
        ↓
Generator(zDefault + conditional vector)
        ↓
Synthetic Default features ↔ critic ↔ real Default features
        ↓
Inverse transformation → synthetic Default CSV
```

In **Stage 1**, the Conv1d encoder learns from the transformed borrower features of both repayment classes. The Risk head receives the posterior mean `mu`, while a sampled latent vector `z` is concatenated with a feature-condition vector and passed to the generator for reconstruction. The objective combines continuous reconstruction MSE, categorical/mode reconstruction cross-entropy, KL regularization, and binary repayment-risk loss. The generator also serves as the VAE decoder.

The Risk head is an MLP with a hidden dimension of 64, ReLU activation, dropout of 0.2, and a scalar output logit. Training uses binary cross-entropy with logits, allowing risk supervision to update both the Risk head and the encoder.

In **Stage 2**, latent vectors are drawn only from actual Default borrowers. The generator learns through a WGAN-GP adversarial objective and information loss, while the Risk-VAE objective continues to fine-tune the encoder, Risk head, and shared generator using both classes. The current information loss compares critic-feature means and variances with configurable margins. The critic's gradient-penalty coefficient is 10.

During **sample generation**, Default seed borrowers supply posterior means and variances. New latent vectors are sampled through reparameterization, combined with feature conditions, and decoded. The generated features are inverse-transformed and saved with `loan_status=1`.

The training examples above use the following settings:

| Setting | Value |
|---|---:|
| Total epochs | 150 |
| Risk-VAE warm-up epochs | 50 |
| Batch size | 512 |
| Latent dimension | 64 |
| Risk-head hidden dimension | 64 |
| Risk-head dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Continuous / categorical reconstruction weights | 1.0 / 1.0 |
| KL / risk weights | 1.0 / 1.0 |
| Adversarial / information weights | 1.0 / 1.0 |
| Risk-VAE fine-tuning weight | 0.1 |

Despite its name, `--encoder_freeze_epoch` sets the warm-up boundary; the encoder continues updating in Stage 2. The current training loop updates the critic once every five loader steps and the generator on every Stage 2 step. The declared `--d_steps_per_g` argument does not currently control this schedule.

## Example Pipeline Code

The following example combines generated Default observations with the original training data for downstream screening:

```python
import pandas as pd

original_train = pd.read_csv("dataset/train.csv")
synthetic_default = pd.read_csv("samples/risk-vae-ctabgan.csv")

assert synthetic_default["loan_status"].eq(1).all()

feature_columns = synthetic_default.columns.drop("loan_status")
original_features_and_label = original_train.loc[:, synthetic_default.columns]

augmented_train = pd.concat(
    [original_features_and_label, synthetic_default],
    ignore_index=True,
)

X_screening_train = augmented_train[feature_columns]
y_screening_train = augmented_train["loan_status"]
```

Train the screening classifier on the augmented training data, then apply it to actual test loans. Fit a common profitability model on the original training data, rank the screened real candidates by predicted returns, and evaluate the selected Top-q investment sets using their observed outcomes.

Synthetic observations are used for screening-model training. The profitability model and final investment evaluation use original loans.
