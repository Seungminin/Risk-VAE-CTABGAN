import argparse
import os
import torch
import pickle
import numpy as np

from model.preprocess import preprocess_data
from model.pipeline.data_utils import load_processed_data
from model.train_loop import train_vae_gan, generate_samples, weights_init
from model.model import VAEEncoder, Generator, Discriminator, RiskHead
from model.pipeline.data_utils import show_all_parameters
from model.condvec import Condvec

import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Risk-VAE-CTAB-GAN")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess", "train", "generate", "only_train"],
                        help="Choose one mode: preprocess, train, generate, only_train")
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--encoder_freeze_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0, help="Backward-compatible default for continuous/categorical reconstruction weights",)
    parser.add_argument("--cont_recon_weight", type=float, default=1.0, help="Weight for MSE reconstruction of tanh (numeric) spans; defaults to --recon_weight",)
    parser.add_argument("--cat_recon_weight", type=float, default=1.0, help="Weight for CE reconstruction of softmax (category/mode) spans; defaults to --recon_weight",)
    parser.add_argument("--risk_weight", type=float, default=1.0, help="Weight for borrower default-risk classification loss")
    parser.add_argument("--risk_hidden_dim", type=int, default=64, help="Hidden dimension of the RiskHead MLP")
    parser.add_argument("--vae_finetune_weight", type=float, default=0.1, help="Weight for the VAE regularization term after encoder pretraining")
    parser.add_argument("--g_weight", type=float, default=1)
    parser.add_argument("--info_weight", type=float, default=1.0)
    parser.add_argument('--delta_mean', type=float, default=0.1)
    parser.add_argument('--delta_var', type=float, default=0.1)
    parser.add_argument('--dataset_path', type=str, default='~~~')
    parser.add_argument('--sample_dir', type=str, default='~~~')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='risk_vae_ctabgan.pt')
    parser.add_argument('--preprocessed_path', type=str, default='preprocess/preprocessed.csv')
    parser.add_argument('--risk_labels_path', type=str, default='preprocess/risk_labels.npy')
    parser.add_argument('--transformer_path', type=str, default='preprocess/transformer/transformer.pkl')
    parser.add_argument('--dataprep_path', type=str, default='preprocess/dataprep/dataprep.pkl')
    parser.add_argument("--real_activate_until_epoch", type=int, default=50,
                    help="Number of epochs to apply activation to real data for fair D comparison")
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples to generate')
    parser.add_argument('--d_steps_per_g', type=int, default=10, help='Number of discriminator steps per generator step')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    preprocessed_dir = os.path.dirname(args.preprocessed_path)
    if preprocessed_dir:
        os.makedirs(preprocessed_dir, exist_ok=True)

    ## Preprocess data if needed
    if args.mode == "preprocess":
        print("Preprocessing original data using DataPrep")
        preprocess_data(
            raw_path=args.dataset_path,
            save_path=args.preprocessed_path,
            label_save_path=args.risk_labels_path,
        )
        return

    if args.mode == "train":
        print("Preprocessing original data using DataPrep")
        preprocess_data(
            raw_path=args.dataset_path,
            save_path=args.preprocessed_path,
            label_save_path=args.risk_labels_path,
        )

    if not os.path.exists(args.preprocessed_path):
        candidates = [
            args.preprocessed_path,
            "preprocess/preprocessed.csv",
            "model/preprocess/processed.csv",
        ]
        existing_candidates = [path for path in candidates if os.path.exists(path)]
        hint = (
            f"Processed CSV not found at: {args.preprocessed_path}\n"
            "Run preprocess with the same --preprocessed_path, or train with the path printed by preprocess.\n"
            "Default preprocess output: preprocess/preprocessed.csv"
        )
        if existing_candidates:
            hint += f"\nExisting processed file candidate(s): {', '.join(existing_candidates)}"
        raise FileNotFoundError(hint)

    if not os.path.exists(args.risk_labels_path):
        raise FileNotFoundError(
            f"Risk label file not found at: {args.risk_labels_path}\n"
            "Run preprocess/train first, using the same --risk_labels_path."
        )

    print("Loading processed features and risk labels")
    risk_full_data = load_processed_data(args.preprocessed_path)
    risk_labels = np.load(args.risk_labels_path).astype(np.float32).reshape(-1)
    if len(risk_full_data) != len(risk_labels):
        raise ValueError(
            "Processed features and labels have different row counts: "
            f"{len(risk_full_data)} != {len(risk_labels)}"
        )
    if not set(np.unique(risk_labels).tolist()).issubset({0.0, 1.0}):
        raise ValueError("risk_labels must contain only 0 (Fully Paid) and 1 (Default).")

    default_data = risk_full_data[risk_labels == 1]
    if len(default_data) == 0:
        raise ValueError("No Default borrowers (loan_status == 1) were found.")

    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
        args.output_info = transformer.output_info
        args.output_dim = transformer.output_dim
        
    # Conditional sampling and GAN dimensions are based only on Default rows.
    condvec = Condvec(default_data, transformer.output_info, device=device)

    ## Generate synthetic samples if needed
    if args.mode == "generate":
        print("Generating synthetic samples")
        generate_samples(
            args=args,
            default_data=default_data,
            device=device
        )
        return

    encoder = VAEEncoder(input_dim=risk_full_data.shape[1], latent_dim=args.latent_dim).to(device)
    risk_head = RiskHead(
        latent_dim=args.latent_dim,
        hidden_dim=args.risk_hidden_dim,
    ).to(device)

    g_input_dim = args.latent_dim + condvec.n_opt
    gside = int(np.ceil(np.sqrt(default_data.shape[1] + condvec.n_opt)))
    num_channel = 64

    generator = Generator(input_dim=g_input_dim, gside=gside, num_channels=num_channel).to(device)
    discriminator = Discriminator(dside=gside, num_channels=num_channel).to(device)
    
    generator.apply(weights_init)

    show_all_parameters(encoder, name="VAE Encoder")
    show_all_parameters(generator, name="Generator")
    show_all_parameters(discriminator, name="Discriminator")
    show_all_parameters(risk_head, name="Risk Head")

    if args.mode in ["train", "only_train"]:
        print("Starting training")
        train_vae_gan(
            encoder=encoder,
            risk_head=risk_head,
            generator=generator,
            discriminator=discriminator,
            risk_full_data=risk_full_data,
            risk_labels=risk_labels,
            default_data=default_data,
            args=args,
            device=device
        )


if __name__ == "__main__":
    main()
