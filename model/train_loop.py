import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss)
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib as plt
import math

from model.pipeline.inverse_transform import apply_activate
from model.synthesizer.transformer import ImageTransformer, DataTransformer
from model.condvec import Condvec
from model.sampler import Sampler
from model.model import VAEEncoder, Generator

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def compute_reconstruction_losses(recon_logits, real_data, output_info):
    """Compute type-aware reconstruction losses over transformer spans.

    ``tanh`` spans are normalized numeric values and use MSE after tanh.
    ``softmax`` spans are categorical values or numeric/mixed mode indicators
    and use cross entropy directly on the decoder logits.

    Losses are averaged within each span and then across spans of the same
    type.  This prevents a high-cardinality one-hot span from dominating only
    because it has more encoded columns.
    """
    continuous_losses = []
    categorical_losses = []
    st = 0

    for dim, activation in output_info:
        dim = int(dim)
        ed = st + dim
        prediction = recon_logits[:, st:ed]
        target = real_data[:, st:ed]

        if activation == "tanh":
            continuous_losses.append(
                F.mse_loss(torch.tanh(prediction), target)
            )
        elif activation == "softmax":
            target_class = torch.argmax(target, dim=1)
            categorical_losses.append(
                F.cross_entropy(prediction, target_class)
            )
        else:
            raise ValueError(
                f"Unsupported transformer activation '{activation}' in output_info."
            )
        st = ed

    if st != recon_logits.size(1) or st != real_data.size(1):
        raise ValueError(
            "output_info width does not match reconstruction tensors: "
            f"spans={st}, recon={recon_logits.size(1)}, real={real_data.size(1)}"
        )

    zero = recon_logits.new_zeros(())
    continuous_loss = (torch.stack(continuous_losses).mean() if continuous_losses else zero)
    categorical_loss = (torch.stack(categorical_losses).mean() if categorical_losses else zero)
    return continuous_loss, categorical_loss

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[0]
    fake = torch.ones_like(d_interpolates).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_info_loss(real_features, fake_features, delta_mean=0.1, delta_var=0.1):

    real_mean = torch.mean(real_features, dim=0)
    fake_mean = torch.mean(fake_features, dim=0)
    real_var = torch.var(real_features, dim=0)
    fake_var = torch.var(fake_features, dim=0)

    mean_loss = torch.sum(F.relu(torch.abs(real_mean - fake_mean) - delta_mean))
    var_loss = torch.sum(F.relu(torch.abs(real_var - fake_var) - delta_var))
    return mean_loss + var_loss

def weights_init(model):
    
    classname = model.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)


def get_st_ed(target_col_index, transformer):
    
    st = 0
    output_pos = 0

    for raw_col_index, column_meta in enumerate(transformer.meta):
        column_blocks = []
        if column_meta["type"] in ["continuous", "skewed", "mixed"]:
            column_blocks = transformer.output_info[output_pos:output_pos + 2]
            output_pos += 2
        else:
            column_blocks = transformer.output_info[output_pos:output_pos + 1]
            output_pos += 1

        column_width = sum(dim for dim, _ in column_blocks)
        if raw_col_index == target_col_index:
            if len(column_blocks) != 1 or column_blocks[0][1] != "softmax":
                raise ValueError(
                    f"Target column index {target_col_index} is encoded as "
                    f"{column_meta['type']} with blocks {column_blocks}, not a categorical softmax block."
                )
            return (st, st + column_blocks[0][0])

        st += column_width

    raise ValueError(f"Target column index {target_col_index} is outside transformer metadata.")

def has_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        

def train_vae_gan(encoder, risk_head, generator, discriminator,
    risk_full_data, risk_labels, default_data, args, device):
    
    cont_recon_weight = getattr(args, "cont_recon_weight", None)
    if cont_recon_weight is None:
        cont_recon_weight = args.recon_weight
    cat_recon_weight = getattr(args, "cat_recon_weight", None)
    if cat_recon_weight is None:
        cat_recon_weight = args.recon_weight

    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)

    with open(args.dataprep_path, 'rb') as f:
        dataprep = pickle.load(f)

    # GAN conditioning and real samples must come exclusively from Defaults.
    cond_generator = Condvec(default_data, transformer.output_info, device=device)
    sampler = Sampler(default_data, transformer.output_info, device=device)

    image_size = int(np.ceil(np.sqrt(default_data.shape[1] + cond_generator.n_opt)))
    G_transformer = ImageTransformer(image_size, orig_dim=args.output_dim)
    D_transformer = ImageTransformer(image_size)

    dside = image_size 
    num_channels = 64

    risk_dataset = TensorDataset(
        torch.tensor(risk_full_data, dtype=torch.float32),
        torch.tensor(risk_labels, dtype=torch.float32),
    )
    loader = DataLoader(risk_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    default_tensor = torch.tensor(default_data, dtype=torch.float32, device=device)

    optimizerE = torch.optim.Adam(list(encoder.parameters()) + list(risk_head.parameters()), lr=args.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        epoch_recon_loss = 0.0
        epoch_cont_recon_loss = 0.0
        epoch_cat_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_vae_loss = 0.0
        epoch_risk_loss = 0.0
        epoch_risk_correct = 0
        epoch_risk_total = 0
        epoch_true_positive = 0
        epoch_predicted_positive = 0
        epoch_actual_positive = 0

        epoch_g_loss = 0.0
        epoch_info_loss = 0.0
        epoch_total_loss = 0.0
        epoch_d_loss = 0.0

        vae_count = 0
        gan_count = 0
        d_count = 0
        inner_bar = tqdm(enumerate(loader), total=len(loader), desc="Step", leave=False)

        for step, (x_risk, y_risk) in inner_bar:
            x_risk = x_risk.to(device)
            y_risk = y_risk.to(device)
            current_batch = x_risk.size(0)

            c, m, col, opt = cond_generator.sample_train(current_batch)
            if not isinstance(c, torch.Tensor):
                c = torch.from_numpy(c)
            c = c.float().to(device)

            # Risk-VAE update: all Fully Paid + Default borrowers.  RiskHead
            # intentionally receives posterior mean mu, never sampled z.
            z_risk, mu_risk, logvar_risk = encoder(x_risk)
            risk_logits = risk_head(mu_risk)
            risk_loss = F.binary_cross_entropy_with_logits(risk_logits, y_risk)

            risk_recon_image = generator(torch.cat([z_risk, c], dim=1))

            risk_recon_tabular = G_transformer.inverse_transform(risk_recon_image)
            cont_recon_loss, cat_recon_loss = compute_reconstruction_losses(
                risk_recon_tabular,
                x_risk,
                transformer.output_info,
            )
            recon_loss = cont_recon_loss + cat_recon_loss
            kl_loss = kl_divergence(mu_risk, logvar_risk)

            weighted_cont_recon_loss = cont_recon_weight * cont_recon_loss
            weighted_cat_recon_loss = cat_recon_weight * cat_recon_loss
            weighted_recon_loss = (weighted_cont_recon_loss + weighted_cat_recon_loss)
            weighted_risk_loss = args.risk_weight * risk_loss
            vae_loss = (weighted_recon_loss + args.kl_weight * kl_loss + weighted_risk_loss)

            epoch_recon_loss += recon_loss.item()
            epoch_cont_recon_loss += cont_recon_loss.item()
            epoch_cat_recon_loss += cat_recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_vae_loss += vae_loss.item()
            epoch_risk_loss += risk_loss.item()

            with torch.no_grad():
                risk_predictions = (risk_logits >= 0).float()
                epoch_risk_correct += (risk_predictions == y_risk).sum().item()
                epoch_risk_total += current_batch
                epoch_true_positive += ((risk_predictions == 1) & (y_risk == 1)).sum().item()
                epoch_predicted_positive += (risk_predictions == 1).sum().item()
                epoch_actual_positive += (y_risk == 1).sum().item()

            vae_count += 1

            if epoch < args.encoder_freeze_epoch:
                optimizerE.zero_grad()
                optimizerG.zero_grad()
                vae_loss.backward()
                optimizerE.step()
                optimizerG.step()

                continue
            
            if step % 5 == 0:
                with torch.no_grad():
                    default_idx = torch.randint(
                        0, len(default_tensor), (current_batch,), device=device
                    )
                    z_default, _, _ = encoder(default_tensor[default_idx])
                    input_gen = torch.cat([z_default, c], dim=1)
                    fake_image = generator(input_gen).detach()
                    fake_tabular = G_transformer.inverse_transform(fake_image)
                    fake_activated = apply_activate(fake_tabular, transformer.output_info)
                    fake_cat = torch.cat([fake_activated, c], dim=1)

                    real_data = torch.from_numpy(
                        sampler.sample(current_batch, col, opt)
                    ).to(device)
                    if epoch < args.real_activate_until_epoch:
                        real_data = apply_activate(real_data, transformer.output_info)
                    real_cat = torch.cat([real_data, c], dim=1)

                    real_image = D_transformer.transform(real_cat)
                    fake_image_d = D_transformer.transform(fake_cat)

                real_validity, _ = discriminator(real_image)
                fake_validity, _ = discriminator(fake_image_d)
                gp = compute_gradient_penalty(discriminator, real_image.data, fake_image_d.data, device)

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

                epoch_d_loss += d_loss.item()
                d_count += 1
            else:
                d_loss = None
                
            # GAN update: latent samples are drawn only from Default posteriors.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            default_idx = torch.randint(0, len(default_tensor), (current_batch,), device=device)
            z_default, _, _ = encoder(default_tensor[default_idx])
            input_gen = torch.cat([z_default, c], dim=1)
            fake_image = generator(input_gen)
            fake_tabular = G_transformer.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, transformer.output_info)
            fake_cat = torch.cat([fake_activated, c], dim=1)
            fake_image_d = D_transformer.transform(fake_cat)

            g_out, _ = discriminator(fake_image_d)
            g_loss = -torch.mean(g_out)

            with torch.no_grad():
                real_data_info = torch.from_numpy(sampler.sample(current_batch, col, opt)).to(device)
                
                if epoch < args.real_activate_until_epoch:
                    real_data_info = apply_activate(real_data_info, transformer.output_info)
                real_cat_info = torch.cat([real_data_info, c], dim=1)
                real_image_info = D_transformer.transform(real_cat_info)

            _, real_features = discriminator(real_image_info)
            real_features = real_features.detach()
            _, fake_features = discriminator(fake_image_d)

            info_loss = compute_info_loss(real_features, fake_features,
                                        delta_mean=args.delta_mean,
                                        delta_var=args.delta_var)

            weighted_adv_loss = (args.g_weight * g_loss)
            weighted_info_loss = (args.info_weight * info_loss)
            weighted_vae_loss = (args.vae_finetune_weight * vae_loss)
            
            total_loss = (weighted_adv_loss + weighted_info_loss + weighted_vae_loss)

            epoch_g_loss += g_loss.item()
            epoch_info_loss += info_loss.item()
            epoch_total_loss += total_loss.item()
            gan_count += 1

            optimizerE.zero_grad()
            optimizerG.zero_grad()

            total_loss.backward()

            optimizerE.step()
            optimizerG.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

        if (epoch + 1)>=50 and (epoch + 1) % 20 == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_path = os.path.join(args.checkpoint_dir, f"risk_vae_ctabgan_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'generator': generator.state_dict(),
                'encoder': encoder.state_dict(),
                'risk_head': risk_head.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, save_path)
            print(f"Saved checkpoint (E + R + G + D) at epoch {epoch + 1} -> {save_path}")

        epoch_metrics = {
            "epoch": epoch + 1,
            "recon_loss": epoch_recon_loss / max(vae_count, 1),
            "cont_recon_loss": (epoch_cont_recon_loss / max(vae_count, 1)),
            "cat_recon_loss": (epoch_cat_recon_loss / max(vae_count, 1)),
            "cont_recon_loss_weighted": (cont_recon_weight * epoch_cont_recon_loss / max(vae_count, 1)),
            "cat_recon_loss_weighted": (cat_recon_weight * epoch_cat_recon_loss / max(vae_count, 1)),
            "recon_loss_weighted": ((cont_recon_weight * epoch_cont_recon_loss + cat_recon_weight * epoch_cat_recon_loss) / max(vae_count, 1)),
            "kl_loss": epoch_kl_loss / max(vae_count, 1),
            "risk_loss_raw": epoch_risk_loss / max(vae_count, 1),
            "risk_loss_weighted": (args.risk_weight * epoch_risk_loss / max(vae_count, 1)),
            "vae_loss": epoch_vae_loss / max(vae_count, 1),
            "risk_accuracy": (epoch_risk_correct / max(epoch_risk_total, 1)),
            "default_recall": (epoch_true_positive / max(epoch_actual_positive, 1)),
            "default_precision": (epoch_true_positive / max(epoch_predicted_positive, 1)),
        }

        if gan_count > 0:
            epoch_metrics.update({
                "g_loss": epoch_g_loss / gan_count,
                "info_loss": epoch_info_loss / gan_count,
                "total_loss": epoch_total_loss / gan_count,
            })

        if d_count > 0:
            epoch_metrics["d_loss"] = epoch_d_loss / d_count

        stage = "warm-up" if epoch < args.encoder_freeze_epoch else "adversarial"
        metric_names = [
            "vae_loss", "recon_loss", "kl_loss", "risk_loss_raw",
            "risk_accuracy", "default_recall", "default_precision",
        ]
        if gan_count > 0:
            metric_names.extend(["g_loss", "info_loss", "total_loss"])
        if d_count > 0:
            metric_names.append("d_loss")
        metrics = " | ".join(
            f"{name}={epoch_metrics[name]:.4f}" for name in metric_names
        )
        tqdm.write(f"Epoch {epoch + 1}/{args.epochs} [{stage}] | {metrics}")

        final_dir = os.path.join(args.checkpoint_dir, "final")
        os.makedirs(final_dir, exist_ok=True)     

    final_checkpoint = {
        'encoder': encoder.state_dict(),
        'risk_head': risk_head.state_dict(),
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(final_dir, f"risk_vae_ctabgan_epoch{epoch+1}.pth"))
    torch.save(final_checkpoint, os.path.join(args.checkpoint_dir, args.save_name))
    print(f"Final checkpoint saved to " f"{os.path.join(final_dir, f'risk_vae_ctabgan_epoch{epoch+1}.pth')}")

def generate_samples(args, default_data, device):
    with open(args.transformer_path, 'rb') as f:
        transformer = pickle.load(f)
    output_info = transformer.output_info
    
    with open(args.dataprep_path, "rb") as f:
        dataprep = pickle.load(f)

    condvec = Condvec(default_data, output_info, device=device)
    image_size = int(np.ceil(np.sqrt(default_data.shape[1] + condvec.n_opt)))
    transformer_G = ImageTransformer(image_size, orig_dim=args.output_dim)

    encoder = VAEEncoder(input_dim=default_data.shape[1], latent_dim=args.latent_dim).to(device)
    generator = Generator(input_dim=args.latent_dim + condvec.n_opt,
                          gside=image_size, num_channels=64).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.save_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    generator.load_state_dict(checkpoint['generator'])

    encoder.eval()
    generator.eval()

    default_tensor = torch.tensor(default_data, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, default_mu, default_logvar = encoder(default_tensor)
    n_seed = default_tensor.size(0)

    samples = []
    for _ in tqdm(range((args.num_samples + args.batch_size - 1) // args.batch_size), desc="Generating"):
        c = condvec.sample(args.batch_size)
        if not isinstance(c, torch.Tensor):
            c = torch.from_numpy(c)
        c = c.float().to(device)

        seed_idx = torch.randint(0, n_seed, (args.batch_size,), device=device)

        with torch.no_grad():
            sample_mu = default_mu[seed_idx]
            sample_std = torch.exp(0.5 * default_logvar[seed_idx])
            z_default = sample_mu + sample_std * torch.randn_like(sample_std)
            input_gen = torch.cat([z_default, c], dim=1)
            fake_image = generator(input_gen)
            fake_tabular = transformer_G.inverse_transform(fake_image)
            fake_activated = apply_activate(fake_tabular, output_info)
            samples.append(fake_activated.cpu().numpy())

    final_samples = np.concatenate(samples, axis=0)[:args.num_samples]

    tabular_data = transformer.inverse_transform(final_samples)
    recovered_df = dataprep.inverse_prep(tabular_data) 
    recovered_df["loan_status"] = 1
    output_path = os.path.join(args.sample_dir, "risk-vae-ctabgan.csv")
    recovered_df.to_csv(output_path, index=False)
