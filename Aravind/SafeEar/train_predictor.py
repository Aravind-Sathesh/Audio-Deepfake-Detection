import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import os

torch.set_float32_matmul_precision("high")

# ===================================================================
# 1. THE NON-AUTOREGRESSIVE TRANSFORMER (NAT) PREDICTOR
# ===================================================================
class NATPredictor(nn.Module):
    def __init__(self, feature_dim=1024, num_layers=4, num_heads=8):
        super().__init__()
        # Project up, apply Transformer, Project down
        self.input_proj = nn.Conv1d(feature_dim * 7, feature_dim, kernel_size=1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dim_feedforward=feature_dim * 4, 
            dropout=0.1, 
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Conv1d(feature_dim, feature_dim * 7, kernel_size=1)

    def forward(self, masked_features):
        # Convert [B, C, T] -> [B, T, C] for Transformer
        x = self.input_proj(masked_features).transpose(1, 2)
        
        # Bidirectional Self-Attention
        x = self.transformer(x)
        
        # Convert back to [B, C, T]
        x = x.transpose(1, 2)
        return self.output_proj(x)

# ===================================================================
# 2. SELF-SUPERVISED LIGHTNING MODULE
# ===================================================================
class PredictorTrainer(pl.LightningModule):
    def __init__(self, decouple_model, mask_ratio=0.4, lr=1e-4):
        super().__init__()
        self.decouple_model = decouple_model
        self.decouple_model.eval() # Freeze the tokenizer
        
        # Determine the feature dimension from SafeEar configs (usually 1024)
        self.feature_dim = 1024 
        self.predictor = NATPredictor(feature_dim=self.feature_dim)
        
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, audio):
        with torch.no_grad():
            _, _, _, acoustic_tokens = self.decouple_model(audio, layers=[0,1,2,3,4,5,6,7])
            # Concatenate the RVQ layers just like SafeEar does before the bottleneck
            ground_truth = torch.cat(acoustic_tokens, dim=1) # Shape: [B, C, T]

        # Apply random masking (Self-Supervision)
        time_energy = torch.norm(ground_truth, dim=1, keepdim=True)
        threshold = torch.quantile(time_energy, self.mask_ratio, dim=-1, keepdim=True)
        mask = (time_energy >= threshold).expand_as(ground_truth)
        masked_input = ground_truth * mask

        # Predict the missing features
        reconstructed = self.predictor(masked_input)

        # Calculate loss ONLY on the masked positions
        inverse_mask = ~mask
        loss = self.criterion(reconstructed[inverse_mask], ground_truth[inverse_mask])
        
        return loss

    def training_step(self, batch, batch_idx):
        x, feat, target, *rest = batch
        x_wav = x.unsqueeze(1) if x.ndim == 2 else x
        loss = self(x_wav)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, feat, target, *rest = batch
        x_wav = x.unsqueeze(1) if x.ndim == 2 else x
        loss = self(x_wav)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.predictor.parameters(), lr=self.lr, weight_decay=1e-4)

# ===================================================================
# 3. MAIN TRAINING LOOP
# ===================================================================
def main(args):
    print(f"--- Initializing Predictor Training (Mask Ratio: {args.mask_ratio}) ---")
    cfg = OmegaConf.load(args.conf_dir)
    
    # Instantiate DataModule and Tokenizer
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    decouple_model = hydra.utils.instantiate(cfg.decouple_model)
    decouple_model.load_state_dict(torch.load(cfg.speechtokenizer_path, map_location="cpu"))
    
    # Initialize our Self-Supervised Predictor
    system = PredictorTrainer(decouple_model=decouple_model, mask_ratio=args.mask_ratio)
    
    # Setup Callbacks and Trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/predictor_mask{int(args.mask_ratio*100)}",
        filename="best-predictor-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    trainer = pl.Trainer(
        devices=[0], # We will override this with CUDA_VISIBLE_DEVICES
        accelerator="gpu",
        max_epochs=15, # Fast training for the Predictor
        callbacks=[checkpoint_callback],
        precision=32, gradient_clip_val=1.0 # Fix NaN
    )
    
    trainer.fit(system, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', type=str, required=True)
    parser.add_argument('--mask_ratio', type=float, default=0.4, help="Ratio of tokens to mask during training")
    args = parser.parse_args()
    main(args)
