import torch
import torch.nn as nn
import hydra
import argparse
import pytorch_lightning as pl
import time
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from safeear.losses.loss import compute_eer

torch.set_float32_matmul_precision("high")

class PrivacySecurityBlock(nn.Module):
    def __init__(self, mask_type="none", mask_ratio=0.0, alpha=0.005):
        super().__init__()
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.alpha = alpha
        print(f"--- Initializing SecurityBlock -> Mask: {self.mask_type}, Ratio: {self.mask_ratio}, Watermark Alpha: {self.alpha} ---")

    def forward(self, acoustic_tokens_list):
        # SpeechTokenizer outputs a list of tensors
        secured_list =[]
        
        for i, tokens in enumerate(acoustic_tokens_list):
            current_tokens = tokens.clone()

            # --- Task 4: Adaptive Watermark ---
            if self.alpha > 0.0:
                variance = torch.var(current_tokens, dim=-1, keepdim=True) + 1e-6
                signature = torch.randint(0, 2, current_tokens.shape, device=current_tokens.device).float() * 2 - 1
                current_tokens = current_tokens + (self.alpha * variance * signature)

            # --- Task 2: Masking Strategies ---
            if self.mask_type == "random" and self.mask_ratio > 0.0:
                mask = torch.rand(current_tokens.shape, device=current_tokens.device) > self.mask_ratio
                current_tokens = current_tokens * mask
            
            elif self.mask_type == "semantic_drop":
                # Layers 0 and 1 represent Semantics. Zero them out.
                if i < 2:  
                    current_tokens = torch.zeros_like(current_tokens)
            
            secured_list.append(current_tokens)

        return secured_list


class SecureSafeEarTrainer(pl.LightningModule):
    def __init__(self, original_system: pl.LightningModule, security_config: dict):
        super().__init__()
        self.original_system = original_system
        self.security_block = PrivacySecurityBlock(**security_config)
        self.decouple_model = self.original_system.decouple_model
        self.detect_model = self.original_system.detect_model
        
        # Lists to store scores and labels for offline calculation
        self.all_scores = []
        self.all_labels =[]

    def test_step(self, batch, batch_idx):
        x, feat, target, *rest = batch
        
        x_wav = x.unsqueeze(1) if x.ndim == 2 else x
        
        with torch.no_grad():
            self.decouple_model.eval()
            _, _, _, acoustic_tokens = self.decouple_model(x_wav, layers=[0,1,2,3,4,5,6,7])
        
        secured_acoustic_tokens = self.security_block(acoustic_tokens)
        
        raw_logits, raw_feature = self.detect_model(secured_acoustic_tokens)
        
        # Probability of being a "Spoof" (Class 1)
        scores = torch.softmax(raw_logits, dim=-1)[:, 1]
        
        # Store scores and labels for offline EER calculation
        self.all_scores.append(scores.cpu().numpy())
        self.all_labels.append(target.cpu().numpy())

    def on_test_epoch_end(self):
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        
        # Label 0 = Bona Fide (Target), Label 1 = Spoof (Non-Target)
        target_scores = scores[labels == 0]
        nontarget_scores = scores[labels == 1]
        
        eer_out = compute_eer(target_scores, nontarget_scores)
        eer = eer_out[0] if isinstance(eer_out, tuple) else eer_out
        
        if eer > 0.9:
            eer = 1.0 - eer
            
        self.log("final_eer", eer * 100)

    def configure_optimizers(self):
        return self.original_system.configure_optimizers()


@rank_zero_only
def print_only(message: str):
    print(message)

def run_benchmark(cfg: DictConfig, args):
    print_only("--- Loading System and Dataloaders ---")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    system_class = hydra.utils.get_class(cfg.system._target_)
    original_system = system_class.load_from_checkpoint(
        args.ckpt_path,
        decouple_model=hydra.utils.instantiate(cfg.decouple_model),
        detect_model=hydra.utils.instantiate(cfg.detect_model),
        lr_raw_former=cfg.system.lr_raw_former,
        save_score_path=cfg.system.save_score_path,
    )
    
    secure_system = SecureSafeEarTrainer(original_system, args.security_config)
    
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    if args.limit_test_batches:
        trainer.limit_test_batches = args.limit_test_batches
    
    print_only(f"\n--- Starting Benchmark for Mask: {args.security_config['mask_type']} @ {args.security_config['mask_ratio']*100}% ---")
    start_time = time.time()
    
    # Run the test
    results = trainer.test(secure_system, datamodule=datamodule)
    
    duration = time.time() - start_time
    num_samples = len(datamodule.test_dataloader().dataset)
    if args.limit_test_batches:
        num_samples = args.limit_test_batches
        
    avg_latency_ms = (duration / num_samples) * 1000
    
    print_only("\n" + "="*50)
    print_only("          FINAL BENCHMARK RESULTS")
    print_only("="*50)
    print_only(f"EER (%):           {results[0]['final_eer']:.4f}")
    print_only(f"Avg Latency (ms):  {avg_latency_ms:.4f}")
    print_only("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Benchmarks for SafeEar")
    parser.add_argument('--conf_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--mask_type', type=str, default="none", choices=["none", "random", "semantic_drop"])
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.0, help="Watermark strength. Set to 0.0 for no watermark.")
    parser.add_argument('--limit_test_batches', type=int, default=2000)
    
    cli_args = parser.parse_args()

    cfg = OmegaConf.load(cli_args.conf_dir)

    cli_args.security_config = {
        "mask_type": cli_args.mask_type,
        "mask_ratio": cli_args.mask_ratio,
        "alpha": cli_args.alpha
    }

    run_benchmark(cfg, cli_args)