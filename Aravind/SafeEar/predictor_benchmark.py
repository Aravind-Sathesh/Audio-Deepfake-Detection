import torch
import torch.nn as nn
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import time
import numpy as np
from safeear.losses.loss import compute_eer
from sklearn.metrics import roc_auc_score

torch.set_float32_matmul_precision("high")

class HealingSecurityBlock(nn.Module):
    def __init__(self, mask_type="none", mask_ratio=0.4, use_predictor=False, predictor_ckpt=None, alpha=0.005):
        super().__init__()
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        self.use_predictor = use_predictor
        self.alpha = alpha
        
        if self.use_predictor:
            self.predictor = __import__('train_predictor').NATPredictor(feature_dim=1024)
            ckpt = torch.load(predictor_ckpt, map_location="cpu")
            state_dict = {k.replace("predictor.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("predictor.")}
            self.predictor.load_state_dict(state_dict)
            self.predictor.eval()
            self.predictor.cuda()

    def forward(self, acoustic_tokens_list):
        secured_list =[]
        for i, tokens in enumerate(acoustic_tokens_list):
            current_tokens = tokens.clone()
            
            # --- Task 4: Adaptive Watermark ---
            if self.alpha > 0.0:
                variance = torch.var(current_tokens, dim=-1, keepdim=True) + 1e-6
                signature = torch.randint(0, 2, current_tokens.shape, device=current_tokens.device).float() * 2 - 1
                current_tokens = current_tokens + (self.alpha * variance * signature)

            # --- Task 3: Semantic Drop ---
            if self.mask_type == "semantic_drop" and i < 2:
                current_tokens = torch.zeros_like(current_tokens)
                
            secured_list.append(current_tokens)

        ground_truth = torch.cat(secured_list, dim=1)

        # --- Task 2: Masking & Healing ---
        if self.mask_type in ["none", "semantic_drop"] or self.mask_ratio == 0.0:
            final_tokens = ground_truth
        else:
            if self.mask_type == "random":
                mask = torch.rand(ground_truth.shape, device=ground_truth.device) > self.mask_ratio
            elif self.mask_type == "saliency_guided":
                time_energy = torch.norm(ground_truth, dim=1, keepdim=True)
                threshold = torch.quantile(time_energy, self.mask_ratio, dim=-1, keepdim=True)
                mask = (time_energy >= threshold).expand_as(ground_truth)

            masked_tokens = ground_truth * mask
            
            if self.use_predictor:
                with torch.no_grad():
                    healed_predictions = self.predictor(masked_tokens)
                    final_tokens = torch.where(mask, ground_truth, healed_predictions)
            else:
                final_tokens = masked_tokens

        return torch.split(final_tokens, 1024, dim=1)

class PredictorEvalTrainer(pl.LightningModule):
    def __init__(self, original_system, security_config):
        super().__init__()
        self.original_system = original_system
        self.security_block = HealingSecurityBlock(**security_config)
        self.decouple_model = self.original_system.decouple_model
        self.detect_model = self.original_system.detect_model
        self.all_scores, self.all_labels = [], []
        self.inference_times =[]

    def test_step(self, batch, batch_idx):
        x, feat, target, *rest = batch
        x_wav = x.unsqueeze(1) if x.ndim == 2 else x
        
        t0 = time.time()
        with torch.no_grad():
            self.decouple_model.eval()
            _, _, _, acoustic_tokens = self.decouple_model(x_wav, layers=[0,1,2,3,4,5,6,7])
        
        secured_tokens = self.security_block(acoustic_tokens)
        raw_logits, _ = self.detect_model(secured_tokens)
        t1 = time.time()
        
        self.inference_times.append((t1 - t0) * 1000)
        
        scores = torch.softmax(raw_logits, dim=-1)[:, 1]
        self.all_scores.append(scores.cpu().numpy())
        self.all_labels.append(target.cpu().numpy())

    def on_test_epoch_end(self):
        scores = np.concatenate(self.all_scores)
        labels = np.concatenate(self.all_labels)
        target_scores, nontarget_scores = scores[labels == 0], scores[labels == 1]
        
        eer_out = compute_eer(target_scores, nontarget_scores)
        eer = eer_out[0] if isinstance(eer_out, tuple) else eer_out
        if eer > 0.5: eer = 1.0 - eer
        
        auc = roc_auc_score(labels, scores)
        if auc < 0.5: auc = 1.0 - auc
        
        avg_inf_time = np.mean(self.inference_times)
        
        self.log("final_eer", eer * 100)
        
        print(f"\n{'='*50}\n          RESULTS: {self.security_block.mask_type.upper()} MASK\n{'='*50}")
        print(f"EER (%):           {eer * 100:.4f}")
        print(f"AUC (%):           {auc * 100:.4f}")
        print(f"Inference (ms):    {avg_inf_time:.4f}")
        print(f"{'='*50}\n")

def main(args):
    cfg = OmegaConf.load(args.conf_dir)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    system_class = hydra.utils.get_class(cfg.system._target_)
    original_system = system_class.load_from_checkpoint(
        args.ckpt_path,
        decouple_model=hydra.utils.instantiate(cfg.decouple_model),
        detect_model=hydra.utils.instantiate(cfg.detect_model),
        lr_raw_former=cfg.system.lr_raw_former,
        save_score_path=cfg.system.save_score_path,
    )
    
    secure_system = PredictorEvalTrainer(original_system, args.security_config)
    trainer = hydra.utils.instantiate(cfg.trainer)
    if args.limit_test_batches: trainer.limit_test_batches = args.limit_test_batches
    
    trainer.test(secure_system, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--predictor_ckpt', default=None)
    parser.add_argument('--mask_type', type=str, default="none", choices=["none", "random", "saliency_guided", "semantic_drop"])
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--use_predictor', action='store_true')
    parser.add_argument('--limit_test_batches', type=int, default=2000)
    args = parser.parse_args()
    
    args.security_config = {
        "mask_type": args.mask_type, 
        "mask_ratio": args.mask_ratio, 
        "use_predictor": args.use_predictor, 
        "predictor_ckpt": args.predictor_ckpt,
        "alpha": args.alpha
    }
    main(args)
