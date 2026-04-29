# Privacy-Preserving Audio Deepfake Detection

This repository contains a client-server preprocessing pipeline designed to secure audio payloads prior to deepfake detection. Built on top of the SafeEar architecture, the pipeline explicitly decouples semantic meaning from acoustic timbre, authenticates the source, and masks the payload to guarantee conversational privacy.

## Architecture Overview

The pipeline operates in two distinct phases:

### 1. Client-Side

- **Semantic Disentanglement:** Raw audio is processed through `SpeechTokenizer`. Residual Vector Quantization (RVQ) layers 0 and 1 (representing spoken words) are permanently discarded.
- **Adaptive Watermarking:** A zero-mean bipolar sequence ($\alpha=0.005$) is injected into the remaining acoustic tokens (layers 2-7). The embedding strength dynamically scales with the local variance of the feature map to hide inside high-energy phonemes.
- **Privacy Masking:** A 40% mask is applied to the acoustic tokens. The system uses **Saliency-Guided Masking** (via PDSM logic) to protect high-energy fricatives (where deepfake artifacts reside) and drops low-energy background tokens.
- **Transport:** The resulting 3D tensor is transmitted to the server.

### 2. Server-Side

- **Reconstruction (Healer):** A Non-Autoregressive Transformer (NAT) predicts and reconstructs the missing 40% of the acoustic tokens using bidirectional self-attention.
- **Classification:** The healed acoustic tokens are passed to the SafeEar Audio Spectrogram Transformer (AST) to classify the audio as Bona Fide (Real) or Spoof (Fake).

---

## Environment Setup

The pipeline requires specific library versions to ensure compatibility with underlying LSTM CUDA kernels and Fairseq dependencies.

```bash
# 1. Create and activate environment
conda create -n safeear python=3.9
conda activate safeear

# 2. Install PyTorch 1.13.1
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. Install core requirements
pip install pip==24.0
pip install -r requirements.txt

# 4. Install forensic evaluation metrics
pip install scikit-learn pesq soundfile
```

---

## Data Preparation

The pipeline uses the **ASVspoof 2019 LA** dataset. The SafeEar AST model requires pre-computed HuBERT L9 feature maps.

1.  Place the ASVspoof 2019 dataset in `datas/datasets/ASVSpoof2019`.
2.  Download the required models to `model_zoos/`:
    - `hubert_base_ls960.pt`
    - `SpeechTokenizer.pt`
3.  Generate the HuBERT features:
    `bash
CUDA_VISIBLE_DEVICES=1 python dump_hubert_avg_feature.py datasets/ASVSpoof2019 datasets/ASVSpoof2019_Hubert_L9
`
    _(Note: The dataloader in `safeear/datas/asvspoof19.py` has been patched with `torch.nn.functional.pad` to handle variable-length audio inputs. Tensors are strictly padded/cropped to 64,000 samples)._

---

## Running the Pipeline

### 1. Training the AST Deepfake Detector (Baseline)

To train the base AST classifier from scratch over the `SpeechTokenizer` acoustic tokens:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --conf_dir config/train19.yaml
```

### 2. Training the NAT Predictor (Healer)

The predictor is a self-supervised Transformer trained to guess masked tokens.
_(Dev Note: The `SpeechTokenizer` LSTMs lack `bfloat16` CUDA support in PyTorch 1.13. The training script uses standard 32-bit floats with `gradient_clip_val=1.0` to prevent NaN loss explosions from MSE)._

```bash
CUDA_VISIBLE_DEVICES=1 python train_predictor.py --conf_dir config/train19.yaml --mask_ratio 0.4
```

This saves a checkpoint in `checkpoints/predictor_mask40/`.

### 3. Running Ablation Benchmarks

The `predictor_benchmark.py` script wraps the original SafeEar test loop. It allows dynamic injection of the watermarking, masking, and healing blocks.

**Arguments:**

- `--mask_type`: `none`, `random`, `saliency_guided`, `semantic_drop`
- `--mask_ratio`: Float (e.g., `0.4` for 40% masking)
- `--alpha`: Watermark strength (default `0.005`, set to `0.0` to disable)
- `--use_predictor`: Flag to route tokens through the NAT before classification
- `--limit_test_batches`: Integer (e.g., `2000` for a quick subset evaluation)

**Example Run (PDSM Saliency Masking + Healer):**

```bash
CUDA_VISIBLE_DEVICES=1 python predictor_benchmark.py \
    --conf_dir config/train19.yaml \
    --ckpt_path Exps/ASVspoof19/checkpoints/epoch=14-val_eer=0.0310.ckpt \
    --mask_type saliency_guided \
    --mask_ratio 0.4 \
    --alpha 0.005 \
    --use_predictor \
    --predictor_ckpt "checkpoints/predictor_mask40/best-predictor-epoch=09-val_loss=0.0630.ckpt"
```

The script outputs Equal Error Rate (EER), AUC, and Inference Latency (ms).

---

## Appendix: Google Lyra Codec Evaluation

An exploratory module evaluated Google Lyra (v1.3.2) as an ultra-low bitrate transport codec.

**Setup:**
Lyra requires compiling C++ binaries via Bazel. Due to legacy TensorFlow `cc_shared_library` dependency issues, Bazel must be strictly pinned to `v5.3.0`.

```bash
cd lyra
echo "5.3.0" > .bazelversion
bazel build -c opt lyra/cli_example:encoder_main
bazel build -c opt lyra/cli_example:decoder_main
```

**Benchmarking:**
Run `python lyra_full_benchmark.py`. This script interfaces with the C++ binaries via `subprocess` and calculates forensic audio metrics (PESQ, LSD, MCD, SAR).

_(Dev Note: Lyra achieved ~0.024x RTF but was abandoned for the main pipeline. As a parametric generative vocoder, it synthesizes new audio, causing MCD > 42 dB and destroying the microscopic acoustic anomalies required for deepfake detection)._

---

## Future Development Notes

1. **Artifact Laundering:** The current NAT Predictor utilizes Mean Squared Error (MSE) loss. At heavy masking ratios, MSE regression causes token smoothing, inadvertently "laundering" jagged deepfake anomalies. The loss function requires refactoring to Discrete Cross-Entropy over the RVQ vocabulary to enforce exact token reconstruction.
2. **Semantic Leakage:** `SpeechTokenizer` induces a ~13.05% EER penalty when explicitly dropping layers 0 and 1. Future iterations should test `SemantiCodec` or `FACodec` for cleaner semantic/acoustic disentanglement.
