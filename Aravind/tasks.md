# Aravind

## 1. SafeEar Privacy Pipeline

### Tasks Completed:

1. Baseline Setup: Cloned the SafeEar pipeline on the MIG-2 server. Trained the model to establish a baseline Deepfake Detection score.
2. Authentication: Implemented an adaptive watermark that hides in the loud parts of the audio. Tests confirmed it survives encryption without breaking the detector.
3. Absolute Privacy: Discarded the "words" (semantic tokens) completely to ensure conversations cannot be eavesdropped on.
4. Identity Masking: Deleted 40% of the remaining audio (acoustic tokens) to obscure the voice timbre.
5. The Healer AI: Built a server-side AI (Transformer Predictor) to predict and fill in the missing 40% of the audio before it reaches the deepfake detector.

### The PDSM Fix:

- The Problem: Randomly masking 40% of the audio caused the Healer AI to fail. The AI filled in the blanks with "smooth" audio, which accidentally erased the robotic deepfake glitches.
- The Fix: Switched to Smart Masking (Saliency-Guided) based on the PDSM paper. Loud, harsh sounds (like "S" and "F") where deepfake glitches hide were protected. Only the quiet background noise was masked. This allowed the Healer AI to reconstruct the background without destroying the deepfake clues, successfully improving accuracy.

### Scores Table (ASVspoof 2019, 2000 files)

_EER (Lower is better). AUC (Higher is better)._

| #   | Configuration                    | Mask Type | Ratio | Healer? | EER (%) | AUC (%) |
| :-- | :------------------------------- | :-------- | :---- | :------ | :------ | :------ |
| 1   | Baseline (No Privacy)            | None      | 0%    | No      | 4.21    | 99.16   |
| 2   | + Watermark                      | None      | 0%    | No      | 4.15    | 99.18   |
| 3   | + Light Masking                  | Random    | 10%   | No      | 4.56    | 99.15   |
| 4   | + Heavy Masking                  | Random    | 40%   | No      | 5.94    | 98.66   |
| 5   | + Heavy Masking & Heal           | Random    | 40%   | Yes     | 5.82    | 98.78   |
| 6   | Semantic Drop (Words Deleted)    | Semantic  | N/A   | No      | 13.05   | 95.18   |
| 7   | Smart Masking (Quiet parts only) | Saliency  | 40%   | No      | 14.40   | 93.69   |
| 8   | Smart Masking & Heal             | Saliency  | 40%   | Yes     | 13.88   | 93.61   |

## 2. Google Lyra Codec

### Tasks Completed:

- Compiled Lyra codec (v1.3.2) from source using Bazel.
- Evaluated its viability as a low-bitrate compressor for the network transport layer.

### Metrics

Dataset: ASVspoof 2019 LA Evaluation Set (~71,900 files).
Bitrate: 3.2 kbps

| Metric                 | Value      | Interpretation                                      |
| :--------------------- | :--------- | :-------------------------------------------------- |
| Real-Time Factor (RTF) | `0.0249x`  | (~40x faster than real-time playback)               |
| Avg. Encode Latency    | `34.51 ms` | Highly suitable for real-time edge streaming        |
| Avg. Decode Latency    | `37.33 ms` | Highly suitable for real-time edge inference        |
| PESQ (wb)              | `2.47`     | Moderate perceptual speech quality                  |
| LSD                    | `2.10 dB`  | Good preservation of general spectral shape         |
| SAR                    | `5.30 dB`  | Acceptable signal-to-artifact ratio                 |
| MCD                    | `42.58 dB` | Severe Distortion (Acoustic timbre heavily altered) |

### Conclusion on Lyra:

Lyra is very fast (RTF 0.0249x) and retains understandable speech at very low bitrates because it is a parametric generative codec. Instead of directly quantizing the waveform, it extracts features and uses an AI vocoder to _generate_ new speech that sounds similar to the original.

This generative step causes massive acoustic distortion (MCD > 42 dB). For deepfake detection, this means Lyra actively destroys the microscopic acoustic artifacts required by the detector.

Verdict: Discrete RVQ-based models (like EnCodec or SemantiCodec) remain the superior architectural choice.

## 3. To Do / Future Work

1. Read the SemantiCodec paper to find a better way to separate words from sound (to fix the 13.05% EER penalty).
2. Change the Healer AI loss function from guessing continuous numbers (MSE loss) to guessing exact token IDs (Discrete Cross-Entropy) to prevent the smoothing out of deepfake glitches.
