# THINGS-EEG reconstruction: published numbers next to ours (read 2026-11-13)

Every paper below was read in full (text, tables, appendix). Numbers are copied from the papers; nothing was re-run.
Metric implementations differ between papers (CLIP model, resolution, number of distractors, images per query are
often not stated), so the comparison is indicative, not a controlled benchmark.

## Inclusion rule (fixed before ranking)

1. The method's own **10-subject mean** on THINGS-EEG (Gifford et al. 2022, 200 test images, 80 repeats), intra-subject.
2. EEG-only inputs at test time. Rows that use information derived from the test stimulus are listed but not ranked.
3. Single-subject numbers (usually subject 8, the subject with the best retrieval) are listed but not ranked.
4. Third-party re-runs are listed separately as a sensitivity check.

## Ranked rows (PixCorr / SSIM / Alex2 / Alex5 / Incep / CLIP; two-way identification for the last four)

| method | source | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | notes |
|---|---|---|---|---|---|---|---|---|
| Brain-HIVE | paper Table 2 | 0.195 | 0.336 | 0.843 | 0.905 | 0.756 | 0.808 | host model of this work |
| Perceptogram (VD) | Table 2, mean ± SD over 10 | 0.267 | 0.347 | 0.910 | 0.927 | 0.752 | 0.807 | ridge α chosen on subject 1 |
| ENIGMA, single-subject | Table 1 | 0.1718 | 0.4233 | 0.8364 | 0.8949 | 0.7765 | 0.8148 | 10 seeds per image, averaged |
| ENIGMA, multi-subject | Table 1 | 0.1668 | 0.4264 | 0.8299 | 0.8912 | 0.7654 | 0.8033 | trains on the test subject's training data too |
| CognitionCapturer | Table 3 | 0.150 | 0.347 | 0.754 | 0.623 | 0.669 | 0.715 | Alex5 < Alex2 in every row (likely a transcription issue) |
| CognitionCapturerPro | Table III | 0.163 | 0.398 | – | – | 0.779 | 0.830 | subject count not stated |
| AVDE (ICLR 2026) | Table 4 / Table 13 mean | 0.147 | 0.366 | 0.766 | 0.835 | 0.724 | 0.747 | main Table 2 is subject 8 only |
| UMind | Table 3 | 0.156 | 0.390 | 0.725 | 0.839 | 0.744 | 0.798 | subject count not stated; "10 images ranked by similarity to the ground truth" (possible ground-truth selection) |
| JMVR*, EEG only | Table 1 | 0.215 | 0.367 | 0.821 | 0.877 | 0.778 | 0.809 | |
| D²-FOSA (CVPR 2026) | Table 3 | 0.193 | 0.350 | – | – | – | – | subject count not stated; reports only SSIM / PixCorr among these |
| NEED (NeurIPS 2025) | Table 2 | 0.162 | 0.352 | – | – | 0.742 | 0.795 | zero-shot transfer from video EEG; subject count not stated |

### This work (same metric code for every row, 10 subjects, K = 4 images per query)

| setting | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | rank among the ranked rows above |
|---|---|---|---|---|---|---|---|
| Brain-HIVE reproduced here | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 | |
| A″ (best of 8, shrinkage 16 px) | 0.2782 | 0.4043 | 0.8823 | 0.9255 | 0.7837 | 0.8251 | 1 / 2 / 2 / 2 / 1 / 2 |
| **B‴** (best of 16, shrinkage 24 px) | 0.2748 | 0.4007 | 0.8861 | 0.9270 | 0.7882 | 0.8278 | 1 / 2 / 2 / 1 (tie) / 1 / 2 |
| O3, original components only (no joint encoder, best-of-N, shrinkage, amplification; one draw) | 0.2636 | 0.4073 | 0.8727 | 0.9152 | 0.7581 | 0.8118 | 2 / 2 / 2 / 2 / 4 / 3 |
| O4, O3 + test-time-adapted start image (label-free, transductive; one draw) | 0.2687 | 0.4090 | 0.8753 | 0.9158 | 0.7573 | 0.8121 | 1 / 2 / 2 / 2 / 4 / 3 |
| **O5**, O4 + 7-seed ensemble condition with repeat bagging (one draw per image) | 0.2692 | 0.4108 | 0.8780 | 0.9176 | 0.7584 | 0.8139 | 1 / 2 / 2 / 2 / 4 / 3 |

A″ and B‴ contain components taken from MindEye / MindEye2 / ENIGMA / Perceptogram (see the README's prior-art table);
O3, O4 and O5 leave them out. Ranks count each method once (ENIGMA's single- and multi-subject rows as one).

Our reproduction of Brain-HIVE is 0.01–0.02 below the paper's own numbers, which suggests our metric code is not more
lenient than theirs. Best of 16 uses 16× the sampling budget of a single draw.

## Listed, not ranked

| method | source | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | why not ranked |
|---|---|---|---|---|---|---|---|---|
| JMVR, EEG + text | Table 1 | 0.236 | 0.372 | 0.874 | 0.920 | 0.793 | 0.829 | text = VLM captions of the stimulus image |
| ATM | Table 1 | 0.160 | 0.345 | 0.776 | 0.866 | 0.734 | 0.786 | subject 8 only |
| AVDE | Table 2 | 0.188 | 0.396 | 0.817 | 0.889 | 0.765 | 0.795 | subject 8 only |
| TS-Diff | Table III | 0.176 | 0.366 | – | – | 0.737 | 0.799 | subject 8 only |
| Perceptogram, re-run by ENIGMA | ENIGMA Table 1 | 0.247 | 0.431 | 0.8546 | 0.8803 | 0.7040 | 0.7198 | third-party re-run, different metric code |
| ATM-S, re-run by ENIGMA | ENIGMA Table 1 | 0.136 | 0.392 | 0.7385 | 0.8083 | 0.6756 | 0.7128 | third-party re-run |
| Saliency-guided diffusion (arXiv 2510.26391) | Tables 1–2 | 0.473 | 0.369 | 0.999 | 0.998 | 0.946 | 0.904 | subject 8; saliency maps computed from the ground-truth test images (EEG-only row: 0.080 / 0.271 / 0.774 / 0.865 / 0.745 / 0.767) |
| Course re-implementation of CogCapPro (arXiv 2605.23996) | Table 2 | 0.166 | 0.409 | 0.818 | 0.913 | 0.831 | 0.870 | one unnamed subject |
| ATM, as listed by UMind | UMind Table 3 | 0.164 | 0.513 | 0.583 | 0.603 | 0.558 | 0.577 | provenance unclear; inconsistent with ATM's own numbers |

## Sources

- [Brain-HIVE: Learning Brain Representation with Hierarchical Visual Embeddings (ICLR 2026), code](https://github.com/ssshamiii/Brain-HIVE) — host model, Table 2
- [Perceptogram, arXiv 2404.01250](https://arxiv.org/abs/2404.01250)
- [ENIGMA, arXiv 2602.10361](https://arxiv.org/abs/2602.10361)
- [CognitionCapturer, arXiv 2412.10489](https://arxiv.org/abs/2412.10489)
- [CognitionCapturerPro, arXiv 2603.12722](https://arxiv.org/abs/2603.12722)
- [AVDE, arXiv 2602.22555](https://arxiv.org/abs/2602.22555)
- [UMind, arXiv 2509.14772](https://arxiv.org/abs/2509.14772)
- [JMVR, arXiv 2603.19667](https://arxiv.org/abs/2603.19667)
- [ATM, arXiv 2403.07721](https://arxiv.org/abs/2403.07721)
- [TS-Diff, arXiv 2609.02582](https://arxiv.org/abs/2609.02582)
- [D²-FOSA, CVPR 2026 (CVF open access)](https://openaccess.thecvf.com/)
- [NEED, NeurIPS 2025 (proceedings)](https://proceedings.neurips.cc/)
- [Saliency-guided diffusion, arXiv 2510.26391](https://arxiv.org/abs/2510.26391)
- [Multimodal EEG alignment course project, arXiv 2605.23996](https://arxiv.org/abs/2605.23996)

Searched and excluded (no comparable table): SGDM (arXiv 2604.22649, own split and metrics), EEG-GAN (arXiv 2609.22282).
