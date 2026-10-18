# Attribution and asset rights

## Original code

Original research scripts were curated from the local MCED research bundle under its MIT license, retained in `LICENSE`. The software package distinguishes byte-identical historical files from release-time evaluation refactors. Standard identities are not claimed as new contributions.

## Upstream models and data

- [ATM / EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode): original model definitions are **not vendored**; historical adapters load and verify an external checkout. The native main revision is `49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d`. Cite [Li et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ba5f1233efa77787ff9ec015877dbd1f-Abstract-Conference.html).
- Diffusers, SDXL/SDXL-Turbo, IP-Adapter, CLIP and DINOv3 retain their original software/model licenses. Their weights are not distributed here.
- THINGS-EEG2 and stimulus photographs must be obtained from their original providers under the applicable terms. No raw participant EEG or stimulus photographs are included.

## Literature screenshots

`docs/assets/{nvol,brainhive,tmpd}_*` are excerpts of third-party papers, included with attribution for research discussion. **They are excluded from the repository's MIT code license.** They do not become our original artwork; further reuse must follow the respective rights holder's terms.

| Source | Version | Included material |
|---|---|---|
| [Wang et al., NVOL](https://arxiv.org/abs/2609.02582) | arXiv v1 | Fig. 1, PDF p.3; short contribution paragraph, p.2 |
| [Zheng et al., BrainHIVE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c5506f154e5f2203a7dca2bee2faeced-Abstract-Conference.html) | Figure/excerpt from arXiv:2602.07495v1; ICLR 2026 proceedings linked | Fig. 1, PDF p.4; selected complete sentence, p.2 |
| [Boys et al., TMPD](https://arxiv.org/abs/2310.06721) | arXiv v3 | Eq. (10), PDF p.5; short paragraph, p.5 |

The PDF hashes, crop coordinates and yellow-highlight coordinates are recorded in [`docs/paper_sources.json`](docs/paper_sources.json). Crops preserve the original typography; highlighting was added for discussion. No paper curves or results were redrawn as our data. Full PDFs are not redistributed.

## Generated examples

`docs/assets/ours_*` are unenhanced rectangular crops of an archived fixed-order generated-output montage. The report shows its first three queries and four of six conditions; reduced-EEG and reduced-plus-shrinkage conditions are omitted from the illustration, not from the aggregate record. Stimulus labels are shown; stimulus photographs are not. The oracle column is generated from a true-image embedding and is not an EEG-only result. Outputs remain subject to the generator/model terms.

