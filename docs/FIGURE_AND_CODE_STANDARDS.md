# Figure and software reporting choices

This release follows the relevant practical guidance from the [Nature research figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) and [Nature code/software submission guidance](https://www.nature.com/documents/GuidelinesCodePublication.pdf). It is a research report and source archive; it has not undergone journal review.

## Statistical figures

- Source values are released separately. Rendering does not reconstruct observations or intervals that are unavailable.
- Each panel has its own modality, units, sample definition and interpretation. Fixed training seeds are shown as individual points; they are not labeled as confidence intervals.
- New plots use sans-serif text, explicit axes, black labels, accessible blue/grey marks, editable SVG text and vector PDF with embedded TrueType fonts. Statistical graphics use 7-pt text and 8-pt bold panel labels at their recorded export size.
- The plots omit background grids, shadows and decorative marks. Captioned intervals retain the exact historical resampling scope.
- Figure files, renderer version, dimensions and source-data hashes are recorded in `provenance/figure_manifest.json`.

Paper figures are attributed excerpts, not newly prepared publication artwork. Their source versions and highlight coordinates are retained. The generated examples are 160-pixel archival montage crops, suitable for discussion; a formal publication would require the original high-resolution outputs. Upsampling is not used to claim additional detail.

## Code and evidence

- The runnable numerical core has a small dependency set, typed interfaces, explicit array/covariance checks, documented assumptions, a deterministic synthetic demo and tests of analytic identities.
- The active code is formatted and checked with Ruff. Historical executed files are excluded from style rewrites and remain byte-identical; refactors have a separate provenance record.
- Released aggregate files have their own integrity manifest. Historical bootstrap intervals are retained rather than fabricated from means.
- Installation, expected demo output, environment, limitations, code availability, citation and licensing are documented. CI checks fresh Linux and Windows environments.

The full historical GPU workflow still depends on external EEG, stimulus access, feature caches and weights. This release is not a complete reproduction capsule for all image results. No DOI, independent external replication, prospective validation or journal endorsement is asserted. Those steps require the relevant data, final study and an archival release.

