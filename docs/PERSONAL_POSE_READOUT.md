# Decoding foreground pose from four EEG presentations

On 96 held-out objects, the single-person EEG readout reduced the normalized error of a photograph's foreground orientation from **1.164 with the eight-photo mean to 1.082**. Giving the decoder EEG from the other held-out photo of the same object produced **1.202**. Both paired comparisons passed the registered criterion. These are measurements of predicted foreground moments with a known object and eight labeled calibration photographs. The personal12 image experiment uses the decoded word and evaluates the resulting JPEGs separately. [Readout registration](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_readout_design.json) · [Measurements and packet hashes](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_readout.json) · [Generated-image evaluation design](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_evaluation_design.json)

| Participant | Eight-support mean | Own EEG | Other photo's EEG | Error reduction versus mean, 95% interval |
| --- | ---: | ---: | ---: | --- |
| sub-04 | 1.164 | 1.111 | 1.179 | 0.053 [0.028, 0.079] |
| sub-05 | 1.164 | 1.104 | 1.193 | 0.059 [0.031, 0.091] |
| sub-01 | 1.164 | 1.032 | 1.234 | 0.132 [0.077, 0.193] |
| Three participants | 1.164 | 1.082 | 1.202 | 0.081 [0.051, 0.116] |

The pooled reduction versus the other photo's EEG was 0.120 [0.087, 0.159]. For all five coordinates combined, mean error was 1.226 with the eight-support mean, 1.178 with own EEG and 1.253 with the other photo's EEG. Every generic query had a different-photo control. Both photos held out in a fold shared the same eight supports. The 10,000 bootstrap draws resampled 96 physical objects and preserved all three participants and their ten photos per object. There were 2,880 participant–photo decisions on 960 distinct photographs. This interval measures object variation for these three people.

## How the moment targets and EEG readout work

A fixed U2NetP estimates a continuous foreground weight at each pixel of a 320×320 version of the 512-pixel center crop. Weighted pixel centers give horizontal and vertical position. The log radius is one half of the natural logarithm of the trace of the central covariance matrix. Its last two coordinates are `(Cxx − Cyy) / trace` and `2 Cxy / trace`, in an image with x increasing rightward and y downward. Together these describe the anisotropy and in-plane axis of the mask. A nearly round mask has a short vector regardless of which axis a small perturbation would select. All 960 photos entered the primary error, including the 214 whose vector length fell below 0.20. Both components share one training-set scale. [Mask measurement](../research/eeg_stack_20261112/code/measure_personal_pose.py)

Each person's two ridge regressions took 1,003 posterior EEG features and 96 features from the single-person four-trial encoder, respectively. Training centered every object on its ten photographs. The fixed ridge penalty was 0.5 times the number of training rows. Four object-disjoint inner folds of the 416 training objects supplied a nonnegative calibration coefficient for each head and coordinate. Each inner neural encoder trained afresh on 312 objects and supplied representations for a 104-object calibration fold. The coefficients remained fixed through the 96-object evaluation and the final 512-object refit. The 96 development objects had appeared in earlier EEG studies. The earlier four-coordinate affine study scored 50 versus 52 out of 97 on strict two-photo assignment for the class-prior and own-EEG rendered pictures. [Earlier geometry experiment](PHOTO_STRUCTURE.md) · [EEG features](../research/eeg_stack_20261112/code/prepare_personal_pose_eeg.py) · [Readout and evaluation](../research/eeg_stack_20261112/code/fit_personal_pose.py)

## Rendering inputs

On personal12, each participant's previously frozen mixed classifier selected the object word and consequently the eight calibration photos. Its word accuracies were already known at registration: 65, 54 and 68 out of 120 for sub-04, sub-05 and sub-01. The three packets preserve each query's row, fold, word, eight supports and donor. The support prior is the mean of the eight pose vectors. Own EEG adds its predicted difference from those supports. The wrong arm uses the registered donor from the same word and fold, with self-assignment for singletons. The packets contain the five predicted coordinates and their scales. The measured query masks and photographs stay outside the packets. [Word readout and registration](TRIAL_OBJECT_PERSONAL12.md) · [Image transform design](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_image_design.json)

| Participant | Queries | Different-photo donors | SHA-256 of private pose packet |
| --- | ---: | ---: | --- |
| sub-04 | 120 | 107 | `f834cec3def2ce46ce6de62ba559703451b996344d8aaa8da5e4ffe383b5375d` |
| sub-05 | 120 | 108 | `54115450803cea7dc068f9f90d85e208ab78ed676eb7737728d10090f6c98fd3` |
| sub-01 | 120 | 102 | `c37dcbfc9930c43aa03851b875c6d7ea371c78580741942731ede255493f312c` |

## Replay

From the repository root, prepare the historical 512-class posterior and physical four-trial caches, the participant-specific mixed word packets, the independent 416-class neural checkpoints, and the earlier 1,520-row four-moment cache. The protocol binds their hashes. Place the original image archive under `outputs/photo-structure/training_images.zip`, its original row manifest under `outputs/official-object-readout/sub-03/atm_train_rows.csv` and the foreground model under `outputs/photo-structure/u2netp.onnx`. The run measured 5,120 generic photos and the 288 distinct personal12 photos used as a labeled support somewhere in the three participants' folds. All 5,408 yielded finite masks. For the 1,280 overlapping historical photos, recomputing the old four moments gave a maximum absolute discrepancy of 3.6×10⁻⁷. [Photo archive](https://huggingface.co/datasets/gasparyanartur/things-eeg2/tree/0e2ad986a26d90b6192cd69daeac32ff069c2460/imgs) · [U2NetP weights](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx) · [EEG release](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/tree/dbe34bb2407164f70883c661c76664f0d596e522/Preprocessed_data_250Hz) · [Historical mask cache](PHOTO_STRUCTURE.md#support-and-foreground) · [Independent neural checkpoints](SINGLE_PERSON_OBJECT_ENCODERS.md#learning-from-other-participants-viewing-the-same-photograph)

The measurement requires Pillow and ONNX Runtime, the neural fits use PyTorch and safetensors, and the readout uses NumPy and SciPy. The archived run used the DirectML ONNX Runtime provider, PyTorch 2.9.1 with CUDA and four BLAS threads for regression. On machines using the CPU ONNX Runtime build, choose `--provider CPUExecutionProvider`. Packet and report hashes also bind provider metadata and numerical results. The following commands reproduce the original settings with DirectML available to Python.

```sh
C=research/eeg_stack_20261112/code
P=research/eeg_stack_20261112/results/object_fidelity
W=outputs/personal-pose
mkdir -p "$W"
cp "$P/personal12_pose_readout_design.json" "$W/design.json"
OPENBLAS_NUM_THREADS=1 python "$C/measure_personal_pose.py" --out "$W" --provider DmlExecutionProvider
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
for s in sub-04 sub-05 sub-01; do
    python "$C/prepare_personal_pose_eeg.py" --subject "$s" --out "$W"
done
python "$C/fit_personal_pose.py" all --out "$W"
cmp "$W/result.json" "$P/personal12_pose_readout.json"
```

The private model coefficients and out-of-fold gains live under `$W/models/<subject>`, held-out moment predictions under `$W/generic/<subject>`, and rendering inputs under `$W/personal12/<subject>/{pose.npz,receipt.json}`. The public result contains summary statistics and input fingerprints. The renderer can use the three pose packets without access to the source photographs or the generic evaluation arrays.
