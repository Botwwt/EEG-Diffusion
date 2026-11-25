# Does moving the foreground recover the viewed photograph?

With the true pose of a previously studied photograph, this editor increased independent DINO identification among ten photos of the same object from 12/120 to 19/120. With EEG-predicted pose on the three personal12 menus, the eight-support pose prior identified 45/360 exact photos, matched EEG 44/360, and exchanged EEG 42/360. Matched EEG showed no reliable additional orientation benefit over the support prior. [Old-panel oracle](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_old_oracle.json) · [Personal12 image results](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_evaluation.json)

## From a pose prediction to pixels

Each of the 360 participant-photo queries starts from the saved four-step mixed trial/Fisher reconstruction with its original word and seed. U2NetP estimates a soft foreground on the 512-pixel center crop. Its full 2×2 covariance gives a pair of principal axes, their scales and the foreground's orientation in the frame. The editor transforms the recipient's generated foreground, carrying its existing appearance with it. Each of the three EEG/visual-prior arms sees the same source JPEG, with an additional neutral JPEG re-encoding. The calibration images supply pose values to the decoder. The editor receives the frozen numerical packet. [Word and H/14 source](PERSONAL_OBJECT_INFERENCE.md) · [Pose decoder design](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_readout_design.json) · [Image intervention](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_image_design.json)

The requested transform covers 80% of the center, angle and log-principal-scale difference. Displacement is capped at 12% of frame size per axis, rotation at 30°, and each scale factor at 0.75–1.33. As either ellipse approaches a circle, the angle contribution decreases. The implementation limits the matrix further when its visible foreground would cross the frame. A coarse-to-fine fill extends colors from the visible background into areas exposed by the move. Bilinear resampling transports the foreground and its opacity. All arms use the same quality-95 JPEG encoding. This operation preserves the existing view of the object, which limits its ability to portray a different three-dimensional viewpoint. [Editor source](../research/eeg_stack_20261112/code/edit_personal_pose.py)

The synthetic ellipse retained every pixel under an identity transform. For a requested rotation of about 20°, the remaining angle error was 3.96° and the centroid error fell from 0.0737 to 0.0143 of frame size. All three edge fixtures kept the visible foreground inside the frame. In the old-panel oracle, remeasured anisotropy error declined from 0.4390 to 0.3155, RGB RMSE from 0.3202 to 0.3127, and category identification stayed at 86/120. Its exact-photo DINO increase was 5.83 percentage points (95% paired object interval 0.83 to 11.67). Only this previously studied panel supplied target pose as an editing condition. [Synthetic and image audit](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_edit_receipt.json) · [Oracle measures and uncertainty](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_old_oracle.json)

The personal12 run produced 1,440 JPEGs. In 100/360 matched EEG edits, the image border limited the requested rotation or scaling. Seventeen matched transforms reduced to identity. Matched EEG and the visual prior produced different JPEGs on 345/360 rows. The matched and exchanged arms differed on 305 of the 317 rows with a distinct same-word, same-fold donor. All 43 self-donor rows produced identical JPEGs in these two arms. The independent scorer reopened the files, resegmented the edited foreground, and compared it with the reference after all three image manifests were sealed. [Rendering receipts](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_edit_receipt.json) · [Independent evaluation protocol](../research/eeg_stack_20261112/results/object_fidelity/personal12_pose_evaluation_design.json)

## Replay

From the repository root, restore the official U2NetP ONNX model, the official DINOv2-S/14 model and the original THINGS training photo archive under `outputs/photo-structure`. The old oracle also needs the `registered/single-1` four-step matched images from the fixed-word study. The personal12 edit needs the three frozen mixed-head reconstruction folders and the pose decoder outputs under `outputs/personal-pose/personal12/<subject>`. The measured run used Python 3.12, ONNX Runtime DirectML, NumPy, Pillow, SciPy and PyTorch CUDA. The old oracle reads the old panel references for its specific upper-bound test. The `edit` stage reads the three EEG pose packets and existing generated JPEGs. [Registered source generation](FIXED_WORD_PHOTO_DOSE.md) · [Asset preparation](PHOTO_STRUCTURE.md) · [Scorer](../research/eeg_stack_20261112/code/score_personal12_pose.py)

```sh
export OPENBLAS_NUM_THREADS=3 OMP_NUM_THREADS=3
C=research/eeg_stack_20261112/code
python "$C/edit_personal_pose.py" synthetic
python "$C/edit_personal_pose.py" old-oracle
python "$C/edit_personal_pose.py" edit
python "$C/score_personal12_pose.py" score
```

Per-person inputs, seeds, full JPEG hashes and applied affine matrices are recorded in `outputs/personal-pose/renders/personal12/<subject>/`. The shared `blind_images.json` binds the three completed settings files. The CLIP naming stage uses the offline archive and the frozen THINGS dictionary.

```sh
python "$C/score_personal12_pose.py" clip --archive "$ARCHIVE" --dictionary "$DICTIONARY"
```
