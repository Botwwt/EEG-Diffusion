"""Execution-only pinned-package setup; delegate unchanged D24 CPU mathematics."""
from pathlib import Path
import brainhive_paired_study_contract_20261025_v1 as C
R=Path(__file__).resolve().parent
C.setup_runtime(C.read(R/'configs/paired_effect_20261025_v1/study_review_v3.json'))
from audit_BrainHIVE_coordinate_gauge_CPU_20261028_v1 import main
if __name__=='__main__':main()
