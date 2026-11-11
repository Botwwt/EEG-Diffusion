"""Same native P coordinates; change only the domain on which EEG response is retained."""
from PKU_conditional_preservation_core_20261110_v317 import R,sha,write,save,states,load_states,fixed_inputs,constrain,check_fixture
ROOT=R/'runs/EEG_visitation_v320'
CASES=['P_visit_native','P_visit_forward','P_visit_mismatch']
def flags(case):
    assert case in CASES
    return dict(relative=False,anchor='free')
