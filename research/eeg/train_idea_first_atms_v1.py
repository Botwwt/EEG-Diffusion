"""Actual full/OOF ATMS training for the user-prioritized mechanism study.

The official direct-encoder branch is fixed prospectively. This is not a
performance-selected replacement for the older two-reference campaign.
"""
from train_full_reference import *
import inspect


class DirectATMS(Trainer):
    def train_prior(self, condition, subject, folder):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--role', choices=['full', 'oof0', 'oof1'], required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    assert config['study'] == 'idea_first_direct_ATMS_v1'
    assert config['seed'] == 1729
    assert config['condition_chain'] == 'REF_REG' and not config['prior_enabled']
    assert config['encoder_epochs_max'] == 100 and config['encoder_patience'] == 50
    assert config['subjects'] == [f'sub-{i:02d}' for i in range(1, 11)]
    assert digest(__file__) == config['idea_trainer_sha256']
    assert digest(ROOT/'train_full_reference.py') == config['base_trainer_sha256']
    assert digest(inspect.getfile(ATMS)) == config['official_atms_source_sha256']
    assert digest(SOURCE/'Generation/benchmark.sh') == config['official_benchmark_sha256']
    gate_path = ROOT/'audit/idea_first_prerequisite_budget_gate_v1.json'
    assert digest(gate_path) == config['budget_gate_sha256']
    assert json.loads(gate_path.read_text())['full_and_two_fold_training_and_condition_sampling_funded']
    parent = Path('/proc')/str(os.getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trainer = DirectATMS(config, args.role)
    training = set(trainer.t.tolist())
    all_t = set(trainer.protocol['image_rows']['T'])
    assert training.isdisjoint(trainer.protocol['image_rows']['D'] + trainer.protocol['image_rows']['V'])
    lock = dict(config_sha256=digest(config_path), role=args.role,
                condition_chain='REF_REG', training_array_rows=sorted(training),
                training_concepts=sorted({trainer.image_rows[i]['concept_id'] for i in training}),
                official_test_accessed=False,
                sampling_contract='Exact deployed raw native ATMS direct condition; no semantic prior')
    if args.role.startswith('oof'):
        fold = int(args.role[-1])
        held = {i for i in all_t if trainer.protocol['oof_fold_by_concept'][trainer.image_rows[i]['concept_id']] == fold}
        assert training.isdisjoint(held) and training | held == all_t
        lock.update(heldout_fold=fold, heldout_T_array_rows=sorted(held),
                    heldout_concepts=sorted({trainer.image_rows[i]['concept_id'] for i in held}))
    else:
        assert training == all_t
    path = trainer.output/('OOF_data_roles.json' if args.role.startswith('oof') else 'full_data_roles.json')
    if path.exists():
        assert json.loads(path.read_text()) == lock
    else:
        write(path, lock)
    try:
        trainer.run()
    except Exception:
        trainer.record(dict(stage='FAILED_OR_BUDGET_STOPPED_RESUMABLE', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
