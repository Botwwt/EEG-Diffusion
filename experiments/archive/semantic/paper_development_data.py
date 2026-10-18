"""Concept-isolated training subset around the unchanged author ATM loader."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from paper_main_runtime import digest,native_dataset


class ConceptTrainingDataset(Dataset):
    def __init__(self,base,split):
        self.base=base;self.source_rows=torch.tensor(split['train_flat_trial_indices'],dtype=torch.long)
        self.image_ids=torch.tensor(split['train_image_ids'],dtype=torch.long)
        concepts=split['train_concepts'];development=split['development_concepts']
        assert len(concepts)==1454 and len(development)==200 and not set(concepts)&set(development)
        assert len(self.source_rows)==58160 and len(self.image_ids)==14540
        assert torch.equal(self.source_rows,((self.image_ids[:,None]*4)+torch.arange(4)[None,:]).flatten())
        assert len(torch.unique(self.source_rows))==58160 and len(torch.unique(self.image_ids))==14540
        assert set((self.image_ids//10).tolist())==set(concepts)
        assert not set(self.image_ids.tolist())&set(split['development_all_image_ids'])
        assert split['official_test_locked'] and split['all_repeats_follow_image_and_concept']
        assert base.img_features.shape==(16540,1024) and base.text_features.shape==(1654,1024)
        self.label_map=torch.full((1654,),-1,dtype=torch.long)
        self.label_map[torch.tensor(concepts)]=torch.arange(1454)
        self.img_features=base.img_features[self.image_ids]
        self.text_features=base.text_features[concepts]
        assert torch.equal(self.image_ids[::10]//10,torch.tensor(concepts))
        assert torch.equal(base.labels[self.source_rows].long(),self.source_rows//40)

    def __len__(self):return len(self.source_rows)

    def __getitem__(self,index):
        source=int(self.source_rows[index]);eeg,label,text,text_feature,image,image_feature=self.base[source]
        assert int(label)==source//40
        return eeg,self.label_map[int(label)],text,text_feature,image,image_feature


def build_training_subset(source,cache_validation,images,split_path,work):
    validation=Path(cache_validation);contract=json.loads((validation/'data_contract.json').read_text())
    assets=json.loads((validation/'asset_manifest.json').read_text());split=json.loads(Path(split_path).read_text())
    ready=json.loads((validation/'results.json').read_text());assert ready['status']=='completed' and ready['EEG_rows_verified']==66160
    cache=Path(contract['output']['path']);assert cache.stat().st_size==contract['output']['bytes']
    assert digest(cache)==contract['output']['sha256']
    assert digest(assets['features']['path'])==assets['features']['sha256']
    assert digest(assets['image_manifest']['path'])==assets['image_manifest']['sha256']
    base=native_dataset(source,cache.parent.parent,images,Path(assets['features']['path']).parent,work)
    dataset=ConceptTrainingDataset(base,split)
    report=dict(status='completed',split_manifest_sha256=digest(split_path),cache_contract_sha256=digest(validation/'data_contract.json'),
        source_rows=58160,unique_images=14540,concepts=1454,development_concepts=200,all_four_repeats_follow_image=True,
        training_metric_gallery_contains_training_concepts_only=True,labels_remapped_for_training_accuracy_only=True,
        published_preprocessing_protocol=True,upstream_test_statistics_independence_not_established=True,
        physical_session_metadata_recovered=False,official_test_read=False)
    return dataset,report
