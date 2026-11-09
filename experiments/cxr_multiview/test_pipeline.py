"""Small checks for label grouping, bag isolation, and submission integrity."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from data import StudyDataset, canonical_image, collate_studies
from metrics import evaluate, fit_thresholds
from model import MultiViewModel
from predict import write_submission
from train import ImageBudgetSampler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"artifacts/cxr_multiview/data"


class TestPipeline(unittest.TestCase):
    def test_raw_and_cache_match(self):
        for split in ["train", "valid", "test"]:
            record = json.loads((DATA/f"{split}.json").read_text())[0]
            for image in record["images"]:
                np.testing.assert_array_equal(np.asarray(canonical_image(image["raw"])),
                                              np.asarray(canonical_image(image["cache"])))

    def test_data_and_image_budget(self):
        dataset = StudyDataset(DATA/"train.json", size=32, training=True)
        sampler = ImageBudgetSampler(dataset, 8, training=True)
        seen = []
        for indices in sampler:
            self.assertLessEqual(sum(min(len(dataset.records[i]["images"]), 2) for i in indices), 8)
            seen.extend(indices)
        self.assertEqual(sorted(seen), list(range(len(dataset))))
        images, views, bags, labels, indices = collate_studies([dataset[i] for i in [0, 1, 2]])
        self.assertEqual(len(images), len(views))
        self.assertEqual(len(images), len(bags))
        self.assertEqual(tuple(labels.shape), (3, 10))
        self.assertEqual(labels[0, 9], 1)

    def test_bag_isolation_and_permutation(self):
        # A tiny encoder makes this a focused test of multi-view fusion, not Swin.
        model = MultiViewModel.__new__(MultiViewModel)
        nn.Module.__init__(model)
        model.pooling = "attention"
        model.encoder = nn.Identity()
        model.view_embedding = nn.Embedding(3, 4)
        model.dropout = nn.Identity()
        model.classifier = nn.Linear(4, 10)
        model.attention = nn.Linear(4, 10)
        model.eval()
        features = torch.randn(3, 4)
        views, bags = torch.tensor([1, 2, 1]), torch.tensor([0, 0, 1])
        combined, _ = model(features, views, bags, 2)
        first, _ = model(features[:2], views[:2], torch.tensor([0, 0]), 1)
        second, _ = model(features[2:], views[2:], torch.tensor([0]), 1)
        torch.testing.assert_close(combined, torch.cat([first, second]))
        perm = torch.tensor([2, 0, 1])
        reordered, _ = model(features[perm], views[perm], bags[perm], 2)
        torch.testing.assert_close(reordered, combined)

    def test_multilabel_pooling_can_predict_multiple_classes(self):
        model = MultiViewModel.__new__(MultiViewModel)
        nn.Module.__init__(model)
        model.pooling = "noisy_or"
        model.encoder = nn.Identity()
        model.view_embedding = nn.Embedding(3, 4)
        model.dropout = nn.Identity()
        model.classifier = nn.Linear(4, 10, bias=False)
        model.attention = nn.Linear(4, 10)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.weight[0, 0] = 12
            model.classifier.weight[1, 1] = 12
            model.view_embedding.weight.zero_()
        x = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        study, _ = model(x, torch.tensor([1, 2]), torch.tensor([0, 0]), 1)
        self.assertTrue(torch.all(study[0, :2] > 2))
        self.assertFalse(torch.isclose(torch.softmax(study[0], 0)[:2].sum(), torch.tensor(1.0)))

    def test_metrics_and_submission(self):
        y = np.vstack([np.zeros(10), np.ones(10)])
        p = y*0.8+0.1
        metrics = evaluate(y, p, fit_thresholds(y, p))
        self.assertEqual(metrics["macro_auc"], 1)
        self.assertEqual(metrics["macro_f1_tuned"], 1)
        records = [{"subject_id": "subject_a", "study_id": "study_a"},
                   {"subject_id": "subject_b", "study_id": "study_b"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"submission.csv"
            self.assertEqual(write_submission(records, p, [0.5]*10, path), 20)
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), ["Subject_id", "Study_id", "Predict_class", "Probability", "Gate"])
            self.assertEqual(frame.Predict_class.tolist(), list(range(1, 11))*2)


if __name__ == "__main__":
    unittest.main()
