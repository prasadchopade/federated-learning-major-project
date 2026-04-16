"""CPU checks for errors that would otherwise corrupt a training experiment."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.phase5_common import image_filename, label_to_class
from preprocessing.phase5_hospital_training import PneumoniaDataset, train_one_epoch, validate, val_transform
from src.data_loader import HospitalDataManager, MedicalImageDataset
from src.models import get_model_weights, set_model_weights
from src.server import FederatedServer


class ModelStateTests(unittest.TestCase):
    def test_state_transfer_preserves_batchnorm_buffers_and_copies_values(self):
        local = nn.BatchNorm2d(3)
        local.train()
        local(torch.full((2, 3, 4, 4), 3.0))
        state = get_model_weights(local)
        shared = nn.BatchNorm2d(3)
        set_model_weights(shared, state)
        self.assertIn('running_mean', state)
        self.assertIn('num_batches_tracked', state)
        for name, value in local.state_dict().items():
            torch.testing.assert_close(shared.state_dict()[name], value)
        state['running_mean'].zero_()
        self.assertGreater(float(local.running_mean.sum()), 0)
        self.assertGreater(float(shared.running_mean.sum()), 0)

    def test_fedavg_weights_values_and_preserves_integer_counters(self):
        server = object.__new__(FederatedServer)
        clients = [
            {'weight': torch.tensor([1.0, 3.0]), 'num_batches_tracked': torch.tensor(2)},
            {'weight': torch.tensor([5.0, 7.0]), 'num_batches_tracked': torch.tensor(7)},
        ]
        state = server.federated_averaging(clients, [1, 3])
        torch.testing.assert_close(state['weight'], torch.tensor([4.0, 6.0]))
        self.assertEqual(state['num_batches_tracked'].dtype, torch.int64)
        self.assertEqual(state['num_batches_tracked'].item(), 7)
        torch.testing.assert_close(clients[0]['weight'], torch.tensor([1.0, 3.0]))

    def test_invalid_client_updates_are_rejected(self):
        server = object.__new__(FederatedServer)
        state = {'weight': torch.tensor([1.0])}
        cases = [([], []), ([state], []), ([state], [0]), ([state], [-1]),
                 ([state], [float('nan')]), ([state, {}], [1, 1])]
        for clients, sizes in cases:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                server.federated_averaging(clients, sizes)


class DataTests(unittest.TestCase):
    def test_label_variants_do_not_confuse_not_normal_with_normal(self):
        expected = {'Normal': 0, 'No Finding': 0, 'Pneumonia': 1,
                    'BACTERIAL': 1, 'viral pneumonia': 1, 'Lung Opacity': 1,
                    'No Lung Opacity / Not Normal': 2, 'Other': 2,
                    'Atelectasis|Effusion': 2}
        for label, target in expected.items():
            with self.subTest(label=label):
                self.assertEqual(label_to_class(label), target)

    def test_missing_images_raise_instead_of_becoming_normal_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing.png')
            dataset = MedicalImageDataset([missing], [1])
            with self.assertRaisesRegex(RuntimeError, 'missing.png'):
                dataset[0]

    def test_unknown_baseline_labels_raise(self):
        frame = pd.DataFrame([{'hospital_id': 'test', 'split': 'train',
                               'image_path': 'unused.png', 'label': 'unmapped'}])
        with self.assertRaisesRegex(ValueError, 'Unmapped labels'):
            HospitalDataManager('test', frame, {'Normal': 0})

    def test_manifest_paths_and_split_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            Image.new('RGB', (16, 16), (100, 100, 100)).save(directory / 'sample.png')
            manifest = directory / 'index.csv'
            pd.DataFrame([
                {'image_path': r'old\windows\sample.png', 'label': 'Pneumonia', 'split': 'test'},
                {'image_path': 'elsewhere/train.png', 'label': 'Normal', 'split': 'train'},
            ]).to_csv(manifest, index=False)
            dataset = PneumoniaDataset(manifest, directory, split='test', transform=val_transform)
            self.assertEqual(len(dataset), 1)
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertEqual(label, 1)
            self.assertTrue(torch.isfinite(image).all())
            self.assertEqual(image_filename(r'C:\images\sample.png'), 'sample.png')


class LocalTrainingTests(unittest.TestCase):
    def test_one_local_epoch_updates_parameters_and_evaluates(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
        loader = DataLoader(TensorDataset(torch.randn(6, 3, 2, 2), torch.tensor([0, 1, 2, 0, 1, 2])), batch_size=3)
        before = model[1].weight.detach().clone()
        loss, accuracy = train_one_epoch(model, loader, nn.CrossEntropyLoss(),
                                        torch.optim.Adam(model.parameters(), lr=0.01), 'cpu')
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model[1].weight))
        val_loss, val_accuracy = validate(model, loader, nn.CrossEntropyLoss(), 'cpu')
        self.assertTrue(torch.isfinite(torch.tensor(val_loss)))
        self.assertTrue(0 <= accuracy <= 100)
        self.assertTrue(0 <= val_accuracy <= 100)


if __name__ == '__main__':
    unittest.main()
