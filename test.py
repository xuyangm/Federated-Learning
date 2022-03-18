import unittest
from model_manager import *
from data_manager import DatasetCreator
from torch.utils.data import DataLoader


class MyTestCase(unittest.TestCase):
    def test_get_loader(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        training_loader = data_creator.get_loader(client_id=1, batch_sz=20)
        test_loader = data_creator.get_loader(batch_sz=20, is_test=True)
        self.assertIsInstance(training_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

    def test_get_training_data_len(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        training_data_len = data_creator.get_training_data_len()
        self.assertEqual(training_data_len, 60000)

    def test_get_training_data_len(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        test_data_len = data_creator.get_test_data_len()
        self.assertEqual(test_data_len, 10000)

    def test_partition(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        training_data_len = data_creator.get_training_data_len()
        for i in range(100):
            self.assertEqual(data_creator.get_training_data_len(i+1), training_data_len/100)

        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='dirichlet')
        for i in range(100):
            self.assertGreater(data_creator.get_training_data_len(i+1), 0)

    def test_model_manager(self):
        model_mgr = ModelManager('TwoNN', num_classes=10)
        self.assertIsInstance(model_mgr.model, TwoNN)
        model_mgr = ModelManager('CNN', num_classes=10)
        self.assertIsInstance(model_mgr.model, CNN)
        model_mgr = ModelManager('shufflenet_v2_x2_0', num_classes=10)
        self.assertIsInstance(model_mgr.model, torchvision.models.shufflenetv2.ShuffleNetV2)
        model_mgr = ModelManager('mobilenet_v2', num_classes=10)
        self.assertIsInstance(model_mgr.model, nn.Module)

    def test_train_test(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        training_loader = data_creator.get_loader(client_id=1, batch_sz=20)
        test_loader = data_creator.get_loader(batch_sz=20, is_test=True)
        model_mgr = ModelManager(model_name='TwoNN', num_classes=10)

        model_mgr.train(training_loader, 1)
        acc, loss = model_mgr.test(test_loader)
        self.assertIsInstance(acc, float)
        self.assertIsInstance(loss, float)


if __name__ == '__main__':
    unittest.main()
