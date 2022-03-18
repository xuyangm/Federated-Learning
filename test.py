import unittest
from utils.selection import select_clients
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
        model_mgr = Client('TwoNN', num_classes=10)
        self.assertIsInstance(model_mgr.model, TwoNN)
        model_mgr = Client('CNN', num_classes=10)
        self.assertIsInstance(model_mgr.model, CNN)
        model_mgr = Client('shufflenet_v2_x2_0', num_classes=10)
        self.assertIsInstance(model_mgr.model, torchvision.models.shufflenetv2.ShuffleNetV2)
        model_mgr = Client('mobilenet_v2', num_classes=10)
        self.assertIsInstance(model_mgr.model, nn.Module)

    def test_train_test(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        model_mgr = Client(model_name='TwoNN', num_classes=10)
        model_mgr.init_data(1, data_creator, 20)

        model_mgr.train(1)
        acc, loss = model_mgr.test()
        self.assertIsInstance(acc, float)
        self.assertIsInstance(loss, float)

    def test_model_avg(self):
        data_creator = DatasetCreator(num_clients=100, dataset_name='MNIST', partition_method='uniform')
        model_mgrs = []
        model_mgrs.append(Aggregator(model_name='TwoNN', num_classes=10))
        for i in range(2):
            model_mgrs.append(Client(model_name='TwoNN', num_classes=10))
        model_mgrs[0].init_data(data_creator, 20)
        model_mgrs[1].init_data(1, data_creator, 20)
        model_mgrs[2].init_data(2, data_creator, 20)
        model_mgrs[1].train(1)
        model_mgrs[0].retrieve_update(1, model_mgrs[1].get_model_state(), model_mgrs[1].get_training_data_len())
        model_mgrs[2].train(5)
        model_mgrs[0].retrieve_update(2, model_mgrs[2].get_model_state(), model_mgrs[2].get_training_data_len())
        model_mgrs[0].update_model()
        model_mgrs[0].test()

    def test_selection(self):
        cids = [x for x in range(10)]
        selected = select_clients(cids, 4, 'random')
        print(selected)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(MyTestCase('test_model_avg'))
    # unittest.TextTestRunner(verbosity=2).run(suite)
