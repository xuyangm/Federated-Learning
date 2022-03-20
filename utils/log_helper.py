from torch.utils.tensorboard import SummaryWriter


def record_test_result(cur_rd, model_name, partition_type, sample_method, sample_sz, acc, loss):
    writer = SummaryWriter("logs-cifar10/test-{}-{}-{}-{}".format(model_name, partition_type, sample_method, sample_sz))
    writer.add_scalar("loss", loss, cur_rd)
    writer.add_scalar("accuracy(%)", acc * 100, cur_rd)
    writer.close()


def record_val_result(cur_rd, model_name, partition_type, sample_method, sample_sz, acc, loss):
    writer = SummaryWriter("logs-cifar10/val-{}-{}-{}-{}".format(model_name, partition_type, sample_method, sample_sz))
    writer.add_scalar("loss", loss, cur_rd)
    writer.add_scalar("accuracy(%)", acc * 100, cur_rd)
    writer.close()