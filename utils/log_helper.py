from torch.utils.tensorboard import SummaryWriter


def record(cur_rd, model_name, partition_type, sample_method, acc, loss):
    writer = SummaryWriter("logs/{}-{}-{}".format(model_name, partition_type, sample_method))
    writer.add_scalar("loss", loss, cur_rd)
    writer.add_scalar("accuracy(%)", acc * 100, cur_rd)
    writer.close()