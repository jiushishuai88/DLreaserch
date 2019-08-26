import argparse
import utils
import logging
import time
import torch
from models import *
from torch.utils.tensorboard import SummaryWriter
import yaml
import easydict
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

# 设置train文件启动参数
parser = argparse.ArgumentParser(description='common train cifar10')
parser.add_argument('--work-path', required=True, type=str)
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

# 加载启动参数
args = parser.parse_args()
# 设置日志
logger = utils.Logger(log_file_name=args.work_path + '/log.txt',
                      log_level=logging.DEBUG,
                      logger_name='CIFAR').get_log()


def train(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    # 设置为tranin模式，仅当有dropout和batchnormal时工作
    net.train()

    train_loss = 0;
    correct = 0;
    total = 0;
    logger.info("====Epoch:[{}/{}]====".format(epoch + 1, config.epochs))
    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if config.mixup:
            inputs, targets_a, targets_b, lam = utils.mixup_data(inputs, targets, config.mixup_alpha, device)
            outputs = net(inputs)
            loss = utils.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += inputs.size()[0]
        if config.mixup:
            correct += (lam * predicted.eq * (targets_a)).sum().item() + (1 - lam) * predicted.eq(
                targets_b).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()
        if batch_index % 100 == 99:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, utils.get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total
    writer.add_scalar('test_loss', train_loss, global_step=epoch)
    writer.add_scalar('test_acc', train_acc, global_step=epoch)
    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, writer
    # 测试模式
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    logger.info("======validate =====".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size()[0]
            correct += predicted.eq(targets).sum().item()
    logger.info("corret:"+str(correct)+"total"+str(total))
    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = acc > best_prec
    utils.save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = acc


def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + '/event')

    # 加载配置文件
    with open(args.work_path + '/config.yaml') as f:
        config = yaml.load(f)
    config = easydict.EasyDict(config)
    logger.info((config))
    # 获取模型
    net = get_model(config)
    logger.info(net)
    logger.info("=====total parameters:" + str(utils.count_parameters(net)))
    device = 'cuda' if config.use_gpu else 'cpu'
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov
    )
    last_epoch = -1
    best_prec = 0

    # 加载训练过的模型继续训练
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = utils.load_checkpoint(ckpt_file_name, net, optimizer=optimizer)

    # 设置数据的格式转换
    transform_train = transforms.Compose(utils.data_augmentation(config))
    transform_test = transforms.Compose(utils.data_augmentation(config, is_train=False))
    train_loader, test_loader = utils.get_data_loader(transform_train, transform_test, config)
    logger.info("==============trian-test-file-pathL{}".format(config.dataset))
    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        lr = utils.adjust_learning_rate(optimizer, epoch, config)
        writer.add_scalar('learning_rate', lr, epoch)
        train(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (
                epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)
    writer.close()
    logger.info(
        "======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))


if __name__ == "__main__":
    main()
