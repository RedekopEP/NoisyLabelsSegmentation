import torch
from torch.autograd import Variable
import time
from tqdm import tqdm
from utils import AverageMeter, calculate_accuracy
import numpy as np


def val_epoch(epoch, data_set, model, criterion, optimizer, opt, logger, writer):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()

    # data_set.file_open()
    valildation_loader = torch.utils.data.DataLoader(dataset=data_set,
                                                     batch_size=opt["validation_batch_size"],
                                                     shuffle=False,
                                                     pin_memory=True)
    val_process = tqdm(valildation_loader)
    start_time = time.time()
    for i, (inputs, targets) in enumerate(val_process):
        if i > 0:
            val_process.set_description("Loss: %.4f, Acc: %.4f" % (losses.avg, accuracies.avg))
        if opt["cuda_devices"] is not None:
            # targets = targets.cuda(async=True)
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
        with torch.no_grad():
            if opt["VAE_enable"]:
                outputs, outputs_dec, distr = model(inputs)
                loss = criterion(outputs, targets, outputs_dec, inputs, distr)
            else:
                if opt["Mheads_enable"]:

                    outputs1, outputs2, outputs3, outputs4, outputs5 = model(inputs)  # .cuda()
                    ooo = [outputs1, outputs2, outputs3, outputs4, outputs5]
                    res1 = calculate_accuracy(outputs1, targets)
                    res2 = calculate_accuracy(outputs2, targets)
                    res3 = calculate_accuracy(outputs3, targets)
                    res4 = calculate_accuracy(outputs4, targets)
                    res5 = calculate_accuracy(outputs5, targets)
                    res = [res1, res2, res3, res4, res5]
                    idx = res.index(max(res))
                    outputs = ooo[idx]
                    loss = criterion(input1=outputs1, input2=outputs2, input3=outputs3, input4=outputs4,
                                     input5=outputs5,
                                     target=targets)
                else:

                    outputs = model(inputs)  # .cuda()
                    # print(inputs.shape, targets.shape, outputs.shape)
                    loss = criterion(outputs, targets)
                    # loss = criterion(input=outputs, target=targets)
        if i == 10:
            img_batch = np.zeros((1, 1, outputs.shape[2] * 2, outputs.shape[3]))

            img_batch[0] = np.concatenate(
                (outputs.cpu().detach().numpy()[0, :, :, :, 64],
                 targets.cpu().detach().numpy()[0, :, :, :, 64]),
                axis=1)

            writer.add_images('Images/train', img_batch, dataformats='NCHW', global_step=epoch)

        acc = calculate_accuracy(outputs.cpu(), targets.cpu())

        losses.update(loss.cpu(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

    epoch_time = time.time() - start_time
    # data_set.file_open()
    print("validation: epoch:{0}\t seg_acc:{1:.4f} \t using:{2:.3f} minutes".format(epoch, accuracies.avg,
                                                                                    epoch_time / 60))

    logger.log(phase="val", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    writer.add_scalar('Loss/validation', losses.avg.item(), global_step=epoch)
    writer.add_scalar('DiceMetric/validation', accuracies.avg.item(), global_step=epoch)

    return losses.avg, accuracies.avg
