import torch
from torch.autograd import Variable
from tqdm import tqdm

from utils import AverageMeter, calculate_accuracy
import numpy as np


def train_epoch(epoch, data_set, model, criterion, optimizer, opt, logger, writer, scheduler):
    print('train at epoch {}'.format(epoch))

    model.train()

    losses = AverageMeter()
    accuracies = AverageMeter()

    # sdata_set.file_open()
    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=opt["batch_size"],
                                               shuffle=True,
                                               pin_memory=True)
    training_process = tqdm(train_loader)
    for i, (inputs, targets) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Loss: %.4f, Acc: %.4f" % (losses.avg.item(), accuracies.avg.item()))

        if opt["cuda_devices"] is not None:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()

            targets = targets.type(torch.FloatTensor)
            targets = targets.cuda()
            # print(inputs.shape, targets.shape)
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
                loss = criterion(input1=outputs1, input2=outputs2, input3=outputs3, input4=outputs4, input5=outputs5,
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
        # imgs = inputs.cpu().detach().numpy()
        #
        # imgs_batch = np.zeros((imgs.shape[0], 1, imgs_d.shape[2] * 5, imgs_d.shape[3]))
        # for i in range(batch_size):
        #     imgs_batch[i] = np.concatenate(
        #         (outls_c.cpu().detach().numpy()[i],
        #          outls_n.cpu().detach().numpy()[i],
        #          (torch.sigmoid(logits2) >= thr).cpu().detach().numpy()[i],
        #          arr[i], aleatoric[i]),
        #         axis=1)
        #
        # writer.add_images('Images/train', img_batch, dataformats='NCHW', global_step=count)
        # logits = model(imgs.cuda()).cuda()
        acc = calculate_accuracy(outputs.cpu(), targets.cpu())
        losses.update(loss.cpu(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = 1e-4*(1-epoch/300)**0.9

    logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'acc': format(accuracies.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    writer.add_scalar('Loss/train', losses.avg.item(), global_step=epoch)
    writer.add_scalar('DiceMetric/train', accuracies.avg.item(), global_step=epoch)
    scheduler.step(accuracies.avg.item())

    # sdata_set.file_close()
