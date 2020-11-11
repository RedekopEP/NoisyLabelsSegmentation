import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from utils import Logger, load_old_model
from train import train_epoch
from glob import glob
from models.unet import UNet
from models.unet_do import UNet_DO
from metrics import CombinedLoss, SoftDiceLoss, BCEDiceLoss, DiceLoss, DiceLoss_Mheads, BCE_MHeads
from dataset import BratsDataset
from validation import val_epoch
import glob
import os

config = {}  # dict()
config["cuda_devices"] = True
# config["labels"] = (1, 2, 4)
config["labels"] = (1,)  # change label to train
config["model_file"] = os.path.abspath("single_label_{}_dice.h5".format(config["labels"][0]))
config["initial_learning_rate"] = 5e-4  # 1e-4
config["batch_size"] = 1
config["validation_batch_size"] = 1
config["image_shape"] = (240, 240, 155)  # (64, 64,64)  # (160, 192, 128)
config["activation"] = "relu"
config["normalizaiton"] = "group_normalization"
config["mode"] = "trilinear"
config["n_labels"] = 1  # 3 # len(config["labels"])
config["all_modalities"] = ["t1ce", "flair"]  # ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["batch_size"]] + [config["nb_channels"]] + list(config["image_shape"]))
config["loss_k1_weight"] = 0.1
config["loss_k2_weight"] = 0.1
config["random_offset"] = False  # Boolean. Augments the data by randomly move an axis during generating a data
config["random_flip"] = True  # Boolean. Augments the data by randomly flipping an axis during generating a data
config["random_scale"] = True
config["random_shift"] = True
# config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["result_path"] = "./checkpoint_models/"
config["data_file"] = os.path.abspath("isensee_mixed_brats_data.h5")
config["training_file"] = os.path.abspath("isensee_mixed_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_mixed_validation_ids.pkl")
config["test_file"] = os.path.abspath("isensee_mixed_validation_ids.pkl")
config["saved_model_file"] = None
config["overwrite"] = False  # If True, will create new files. If False, will use previously written files.
config["L2_norm"] = 1e-5
config["patience"] = 7
config["lr_decay"] = 0.7
config["epochs"] = 300
config["checkpoint"] = True  # Boolean. If True, will save the best model as checkpoint.
config["label_containing"] = True  # Boolean. If True, will generate label with overlapping.
config["VAE_enable"] = False  # Boolean. If True, will enable the VAE module.
config["train_dropout"] = False #True
config["Mheads_enable"] = False


def main():
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
    }, str(model_path))
    start_epoch = 1
    writer = SummaryWriter('/home/eredekop/MedicalAI/runs/BraTS_Full_Flair_T1ce_TTA_Dice_240_128')
    model_path = '/MedicalAI/modelsBraTS/BraTS_Full_Flair_T1ce_TTA_Dice_240_128.pt' #/MedicalAI/modelsBraTS/BraTS_Full_Flair_T1ce_MHeads_Dice_240_128.pt'

    model = NvNet(config=config)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters,
                           lr=config["initial_learning_rate"],
                           weight_decay=config["L2_norm"])

    loss_function = DiceLoss()


    source = '/data/MICCAI_BraTS_2018_Data_Training/'
    img_paths = glob.glob(source + '*GG/*/*t1.nii.gz')

    train_file_names = []
    val_file_names = []

    # for j in range(285):
    #     train_file_names.append(img_paths[j])
    for j in range(210):
        if j < 140:
            train_file_names.append(img_paths[j])
        else:
            val_file_names.append(img_paths[j])

    for j in range(210, 285):
        if j < 255:
            train_file_names.append(img_paths[j])
        else:
            val_file_names.append(img_paths[j])

    training_data = BratsDataset(phase="train", config=config, file_names=train_file_names)
    validation_data = BratsDataset(phase="validate", config=config, file_names=val_file_names)

    train_logger = Logger(model_name=config["model_file"], header=['epoch', 'loss', 'acc', 'lr'])

    if config["cuda_devices"] is not None:
        model = model.cuda()
        # loss_function = loss_function.cuda()

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config["lr_decay"], patience=config["patience"])

    print("training on label:{}".format(config["labels"]))
    max_val_acc = 0.
    for i in range(start_epoch, config["epochs"]):
        try:
            train_epoch(epoch=i,
                        data_set=training_data,
                        model=model,
                        criterion=loss_function,
                        optimizer=optimizer,
                        opt=config,
                        logger=train_logger, writer=writer, scheduler=scheduler)
            val_epoch(epoch=i,
                      data_set=validation_data,
                      model=model,
                      criterion=loss_function,
                      optimizer=optimizer,
                      opt=config,
                      logger=train_logger, writer=writer)

            # val_loss, val_acc = val_epoch(epoch=i,
            #                               data_set=valildation_data,
            #                               model=model,
            #                               criterion=loss_function,
            #                               opt=config,
            #                               optimizer=optimizer,
            #                               logger=train_logger, writer=writer)
            ## scheduler.step(val_loss)
            # if val_acc > max_val_acc or i == config["epochs"] - 1:
            if i == config["epochs"] - 1:
                # max_val_acc = val_acc save_dir = os.path.join(config["result_path"], config["model_file"].split(
                # "/")[-1].split(".h5")[0]) if not os.path.exists(save_dir): os.makedirs(save_dir) save_states_path =
                # os.path.join(save_dir, 'epoch_{0}_val_loss_{1:.4f}_acc_{2:.4f}.pth'.format(i, val_loss,
                # val_acc)) states = { 'epoch': i + 1, 'state_dict': model.state_dict(), 'optimizer':
                # optimizer.state_dict(), } torch.save(states, save_states_path) save_model_path = os.path.join(
                # save_dir, "best_model_file.pth") if os.path.exists(save_model_path): os.system("rm " +
                # save_model_path) torch.save(model, save_model_path)

                # if i > 0 and (i % 20 == 0 or i == config["epochs"] - 1):
                # write_event(log, step, loss=np.mean(MD_val))
                save(i)
                print("ok")
        except KeyboardInterrupt:
            print('Ctrl+C, saving snapshot')
            save(i)
            # write_event(log, step, loss=np.mean(MD_val))
            print('done.')

            return


if __name__ == '__main__':
    main()
