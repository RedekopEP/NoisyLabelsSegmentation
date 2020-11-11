import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train SRGAN Model')
    parser.add_argument('--cuda_index', default=0, type=int, help='CUDA device index')
    parser.add_argument('--epochs', default=10, type=int, help='Train epoch number')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for train loader')
    parser.add_argument('--random_state', default=None, type=int, help='Random state')
    parser.add_argument('--train_path', type=str, help='Path to train data')
    parser.add_argument('--val_path', type=str, help='Path to val data')
    parser.add_argument('--uncertainty_method', type=str, help='Uncertainty estimation method', choices=['TTA', 'MCDO',
                                                                                                         'M-Heads',
                                                                                                         'Ensembles'])
    return parser.parse_args()
