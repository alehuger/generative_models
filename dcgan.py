import os
from utils import *
from model import DCGAN
from torch.utils.tensorboard import SummaryWriter
import argparse


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def GAN_training(path, bs_size, n_epochs, lr_rate, latent_dimension):

    #######################################################################
    #                       ** DEVICE SELECTION **
    #######################################################################

    GPU = True
    device_idx = 0
    if GPU:
        DEVICE = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")

    print('Type of device use : ', DEVICE)

    # Random seed for reproducible results
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    #######################################################################
    #                       ** DATA LOADING **
    #######################################################################

    if not os.path.exists(path):
        os.makedirs(path)

    loader_train, loader_test, loader_val = load_CIFAR(bs_size, 'GAN/data/')
    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[:2, :, :, :]
    save_img(fixed_input, path + '/image_original.png')

    #######################################################################
    #                       ** MODEL LOADING **
    #######################################################################
    writer = SummaryWriter(log_dir='GAN/runs')
    model = DCGAN(DEVICE, bs_size, loader_train, loader_test, loader_val, latent_dimension, nb_filter, lr_rate, path, writer)
    model.summary()
    writer.close()

    #######################################################################
    #                       ** MODEL TRAINING **
    #######################################################################

    generator_training_loss, discriminator_training_loss = model.perform_training(n_epochs)
    # plot(generator_training_loss, discriminator_training_loss, path + '/training_losses.png')
    model.save()


#######################################################################
#                       ** PARAMETERS **
#######################################################################
PATH_RESULTS = 'GAN/results'
neg_slope = -0.2
image_size = 32*32
nb_color = 3
#######################################################################
#                       ** HYPERPARAMETERS **
#######################################################################

num_epochs = 1
batch_size = 64
nb_filter = 32
learning_rate = 0.0002
latent_vector_size = 100

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", help="number of epochs")
parser.add_argument("-bs", "--batch_size", help="batch size")
parser.add_argument("-lr", "--learning_rate", help="learning rate")
parser.add_argument("-vs", "--vector_size", help="latent vector size")

args = parser.parse_args()


GAN_training(PATH_RESULTS, int(args.batch_size), int(args.n_epochs), float(args.learning_rate), int(args.vector_size))
