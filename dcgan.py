import os
from utils import *
from model import DCGAN


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

    model = DCGAN(DEVICE, bs_size, loader_train, loader_test, loader_val, latent_dimension, nb_filter, lr_rate, path)
    model.summary()

    #######################################################################
    #                       ** MODEL TRAINING **
    #######################################################################

    generator_training_loss, discriminator_training_loss = model.perform_training(n_epochs)
    # plot(generator_training_loss, discriminator_training_loss, path + '/training_losses.png')
    model.save()
    pass


#######################################################################
#                       ** PARAMETERS **
#######################################################################
PATH_RESULTS = 'GAN/results'

#######################################################################
#                       ** HYPERPARAMETERS **
#######################################################################
num_epochs = 200
learning_rate = 0.0002
latent_vector_size = 100

neg_slope = -0.2
nb_filter = 32
batch_size = 64
image_size = 32*32
nb_color = 3


GAN_training(PATH_RESULTS, batch_size, num_epochs, learning_rate, latent_vector_size)
