import os
from utils import *
from model import VAE
from torch.utils.tensorboard import SummaryWriter


def VAE_training(path, bs_size, n_epochs, lr_rate, latent_dimension):

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

    loader_train, loader_test = load_MNIST(bs_size, 'data/')
    sample_inputs, _ = next(iter(loader_test))
    fixed_input = sample_inputs[:2, :, :, :]

    save_img(fixed_input, path + '/image_original.png')

    #######################################################################
    #                       ** MODEL LOADING **
    #######################################################################
    writer = SummaryWriter()
    model = VAE(DEVICE, bs_size, loader_train, loader_test, latent_dimension,
                low_size, middle_size, high_size, lr_rate, path, writer)

    writer.close()
    model.summary()

    #######################################################################
    #                       ** MODEL TRAINING **
    #######################################################################

    training_loss, testing_loss = model.perform_training(n_epochs)
    plot(training_loss, testing_loss, path + '/training_losses.png')
    model.save(path + '/VAE_model.pth')


#######################################################################
#                       ** PARAMETERS **
#######################################################################
custom_path = 'results'

#######################################################################
#                       ** HYPERPARAMETERS **
#######################################################################
num_epochs = 1
learning_rate = 0.001
batch_size = 64
latent_dim = 10

low_size = 10 * 10
middle_size = 20 * 20
high_size = 28 * 28

VAE_training(custom_path, batch_size, num_epochs, learning_rate, latent_dim)
