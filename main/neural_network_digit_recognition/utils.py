import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

dlc = dict(dlblue='#0096ff', dlorange='#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0',
           dldarkblue='#0D5BDC', dlmedblue='#4285F4')
dlblue = '#0096ff';
dlorange = '#FF9300';
dldarkred = '#C00000';
dlmagenta = '#FF40FF';
dlpurple = '#7030A0';
dldarkblue = '#0D5BDC';
dlmedblue = '#4285F4'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
plt.style.use('./deeplearning.mplstyle')


def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y


def widgvis(fig):
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False


def display_errors(model, X, y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    doo = yhat != y[:, 0]
    idxs = np.where(yhat != y[:, 0])[0]
    if len(idxs) == 0:
        print("no errors found")
    else:
        cnt = min(8, len(idxs))
        fig, ax = plt.subplots(1, cnt, figsize=(5, 1.2))
        fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.80])  # [left, bottom, right, top]
        widgvis(fig)

        for i in range(cnt):
            j = idxs[i]
            X_reshaped = X[j].reshape((20, 20)).T

            # Display the image
            ax[i].imshow(X_reshaped, cmap='gray')

            # Predict using the Neural Network
            prediction = model.predict(X[j].reshape(1, 400))
            prediction_p = tf.nn.softmax(prediction)
            yhat = np.argmax(prediction_p)

            # Display the label above the image
            ax[i].set_title(f"{y[j, 0]},{yhat}", fontsize=10)
            ax[i].set_axis_off()
            fig.suptitle("Label, yhat", fontsize=12)
    return (len(idxs))


# - We randomly select 64 rows from X, map each row back to a 20 pixel by 20 pixel grayscale
#   image and display the images together.
# - The label for each image is displayed above the image
def display_digits(X, y, grid_size):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print('The shape of X is: ' + str(X.shape))
    print('The shape of y is: ' + str(y.shape))
    m, n = X.shape
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

    widgvis(fig)
    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)
    plt.show()


def display_digit(X):
    """ display a single digit. The input is one digit (400,). """
    fig, ax = plt.subplots(1, 1, figsize=(0.5, 0.5))
    widgvis(fig)
    X_reshaped = X.reshape((20, 20)).T
    # Display the image
    ax.imshow(X_reshaped, cmap='gray')
    plt.show()


def plot_loss_tf(history):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    widgvis(fig)
    ax.plot(history.history['loss'], label='loss')
    ax.set_ylim([0, 2])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('loss (cost)')
    ax.legend()
    ax.grid(True)
    plt.show()
