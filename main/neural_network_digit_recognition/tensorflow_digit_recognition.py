import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# %matplotlib widget
import matplotlib.pyplot as plt
import warnings

plt.style.use('./deeplearning.mplstyle')
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.autograph.set_verbosity(0)
from utils import *

np.set_printoptions(precision=2)

# 1. Load Dataset
# ***************
# - The data set contains 5000 training examples of handwritten digits
#   - Each training example is a 20-pixel x 20-pixel grayscale image of the digit.
#       - Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
#       - The 20 by 20 grid of pixels is “unrolled” into a 400-dimensional vector.
#       - Each training examples becomes a single row in our data matrix X.
#       - This gives us a 5000 x 400 matrix X where every row is a training example of a handwritten digit image.
# - The second part of the training set is a 5000 x 1 dimensional vector y that contains labels for the training set
#   - y = 0 if the image is of the digit 0, y = 4 if the image is of the digit 4 and so on.
X, y = load_data()

print('The shape of X is: ' + str(X.shape))
print('The shape of y is: ' + str(y.shape))

# 2. Visualizing the Data. Show random 8x8=64 images.
# ***************
display_digits(X, y, 8)

# 3. Build Tensorflow model
# ***************
# - Neural network has two Dense layers with ReLU activations followed by an output layer with a Linear activation.
# - Recall that our inputs are pixel values of images. Since the images are of size  20×20, this gives us  400 inputs.
tf.random.set_seed(1234)  # for consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(25, activation="relu", name="L1"),
        Dense(15, activation="relu", name="L2"),
        Dense(10, activation="linear", name="L3")
    ], name="my_model"
)

# 4. Print model info
# ***************
model.summary()
[layer1, layer2, layer3] = model.layers
W1, b1 = layer1.get_weights()
W2, b2 = layer2.get_weights()
W3, b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

# 5. Train Model
# ***************
# - Defines a loss function, SparseCategoricalCrossentropy and indicates the softmax should be included with
#    the loss calculation by adding from_logits=True)
# - Defines an optimizer. A popular choice is Adaptive Moment (Adam).
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
history = model.fit(
    X, y,
    epochs=20
)

# 6. Print Loss by Epoch
# ***************
# The .fit method returns a variety of metrics including the loss. This can be used to build Loss by Epoch plot.
plot_loss_tf(history)

# 7. Use model to predict image
# ***************
# The largest output is prediction.
# If the problem only requires a selection, that is sufficient. But if you need probability then use Softmax.
image = X[3654]
prediction = model.predict(image.reshape(1, 400))
print(f" predicting: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")
display_digit(image)

# 8. Apply Softmax to convert prediction value in probability
# ***************
prediction_p = tf.nn.softmax(prediction)
print(f" predicting. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")

yhat = np.argmax(prediction_p)
print(f"np.argmax(prediction_p): {yhat}")

# 9. Compare the predictions vs the labels for a random sample of 64 digits
# ***************
warnings.simplefilter(action='ignore', category=FutureWarning)
m, n = X.shape
fig, axes = plt.subplots(8, 8, figsize=(5, 5))
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

    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1, 400))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)

    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]},{yhat}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()
