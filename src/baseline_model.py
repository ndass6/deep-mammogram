import sys
import tensorflow as tf

from model import Model


class BaselineModel(Model):

  def create_model(self):
    # Input and training status placeholder
    self.image_input = tf.placeholder(tf.float32, [None, self.config.image_height, self.config.image_width, self.config.image_channels])
    self.training = tf.placeholder(tf.bool)

    l = self.image_input
    print(l.shape)

    # conv_layers = [(32, 5, 2, 2), (64, 5, 2, 2), (128, 3, 1, 2), (256, 3, 1, 2)]
    conv_layers = [(32, 5, 2, 2), (64, 5, 2, 2), (128, 3, 1, 2)]
    l = self.create_conv_layers(l, conv_layers)


    # Flatten output of previous layer, then feed into dense
    l = tf.layers.flatten(l)
    print(l.shape)

    dense_layers = [2048, 512]
    l = self.create_dense_layers(l, dense_layers)

    self.output = l

    # Network output
    self.probabilities = tf.nn.sigmoid(self.output)
    self.predictions = tf.math.round(self.probabilities)

    # Labels placeholder
    self.label_input = tf.placeholder(tf.float32, [None, 1])

    # Loss
    self.loss = -tf.reduce_mean(self.config.positive_error_rate_multiplier * self.label_input * tf.log(self.probabilities) + (1 - self.label_input) * tf.log(1 - self.probabilities))
    self.loss += self.config.regularization_loss_weight * tf.reduce_mean(tf.losses.get_regularization_losses())
