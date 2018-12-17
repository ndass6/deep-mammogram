import sys
import tensorflow as tf
import tensorflow_hub as hub

from model import Model

class TransferModel(Model):

  def create_model(self):
    self.module_spec = hub.load_module_spec(self.config.tfhub_module)
    if self.config.trainable:
      self.log("Inception module is trainable!")
      self.module = hub.Module(self.module_spec, trainable=True, tags=['train'])
    else:
      self.log("Inception module is frozen!")
      self.module = hub.Module(self.module_spec, trainable=False)

    self.image_input = tf.placeholder(tf.float32, [None, self.config.image_height, self.config.image_width, self.config.image_channels])
    l = self.image_input

    l = self.module(l)

    # Flatten module output and add dense layers
    l = tf.layers.flatten(l)
    
    self.training = tf.placeholder(tf.bool)
    dense_layers = [128]
    l = self.create_dense_layers(l, dense_layers)

    self.output = l
    self.probabilities = tf.nn.sigmoid(self.output)
    self.predictions = tf.math.round(self.probabilities)

    # Labels placeholder
    self.label_input = tf.placeholder(tf.float32, [None, 1])

    # Loss
    cross_entropy_loss = -tf.reduce_mean(self.config.positive_error_rate_multiplier * self.label_input * tf.log(self.probabilities) + (1 - self.label_input) * tf.log(1 - self.probabilities))
    reg_losses = tf.reduce_mean(tf.losses.get_regularization_losses())
    self.loss = cross_entropy_loss + self.config.regularization_loss_weight * reg_losses
