import sys
import tensorflow as tf
import tensorflow_hub as hub

from model import Model

class MILModel(Model):

  def create_model(self):
    # Input and training status placeholder
    self.image_input = tf.placeholder(tf.float32, [None, self.config.image_height, self.config.image_width, self.config.image_channels])
      
    self.training = tf.placeholder(tf.bool)

    if self.config.mil_type == "vote":
      left_images, right_images = tf.split(self.image_input, 2, axis=1)
      left_cc, left_mlo = tf.split(left_images, 2, axis=2)
      right_cc, right_mlo = tf.split(right_images, 2, axis=2)
      inputs = [left_cc, left_mlo, right_cc, right_mlo]
    else:
      inputs = [self.image_input]

    self.logits = []
    for input in inputs:
      print("~~~")
      print(input.shape)

      l = self.encode(input)

      # Flatten output of previous layer, then feed into dense
      l = tf.layers.flatten(l)
      print(l.shape)

      if self.config.mil_type == "vote" and self.config.sigmoid_before_vote:
        l = tf.sigmoid(l)

      self.logits.append(l)

    print("---")
    if self.config.mil_type == "vote":
      self.stacked_logits = tf.stack(self.logits, axis=1)
      print(self.stacked_logits.shape)
      if self.config.vote_type == "nn":
        print("nn")
        self.stacked_logits = tf.reduce_mean(self.stacked_logits,axis=2)
        self.output = self.create_dense_layers(self.stacked_logits, [len(self.logits)])
      elif self.config.vote_type == "mean":
        print("mean")
        self.output = tf.reduce_mean(self.stacked_logits, axis=1, keepdims=True)
      elif self.config.vote_type == "max":
        print("max")
        self.output = tf.reduce_max(self.stacked_logits, axis=1, keepdims=True)
    else:
      self.output = self.logits[0]
    print(self.output.shape)
    print('--')

    # Network output
    self.probabilities = tf.nn.sigmoid(self.output)
    self.predictions = tf.math.round(self.probabilities)

    # Labels placeholder
    self.label_input = tf.placeholder(tf.float32, [None, 1])

    # Loss
    self.loss = -tf.reduce_mean(self.config.positive_error_rate_multiplier * self.label_input * tf.log(self.probabilities) + (1-self.label_input) * tf.log(1-self.probabilities))
    self.loss += self.config.regularization_loss_weight * tf.reduce_mean(tf.losses.get_regularization_losses())

class BaselineMILModel(MILModel):

  def encode(self, input):
      
    if self.config.large:
      conv_layers = [(32, 7, 2, 2), (64, 5, 2, 2), (128, 3, 2, 2), (256, 3, 1, 2)]
    else:
      conv_layers = [(32, 5, 1, 3), (64, 5, 1, 3), (128, 3, 1, 2), (256, 3, 1, 2)]

    l = self.create_conv_layers(input, conv_layers)

    # Flatten output of previous layer, then feed into dense
    l = tf.layers.flatten(l)
    print(l.shape)

    dense_layers = [4096, 128]
    l = self.create_dense_layers(l, dense_layers)
    return l

class TransferMILModel(MILModel):
  def __init__(self, model_name, config):
    module_spec = hub.load_module_spec(config.tfhub_module)
    if config.trainable:
      self.module = hub.Module(module_spec, trainable=True, tags=['train'])
    else:
      self.module = hub.Module(module_spec, trainable=False)

    super(TransferMILModel, self).__init__(model_name, config)

    if config.trainable:
      self.log("Inception module is trainable!")
    else:
      self.log("Inception module is frozen!")

  def encode(self, input):

    l = self.module(input)

    # Flatten module output and add dense layers
    l = tf.layers.flatten(l)
    
    self.training = tf.placeholder(tf.bool)
    dense_layers = [128]
    l = self.create_dense_layers(l, dense_layers)
    return l