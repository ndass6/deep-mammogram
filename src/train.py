import sys
import tensorflow as tf

from mil_models import TransferMILModel, BaselineMILModel
from transfer_model import TransferModel
from baseline_model import BaselineModel
from util.arg_parser import parse_args


def main(_):
  # Create model and hardcode arguments that are specific to models
  if FLAGS.model_name == 'mil':
    assert FLAGS.mil_type is not None
    if FLAGS.mil_type == 'stack':
      FLAGS.image_channels = 4
    else: # stitch and vote
      FLAGS.image_dir = '../data/stitch_case_images/'
      FLAGS.image_height = 299 * 2
      FLAGS.image_width = 299 * 2
      FLAGS.image_channels = 3
    FLAGS.grayscale = True
    # FLAGS.augment = True
    model = BaselineMILModel('mil', FLAGS)
  elif FLAGS.model_name == 'transfer_mil':
    assert FLAGS.mil_type == 'vote'
    FLAGS.image_dir = '../data/stitch_case_images/'
    FLAGS.image_height = 299 * 2
    FLAGS.image_width = 299 * 2
    FLAGS.image_channels = 3
    FLAGS.grayscale = True
    FLAGS.augment = True
    model = TransferMILModel('mil', FLAGS)
  elif FLAGS.model_name == 'transfer':
    model = TransferModel('transfer_inception', FLAGS)
  elif FLAGS.model_name == 'baseline':
    FLAGS.learning_rate = 1e-7
    model = BaselineModel('baseline', FLAGS)
  else:
    raise Exception("Unrecognized model name '{}'".format(FLAGS.model_name))


  model.train()
  model.test('test')

      
if __name__ == '__main__':
  FLAGS, unparsed = parse_args()
  print("Training {} model!".format(FLAGS.model_name))
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
