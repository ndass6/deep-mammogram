#!/bin/bash
# python3 train.py \
#   --model_name "mil" \
#   --mil_type "stack" \
#   --image_dir "../data/stack_case_images_large" \
#   --positive_error_rate_multiplier 2. \
#   --learning_rate 1e-5 \
#   --regularization_loss_weight 1e-7 \
#   --normalize_input \
#   --image_height 1024 \
#   --image_width 1024 \
#   --large \
#   $1
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "vote" \
#  --vote_type "nn" \
#  --positive_error_rate_multiplier 1. \
#  --learning_rate 1e-7 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --image_height 598 \
#  --image_width 598 \
#  --image_channels 4 \
#  --augment
#  $1
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-7 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  $1
# python3 train.py --model_name "baseline" --normalize --augment $1
# python3 train.py --model_name "transfer" --normalize --augment $1

# KAIS OVERNIGHT TESTS
# mil w/ stack
# got decent results with this one
##python3 train.py \
##  --model_name "mil" \
##  --mil_type "stack" \
##  --image_dir "../data/stack_case_images" \
##  --learning_rate 1e-7 \
##  --positive_error_rate_multiplier 1. \
##  --regularization_loss_weight 1e-7 \
##  --normalize_input \
##  --augment \
##  --image_height 299 \
##  --image_width 299 \
##  --num_epochs 50 \
##  $1
# mil w/ stitch
# david got this
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stitch" \
#  --image_dir "../data/stitch_case_images" \
#  --learning_rate 1e-7 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 75 \
#  $1
# mil vote w/ max
# this has issues, strictly decreasing
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "vote" \
#  --vote_type "max" \
#  --positive_error_rate_multiplier 1. \
#  --learning_rate 1e-7 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --image_height 598 \
#  --image_width 598 \
#  --image_channels 4 \
#  --num_epochs 75 \
#  --batch_norm \
#  --augment
#  $1
# mil vote w/ nn
# david got this
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "vote" \
#  --vote_type "nn" \
#  --positive_error_rate_multiplier 1. \
#  --learning_rate 1e-7 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --image_height 598 \
#  --image_width 598 \
#  --image_channels 4 \
#  --num_epochs 75 \
#  --augment
#  $1
## mil vote w/ mean
## SKIP
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "vote" \
#  --vote_type "mean" \
#  --positive_error_rate_multiplier 1. \
#  --learning_rate 1e-7 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --image_height 598 \
#  --image_width 598 \
#  --image_channels 4 \
#  --num_epochs 75 \
#  --augment
#  $1
## mil vote w/ nn + transfer
# david got this
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "transfer_mil" \
#  --freeze \
#  --vote_type "nn" \
#  --positive_error_rate_multiplier 1. \
#  --learning_rate 1e-7 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --image_height 598 \
#  --image_width 598 \
#  --image_channels 4 \
#  --num_epochs 75 \
#  --augment
#  $1
#
## non-mil baseline
# david got these
#python3 train.py --model_name "baseline" --normalize --num_epochs 75 --augment $1
## non-mil transfer freeze
#python3 train.py --model_name "transfer" --freeze --normalize --num_epochs 75 --augment $1
## non-mil transfer no freeze
#python3 train.py --model_name "transfer" --normalize --num_epochs 75 --augment $1


## Hyperparameter tuning
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-8 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-9 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-10 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-5 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1

### USE LR = 1e-6

#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1. \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 0.75 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.5 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.75 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 2 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 2.5 \
#  --regularization_loss_weight 1e-7 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1

### USE multiplier = 1.25
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 32 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 1 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 4 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 8 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 64 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 128 \
#  --normalize_input \
#  --augment \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 50 \
#  $1

### USE BATCH SIZE = 16

#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --dropout \
#  --dropout_rate 0.0 \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 20 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --dropout \
#  --dropout_rate 0.1 \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 20 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --dropout \
#  --dropout_rate 0.2 \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 20 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --dropout \
#  --dropout_rate 0.3 \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 20 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --dropout \
#  --dropout_rate 0.4 \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 20 \
#  $1
#
#python3 train.py \
#  --model_name "mil" \
#  --mil_type "stack" \
#  --image_dir "../data/stack_case_images" \
#  --learning_rate 1e-6 \
#  --positive_error_rate_multiplier 1.25 \
#  --regularization_loss_weight 1e-7 \
#  --batch_size 16 \
#  --normalize_input \
#  --augment \
#  --dropout \
#  --dropout_rate 0.5 \
#  --image_height 299 \
#  --image_width 299 \
#  --num_epochs 20 \
#  $1


python3 train.py \
  --model_name "mil" \
  --mil_type "stack" \
  --image_dir "../data/stack_case_images" \
  --learning_rate 1e-6 \
  --positive_error_rate_multiplier 1.25 \
  --regularization_loss_weight 1e-7 \
  --batch_size 16 \
  --normalize_input \
  --augment \
  --dropout \
  --dropout_rate 0.2 \
  --image_height 299 \
  --image_width 299 \
  --num_epochs 75 \
  $1
