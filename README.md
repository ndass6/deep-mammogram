### Deep Mammogram

The goal of this project is to improve the quality of classification on mammograms through transfer learning and multiple-instance learning (MIL).


### Data
The original data is from http://marathon.csee.usf.edu/Mammography/Database.html. The script used to preprocess the data into the format found in data/ddsm is in src/util/preprocess.py. To create the train/dev/test splits, go to src/util/ and run split_data.py. To create the data used for MIL models, go to src/util/ and run stack_and_stitch_case_images.py.

### Running
To run models, go to src/ and run train.py with the arguments that you want to pass in. We also provide a bash script, train.sh, in the same directory with an example of the command for train.py. The main arguments are model_name which represents the type of model to train and can be one of: mil, transfer_mil, transfer, or baseline. If training a MIL model, the other two important arguments are mil_type which can be one of: stack, stitch, or vote and vote_type (only applicable if mil_type is set to vote) which can be one of: nn, mean, or max. For more details about the rest of the arguments, see src/util/arg_parser.py.
