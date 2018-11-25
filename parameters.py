import argparse
import logging

import torch


def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  
    # directories
    parser.add_argument('--checkpoint-dir', type=str, default='./',
      help="path to directory to output logs and model checkpoints")
  
    # hardware
    parser.add_argument('--gpus', type=int, default=None, nargs='+',
      help="device IDs of GPUs to use; uses all GPUs by default")
  
    # data pipeline
    parser.add_argument('--num-environments', type=int, default=1024,
      help="total number of different objects in the dataset")
    parser.add_argument('--height', type=int, default=256,
      help="input image height")
    parser.add_argument('--width', type=int, default=256,
      help="input image width")
  
    # model
    parser.add_argument('--model', type=str, default='drn1',
      help="model to use ([FILL ME IN])")
  
    # training / validation
    parser.add_argument('--num-epochs', type=int, default=50,
      help="total epochs to train")
    parser.add_argument('--num-viewpoints', type=int, default=16,
      help="number of viewpoints to query network during each batch")
    parser.add_argument('--train-batch-size', type=int, default=32,
      help="training batch size")
    parser.add_argument('--val-batch-size', type=int, default=None,
      help="validation batch size; defaults to training batch size")
    parser.add_argument('--save-every', type=int, default=0,
      help="interval to write model weights (in epochs); 0 to disable")
  
    # training schedule
    parser.add_argument('--optimizer', type=str, default='adam',
      help="optimizer: one of adam, rmsprop, sgd")
    parser.add_argument('--learning-rate', type=float, default=None,
      help="learning rate; see PyTorch optimizer docs for defaults")
    parser.add_argument('--patience', type=int, default=10,
      help="number of epochs to wait before lowering learning rate")
    parser.add_argument('--early-stopping', type=int, default=None,
      help="stop if val loss does not improve after specified epochs")
  
    # logging / files
    parser.add_argument('--logging', default='INFO',
      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
      help="logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument('--infile', default='',
      help="file for initial model weights")
    parser.add_argument('--outfile', default='weights-final.pt',
      help="file to save final model weights")
    parser.add_argument('--debug', action='store_true',
      help="dump first batch of images")
  
    args = parser.parse_args()
  
    return args


def set_defaults(args):
    # modifies `args` in place

    # automatically set GPU usage
    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))
        
    # default validation batch size to training batch size
    if args.val_batch_size is None:
        args.val_batch_size = args.train_batch_size


def get_args(log=True):
    args = parse_args()
    set_defaults(args)

    # setup logging level
    numeric_level = getattr(logging, args.logging.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s %(message)s'
    )

    if log:
        logging.info("Arguments for execution:")
        for k, v in sorted(vars(args).items()):
            logging.info("{} = {}".format(k, v))

    return args
