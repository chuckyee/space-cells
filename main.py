#!/usr/bin/env python

from __future__ import print_function, division

from torch import nn

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss


def main():
    logging.info("Load dataset...")
    train_loader, val_loader = dataset.get_data_loaders(args)

    logging.info("Construct model...")
    model = select_model(args.model, args.nlabels, args.pretrained)
    logging.info("Number of trainable parameters: {}".format(
        model.num_trainable_parameters()))
    logging.info(str(model))

    # loss
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    OptimClass, kwargs = select_optimizer(args.optimizer, args.learning_rate)
    optimizer = OptimClass(model.parameters(), **kwargs)

    encoder = Encoder()
    decoder = Decoder()
    for epoch in range(args.epochs):
        viewpoint = random_viewpoint()
        x = Environment(viewpoint)
        for steps in range(args.rlsteps):
            viewpoint2 = random_viewpoint()
            z = Encoder(x)
            y_pred = Decoder(z, viewpoint2)
            y_true = Environment(viewpoint2)
            optimizer.zero_grad()
            loss = criterion(y_true, y_pred)
            loss.backward()
            optimizer.step()
    

if __name__ == '__main__':

    args = parameters.get_args(log=True)

    main(args)
