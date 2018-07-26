#!/usr/bin/env python

from __future__ import print_function, division
from argparse import ArgumentParser

from torch import nn

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss


def main():
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
    parser = ArgumentParser()

    parser.add_argument()

    args = parser.parse_args()
    
    main(args)
