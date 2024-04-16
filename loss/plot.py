#!/bin/python3
import pandas
import argparse
import numpy
import matplotlib.pyplot as plt
import addcopyfighandler

ap = argparse.ArgumentParser()
ap.add_argument('filename')
ap.add_argument('-e', '--epoch', default=128)

args = ap.parse_args()
loss = pandas.read_csv(args.filename, usecols=[0], names=['loss'])
avg = float(numpy.mean(loss))
plt.title(args.filename + "; avg=" + f'{avg:.2f}')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss[:args.epoch])
plt.show()

