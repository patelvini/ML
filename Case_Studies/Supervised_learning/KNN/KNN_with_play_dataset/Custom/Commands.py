import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-tr', '--train', action = 'store_true', help = 'To train the model')
parser.add_argument('-ts', '--test', action = 'store_true', help = 'To test the model')
parser.add_argument('-w', '--whether', type = float )
parser.add_argument('-t', '--temperature', type = float)
parser.add_argument('-pr', '--predict', action = 'store_true', help = 'To predict label of the input data')

args = parser.parse_args()