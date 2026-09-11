import json,argparse
from train_entity import train
if __name__=='__main__':
 x=json.load(open('output/v5-smoke/prepared/SETUP.json'));a=argparse.Namespace(**x['arguments']);a.stop_after_updates=1
 train(a)
