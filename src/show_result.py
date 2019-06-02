import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

def show_result(res_dict):
    score = res_dict['score']
    mean = res_dict['mean_score']

    plt.title('DQN plying CartPole')
    plt.plot(score)
    plt.plot(mean)
    plt.xlabel('Episode')
    plt.ylabel('total_reward')
    
    ymax = max(mean)
    xpos = mean.index(ymax)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    bbox_props = dict(boxstyle="square", pad=0.3, fc="w", ec="k", lw=0.72)
    plt.annotate('Average max: {}'.format(ymax), 
                  xy=(xpos, ymax), xytext=(xpos, ymax+5000),
                  arrowprops=arrowprops, bbox=bbox_props)
    plt.text(len(mean)-1, mean[-1], str(mean[-1]))
    plt.show()

def main(args):
    source = {}
    with open(args.file_name, 'r') as f:
        source = json.load(f)
    show_result(source)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()