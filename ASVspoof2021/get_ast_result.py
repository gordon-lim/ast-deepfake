# -*- coding: utf-8 -*-
# @Time    : 11/15/20 1:04 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_esc_result.py

# summarize results from a single train-validation split

import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_path", type=str, default='', help="the root path of the experiment")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Load the result for the single train-validation split
    result = np.loadtxt(args.exp_path + '/result.csv', delimiter=',')
    assert result.ndim != 1, f"Expected 'result' to be a 2D array, but got {result.ndim}D instead. Did you train for at least 2 epochs?"
    
    # Compute metrics
    best_epoch = np.argmax(result[:, 0])  # Select the best epoch based on accuracy
    np.savetxt(args.exp_path + '/best_result.csv', result[best_epoch, :], delimiter=',')

    print('--------------Result Summary--------------')
    print('Best epoch accuracy: {:.4f}'.format(result[best_epoch, 0]))
    print('Results saved to', args.exp_path)