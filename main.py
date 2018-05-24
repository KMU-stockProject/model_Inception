"""
	Deep learning project for predicting stock trend with tensorflow.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Main file for run project.

	:copyright: Hwang.S.J.
	:license: MIT LICENSE 1.0 .
"""

import os
import sys

from learningModel.inception import Inception
from preprocess.preprocessing import Preprocessing


if __name__ == '__main__':
    # pre = Preprocessing()
    # pre.selectData()
    # pre.preprocessing()
    arg = sys.argv
    learning = Inception()
    print('okok')
    learning.run()
