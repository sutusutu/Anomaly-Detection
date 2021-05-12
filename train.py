"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from config import Config
from tool.data import load_data
from model.model import TNT_GANomaly as Model


##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Config().parse()
    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Model(opt, dataloader)
    ##
    # TRAIN MODEL
    # torch.backends.cudnn.enabled = False
    model.train()

if __name__ == '__main__':
    train()
