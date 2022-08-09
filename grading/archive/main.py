from tensorflow.keras.layers import ELU, LeakyReLU
from dataset import get_data_invivo
from dataset import get_data_exvivo
from generate_model import generate_model
from finetune import finetune_exvivo
import random
import numpy as np
from train import train
from statistics.roc_all import roc_all
from statistics.prc_all import prc_all
from statistics.cm_all import cm_all
from opts import parse_opts
from test_exvivo import test_exvivo


def main(opt):

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    if opt.data_type == 'grading':
        x_train, y_train, x_val, y_val, x_test, y_test, \
        df_val, df_test = get_data(
            data_dir=opt.data_dir,
            x_input=opt.x_input)
    if opt.train:
        train(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            proj_dir=proj_dir,
            saved_model=saved_model,
            batch_size=opt.batch_size,
            epoch=opt.train_epoch,
            lock_base_model=True)
    if opt.get_stat:
        cm_all(
            proj_dir=opt.proj_dir,
            output_dir=opt.output_dir,
            data_type='invivo')
        roc_stat = get_roc(
            proj_dir=opt.proj_dir,
            output_dir=opt.output_dir,
            bootstrap=opt.bootstrap,
            data_type='invivo')
        prc_stat = get_prc(
            proj_dir=opt.proj_dir,
            output_dir=opt.output_dir,
            data_type='invivo')


if __name__ == '__main__':


    opt = parse_opts()

    main(opt)



