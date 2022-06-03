from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import ELU, LeakyReLU
from dataset import get_data_invivo
from dataset import get_data_exvivo
from generate_model import generate_model
from finetune import finetune_exvivo
import random
import numpy as np
from train import train
from statistics.get_roc import get_roc
from statistics.get_prc import get_prc
from opts import parse_opts
from test_exvivo import test_exvivo


def main(opt):

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    if opt.data_type == 'invivo':
        x_train, y_train, x_val, y_val, x_test, y_test, \
        df_val, df_test = get_data_invivo(
            proj_dir=opt.proj_dir,
            benign_bix=opt.benign_bix,
            benign_nobix=opt.benign_nobix,
            pca_bix=opt.pca_bix,
            exclude_patient=opt.exclude_patient,
            x_input=opt.x_input)
        model = generate_model(
            init=opt.init,
            dropout_rate=opt.dropout_rate,
            momentum=opt.momentum,
            n_input=len(opt.x_input),
            n_layer=opt.n_layer)
        if opt.train:
            if opt.optimizer_function == 'adam':
                optimizer = Adam(learning_rate=opt.lr)
            train(
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                df_val=df_val,
                df_test=df_test,
                proj_dir=opt.proj_dir,
                batch_size=opt.batch_size,
                epoch=opt.train_epoch,
                loss=opt.loss_function,
                optimizer=optimizer)
        if opt.get_stat:
            roc_stat = get_roc(
                proj_dir=opt.proj_dir, 
                output_dir=opt.output_dir, 
                bootstrap=opt.bootstrap,
                data_type='invivo')
            prc_stat = get_prc(
                proj_dir=opt.proj_dir,
                output_dir=opt.output_dir,
                data_type='invivo')

    if opt.data_type == 'exvivo':
        x_train, y_train, x_test, y_test, df_test, x_test1, y_test1, \
        x_test2, y_test2, df_test1, df_test2 = get_data_exvivo(
            proj_dir=opt.proj_dir,
            exvivo_data=opt.exvivo_data,
            x_input=opt.x_input)
        # fine tune invivo model
        tuned_model = finetune_exvivo(
            x_train=x_train,
            y_train=y_train,
            proj_dir=opt.proj_dir,
            saved_model=opt.saved_model,
            batch_size=opt.batch_size,
            epoch=opt.finetune_epoch,
            freeze_layer=opt.freeze_layer)
        # model test
        test_exvivo(
            proj_dir=opt.proj_dir,
            output_dir=opt.output_dir,
            wu_split=opt.wu_split,
            x_test=x_test,
            y_test=y_test,
            df_test=df_test,
            x_test1=x_test1,
            y_test1=y_test1,
            x_test2=x_test2,
            y_test2=y_test2,
            df_test1=df_test1,
            df_test2=df_test2)
        if opt.get_stat:
            roc_stat = get_roc(
                proj_dir=opt.proj_dir,
                output_dir=opt.output_dir,
                bootstrap=opt.bootstrap,
                data_type='exvivo')
            prc_stat = get_prc(
                proj_dir=opt.proj_dir,
                output_dir=opt.output_dir,
                data_type='exvivo')


if __name__ == '__main__':


    opt = parse_opts()

    main(opt)








