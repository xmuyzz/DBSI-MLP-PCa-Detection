from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import ELU, LeakyReLU
from dataset import get_data_invivo
from dataset import get_data_exvivo
from generate_model import generate_model
from train import train
from statistics.get_roc import get_roc
from statistics.get_prc import get_prc
from opts import parse_opts



def main(opt):

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    x_train, y_train, x_val, y_val, x_test, y_test, \
    df_val, df_test = get_data_invivo(
        proj_dir=opt.proj_dir,
        benign_bix=opt.benign_bix,
        benign_nobix=opt.benign_nobix,
        pca_bix=opt.pca_bix,
        exclude_patient=opt.exclude_patient,
        exclude_list=opt.exclude_list,
        x_input=opt.x_input)

    model = get_model(
        init=opt.init,
        dropout_rate=opt.dropout_rate,
        momentum=opt.momentum,
        n_input=len(x_input),
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
            optimizer=opt.optimizer)
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

    if opt.finetune:
        x_train, y_train, x_test, y_test, df_test, x_test1, y_test1, \
        x_test2, y_test2, df_test1, df_test2 = get_data_exvivo(
            proj_dir=opt.proj_dir,
            exvivo_data=exvivo_data,
            exclude_list=opt.exclude_list,
            x_input=x_input)

        tuned_model = finetune_exvivo(
            x_train=x_train,
            y_train=y_train,
            proj_dir=opt.proj_dir,
            saved_model=opt.saved_model,
            batch_size=opt.batch_size,
            epoch=opt.finetune_epoch,
            freeze_layer=opt.freeze_layer)

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
                data_type='invivo')


if __name__ == '__main__':

    parser.add_argument('--exclude_patient',
                        action='store_true',
                        help='If true, training is performed.')
    parser.set_defaults(exclude_patient=True)
    parser.add_argument('--data_type',
                        default='invivo',
                        type=str,
                        help='Mannual seed')

    opt = parse_opts()

    main(opt)








