from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import ELU, LeakyReLU
from get_data_invivo import get_data_invivo
from generate_model import generate_model
from train import train
from statistics.get_roc import get_roc
from statistics.get_prc import get_prc
from opts import parse_opts



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
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    x_train, y_train, x_val, y_val, x_test, y_test, \
    df_val, df_test = get_data_invivo(
        proj_dir=proj_dir,
        benign_bix=benign_bix,
        benign_nobix=benign_nobix,
        pca_bix=pca_bix,
        exclude_patient=exclude_patient,
        exclude_list=exclude_list,
        x_input=x_input)

    model = get_model(
        init=init,
        dropout_rate=dropout_rate,
        momentum=momentum,
        n_input=len(x_input),
        n_layer=n_layer)
    
    if opt.train:
        if opt.optimizer_function == 'adam':
            optimizer = Adam(learning_rate=lr)
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
            proj_dir=proj_dir,
            batch_size=batch_size,
            epoch=epoch,
            loss=loss_function,
            optimizer=optimizer)
    if opt.get_stat:
        roc_stat = get_roc(
            proj_dir=proj_dir, 
            output_dir=output_dir, 
            bootstrap=bootstrap,
            data_type=data_type)
        prc_stat = get_prc(
            proj_dir=proj_dir,
            output_dir=output_dir)

