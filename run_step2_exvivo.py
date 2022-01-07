from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import ELU, LeakyReLU
from get_data_invivo import get_data_invivo
from get_model import get_model
from train_invivo import train_invivo
from get_roc import get_roc
from get_prc import get_prc
from get_data_exvivo import get_data_exvivo
from finetune_exvivo import finetune_exvivo
from test_exvivo import test_exvivo


if __name__ == '__main__':

    proj_dir = '/home/xmuyzz/Harvard_AIM/others/pca'
    output_dir  = '/mnt/aertslab/USERS/Zezhong/others/pca/output'

    exvivo_data = 'pca_exvivo.csv'
    exclude_list = ['001_ZHOU_CHAO_GANG', '002_ZHU_XIN_GEN', '007_SHEN_QIU_YU',
                    '016_LIU_FENG_MEI', '028_XUE_LUO_PING']
    exclude_list = None
    random_state = 42
    n_layer = 5
    #x_input = [12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 28, 29]
    x_input = range(7, 30)
    #x_input = [7, 8]
    n_input = len(x_input)
    n_output = 2
    lr = 0.001
    momentum = 0.97
    dropout_rate = 0.3
    batch_size = 256
    epoch = 1
    n_neurons = 100
    init = 'he_uniform'
    optimizer = Adam(learning_rate=lr)
    loss = 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    bootstrap = 100
    data_type = 'exvivo'
    saved_model = 'invivo_model.h5'
    freeze_layer = None
    wu_split = False


    x_train, y_train, x_test, y_test, df_test, x_test1, y_test1, \
    x_test2, y_test2, df_test1, df_test2 = get_data_exvivo(
        proj_dir=proj_dir,
        exvivo_data=exvivo_data,
        exclude_list=exclude_list,
        x_input=x_input
        )

    tuned_model = finetune_exvivo(
        x_train=x_train,
        y_train=y_train,
        proj_dir=proj_dir,
        saved_model=saved_model,
        batch_size=batch_size,
        epoch=epoch,
        freeze_layer=freeze_layer
        )

    test_exvivo(
        proj_dir=proj_dir,
        output_dir=output_dir,
        wu_split=wu_split,
        x_test=x_test,
        y_test=y_test,
        df_test=df_test,
        x_test1=x_test1,
        y_test1=y_test1,
        x_test2=x_test2,
        y_test2=y_test2,
        df_test1=df_test1,
        df_test2=df_test2
        )

    roc_stat = get_roc(
        proj_dir=proj_dir, 
        output_dir=output_dir, 
        bootstrap=bootstrap,
        data_type=data_type
        )

    prc_stat = get_prc(
        proj_dir=proj_dir,
        output_dir=output_dir,
        data_type=data_type
        )

