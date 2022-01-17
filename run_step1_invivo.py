from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import ELU, LeakyReLU
from get_data_invivo_kFoldVal import get_data_invivo_kFoldVal
from get_model import get_model
from train_invivo import train_invivo
from get_roc import get_roc
from get_prc import get_prc



if __name__ == '__main__':

    proj_dir = r'C:\Users\atwu\Desktop\PCa_voxel_data'
    output_dir  = r'C:\Users\atwu\Desktop\PCa_results'

    benign_nobix = 'benign_no_biopsy.csv'
    benign_bix = 'benign_biopsy.csv'
    pca_bix = 'pca_biopsy.csv'
    exclude_patient = True
    exclude_list = ['001_ZHOU_CHAO_GANG', '002_ZHU_XIN_GEN', '007_SHEN_QIU_YU',
                    '016_LIU_FENG_MEI', '028_XUE_LUO_PING']
    alpha = 0.3
    random_state = 42
    ELU_alpha = 1.0
    digit = 3
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
    activation = ELU(alpha=ELU_alpha)
    bootstrap = 100
    data_type = 'invivo'
    numFolds = 5


    x_train, y_train, x_val, y_val, x_test, y_test, \
    df_val, df_test, trainIDs, valIDs, testIDs, df_excludeX, df_excludeY = get_data_invivo_kFoldVal(
        proj_dir=proj_dir,
        benign_bix=benign_bix,
        benign_nobix=benign_nobix,
        pca_bix=pca_bix,
        exclude_patient=exclude_patient,
        exclude_list=exclude_list,
        x_input=x_input,
        folds=numFolds
        )

    model = get_model(
        init=init,
        dropout_rate=dropout_rate,
        momentum=momentum,
        n_input=n_input,
        n_layer=n_layer
        )

    train_invivo(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        trainIDs=trainIDs,
        valIDs=valIDs,
        testIDs=testIDs,
        df_val=df_val,
        df_test=df_test,
        proj_dir=proj_dir,
        batch_size=batch_size,
        epoch=epoch,
        loss=loss,
        optimizer=optimizer,
        folds=numFolds,
        exclude_patient=exclude_patient,
        exclude_list=exclude_list,
        df_excludeX=df_excludeX, 
        df_excludeY=df_excludeY
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
        data_type = data_type
        )

