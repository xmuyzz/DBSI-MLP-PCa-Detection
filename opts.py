import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')    
    # path
    parser.add_argument('--proj_dir', default='/home/xmuyzz/Harvard_AIM/others/pca', type=str, help='Root path')
    parser.add_argument('--output_dir', default='/mnt/aertslab/USERS/Zezhong/others/pca/output', type=str, help='output path')
    parser.add_argument('--pro_data', default='pro_data', type=str, help='Processed data path')
    parser.add_argument('--model', default='output/model', type=str, help='Results output path')
    parser.add_argument('--log', default='output/log', type=str, help='Log data path')
    
    # load data
    parser.add_argument('--benign_nobix', default='benign_no_biopsy.csv', type=str, help='data csv')
    parser.add_argument('--benign_bix', default='benign_biopsy.csv', type=str, help='data csv')
    parser.add_argument('--pca_bix', default='pca_biopsy.csv', type=str, help='data csv')
    parser.add_argument('--exclude_patient', default=['001_ZHOU_CHAO_GANG', '002_ZHU_XIN_GEN', '007_SHEN_QIU_YU',
                        '016_LIU_FENG_MEI', '028_XUE_LUO_PING'], type=list, help='exclude list')
    parser.add_argument('--exvivo_data', default='pca_exvivo.csv', type=str, help='data csv')
    parser.add_argument('--invivo_tissue_type', default='benign', type=str, help='(benign|BPZ|BTZ)')
    parser.add_argument('--exvivo_tissue_type', default='BPH', type=str, help='(benign|BPZ|BPH|SBPH)')

    # train model
    parser.add_argument('--data_type', default='exvivo', type=str, help='(invivo|exvivo')
    parser.add_argument('--x_input', default=range(12, 30), type=list, help='input image metrics')
    parser.add_argument('--output_size', default=2, type=int, help='output size')
    parser.add_argument('--n_layer', default=5, type=int, help='MLP layer')
    parser.add_argument('--n_neurons', default=100, type=int, help='number of neurons per layer')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--train_epoch', default=20, type=int, help='train epoch')
    parser.add_argument('--activation', default='elu', type=str, help='(relu|elu|leaky_relu)')
    parser.add_argument('--output_activation', default='softmax', type=str, help='Output activation function')
    parser.add_argument('--loss_function', default='sparse_categorical_crossentropy', type=str, help='loss function')
    parser.add_argument('--optimizer_function', default='adam', type=str, help='optmizer function')
    parser.add_argument('--dropout_rate', default=0.3, type=int, help='drop out rate')
    parser.add_argument('--init', default='he_uniform', type=str, help='kernal initialization')
    parser.add_argument('--momentum', default=0.97, type=float, help='batch momentum')

    # evalute model                        
    parser.add_argument('--bootstrap', default=1000, type=int, help='bootstrap to calcualte 95% CI of AUC')

    # finetune model                        
    parser.add_argument('--freeze_layer', default=None, type=int, help='freeze layer to fine tune model')
    parser.add_argument('--saved_model', default='invivo_model.h5', type=str, help='saved model name')
    parser.add_argument('--finetune_epoch', default=20, type=int, help='fine tune epoch')

    # others 
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=True)
    parser.add_argument('val', action='store_true', help='If true, validation is performed.')
    parser.set_defaults(val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument('--finetune', action='store_true', help='If true, finetune is performed.')
    parser.set_defaults(finetune=False)
    parser.add_argument('--get_stat', action='store_true', help='If true, get_stat is performed.')
    parser.set_defaults(get_stat=True)
    parser.add_argument('--wu_spit', action='store_true', help='If true, wu_slpit is performed.')
    parser.set_defaults(wu_split=False)

    args = parser.parse_args()

    return args



