import os
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from get_data import get_data



def main(proj_dir, x_input, saved_model, lock_base_model, train_type):

    """
    finetune a CNN model
    @params:
      saved_model   - required : saved CNN model for finetuning
      run_model     - required : CNN model name to be saved
      model_dir     - required : folder path to save model
      input_channel - required : model input image channel, usually 3
      freeze_layer  - required : number of layers to freeze in finetuning     
    """
    
    pro_data_dir = proj_dir + '/pro_data'
    output_dir = proj_dir + '/output'
    data_dir = proj_dir + '/data'

    # get data
    x_train, x_val, x_test, y_train, y_val, y_test = get_data(data_dir, x_input)

    ## fine tune model
    base_model = load_model(pro_data_dir + '/' + saved_model)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    if lock_base_model:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in model.layers[0:16]:
            layer.trainable = False
        for layer in model.layers:
            print(layer, layer.trainable)
    model.summary()
    
    ## fit data into dnn models
    if train_type == 'cm':
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size,
            epochs=epoch, validation_data=(x_val, y_val), verbose=1,
            callbacks=None, validation_split=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0)
        print('fine tuning model complete!!')
        y_pred = model.predict(x_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        score = model.evaluate(x_test, y_test, verbose=0)
        loss = np.around(score[0], 3)
        acc = np.around(score[1], 3)
        print('acc:', acc)
        print('loss:', loss)
        model.save(pro_data_dir + '/grading_model.h5'))
        # save a df for test and prediction
        df_test['y_pred'] = y_pred[:, 1]
        df_test['y_pred_class'] = y_pred_class
        df_test.rename(columns={'ROI_Class': 'y_test'}, inplace=True)
        test_pred = df_test[['Sub_ID', 'y_test', 'y_pred', 'y_pred_class']]
        test_pred.to_csv(pro_data_dir + '/grading_voxel_pred.csv')
        print('successfully save test voxel prediction!')
        cm = confusion_matrix(label, pred)
        cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.around(cm_norm, 2)
        report = classification_report(label, pred, digits=3)
        print(cm)
        print(cm_norm)
        print(report)
        # plot confusion matrix
        for cm_type, fmt in zip(['norm', 'raw'], ['', 'd']):
            ax = sn.heatmap(cm0, annot=True, cbar=True, cbar_kws={'ticks': [-0.1]},
                annot_kws={'size': 26, 'fontweight': 'bold'}, cmap='Blues',
                fmt=fmt, linewidths=0.5)
            ax.axhline(y=0, color='k', linewidth=4)
            ax.axhline(y=5, color='k', linewidth=4)
            ax.axvline(x=0, color='k', linewidth=4)
            ax.axvline(x=5, color='k', linewidth=4)
            ax.tick_params(direction='out', length=4, width=2, colors='k')
            ax.xaxis.set_ticks_position('top')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.tight_layout()
            fn = 'cm_' + cm_type + '.png'
            plt.savefig(output_dir + '/' + fn, format='png', dpi=600)
            plt.close()
    
    elif train_type == 'roc':
        history = OneVsRestClassifier(
            model.fit(
                x=x_train,
                y=y_train,
                batch_size=256,
                epochs=20,
                verbose=0,
                callbacks=None,
                validation_split=None,
                validation_data=(x_val, y_val),
                shuffle=True))
        score = model.evaluate(x_test, y_test, verbose=0)
        y_pred = model.predict(x_test)  
        y_pred_label = np.argmax(y_pred, axis=1)
        test_loss = round(score[0], 3)
        test_accuracy = round(score[1], 3)
        # plot roc curve
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        threshold = dict()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')        
        #colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])
        #for i, color in zip(range(n_classes), colors):
        for i in range(n_classes):
            fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            print('ROC AUC %.2f' % roc_auc[i])
            plt.plot(fpr[i], tpr[i], color='blue', linewidth=3, label='AUC %0.2f' % roc_auc[i])
            plt.xlim([-0.03, 1])
            plt.ylim([0, 1.03])
            ax.axhline(y=0, color='k', linewidth=4)
            ax.axhline(y=1.03, color='k', linewidth=4)
            ax.axvline(x=-0.03, color='k', linewidth=4)
            ax.axvline(x=1, color='k', linewidth=4) 
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
            #plt.xlabel('False Positive Rate', fontweight='bold', fontsize=15)
            #plt.ylabel('True Positive Rate', fontweight='bold', fontsize=15)
            plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'}) 
            plt.grid(True)
            fn = 'ROC' + '_' + str(i) + '.png'
            plt.savefig(output_dir + '/' + fn, format='png', dpi=600)
            plt.close()
        # plot precision-recall 
        precision = dict()
        recall = dict()
        threshold = dict()
        prc_auc = []
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        #colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])
        #for i, color in zip(range(n_classes), colors):
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
            RP_2D = np.array([recall[i], precision[i]])
            RP_2D = RP_2D[np.argsort(RP_2D[:,0])]
            prc_auc.append(auc(RP_2D[1], RP_2D[0]))
            print('PRC AUC %.2f' % auc(RP_2D[1], RP_2D[0]))
            plt.plot(recall[i], precision[i], color='red', linewidth=3, label='AUC %0.2f' % prc_auc[i])
        plt.xlim([0, 1.03])
        plt.ylim([0, 1.03])
        ax.axhline(y=0, color='k', linewidth=4)
        ax.axhline(y=1.03, color='k', linewidth=4)
        ax.axvline(x=0, color='k', linewidth=4)
        ax.axvline(x=1.03, color='k', linewidth=4) 
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
        #plt.xlabel('recall', fontweight='bold', fontsize=16)
        #plt.ylabel('precision', fontweight='bold', fontsize=16)
        plt.legend(loc='lower left', prop={'size': 14, 'weight': 'bold'}) 
        plt.grid(True)
        PRC_filename = 'PRC' + str(i) + '.png'
        plt.savefig(output_dir + '/' + fn, format='png', dpi=600)
        plt.close()


if __name__ == '__main__':
    
    proj_dir = '/home/xmuyzz/Harvard_AIM/others/pca/grading'
    saved_model = 'saved_model.h5'
    x_input = range(7, 30)
    main(proj_dir=proj_dir, x_input=x_input, saved_model=saved_model, lock_base_model=True, train_type='cm')





