"""
*line_config.py
*this file is a configuration file.
*created by longhaixu
*copyright USTC
*16.11.2020
"""


class Training_Flag(object):
    batch_size = 64
    sync_bn = False
    resume = False
    epochs = 60
    num_gpus = 1
    train_num_workers = 8
    val_num_workers = 4
    save_pth_root = r'/home/hxlong/Project/NEW_HWR/line_recognition/save'
    save_pth_name = r'save_3xensemble.pth'

    train_root = r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/line'
    # train_folder_list = [r'HWDB2.0Test', r'HWDB2.0Train',
    #                      r'HWDB2.1Test', r'HWDB2.1Train',
    #                      r'HWDB2.2Test', r'HWDB2.2Train',
    #                      r'Generate_01_random', r'Generate_02_random',
    #                      r'Generate_01_corpus']
    train_folder_list = [r'HWDB2.0Train_pure', r'HWDB2.1Train_pure', r'HWDB2.2Train_pure']

    # val_root = r'/home/hxlong/DataSet/HWDB/CASIA_HWDB/Data/test/line'
    # val_folder_list = [r'Competition13Line', r'test']
    val_root = train_root
    val_folder_list = [r'HWDB2.0Test_pure', r'HWDB2.1Test_pure', r'HWDB2.2Test_pure']


class Validation_Flag(object):
    batch_size = int(Training_Flag.batch_size / 2)


class Net_Flag(object):
    seq_length = 55
    num_class = 7358 + 1
    output_node = seq_length * num_class
    img_ratio = 20
    img_H = 60
    img_W = img_H * img_ratio
    end_flag = 9999
