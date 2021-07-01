"""
*utils.py
*this file provide some utils functions.
*created by longhaixu
*copyright USTC
*16.11.2020
"""

import numpy as np
import difflib
# import tensorflow as tf


def _ctc_remove_blank(char_list, blank):
    """
    :param char_list: list of index
    :param blank: index of black
    :return:
    """
    tmp = []
    previous = None
    for char in char_list:
        if char != previous:
            tmp.append(char)
            previous = char
        tmp = [c for c in tmp if c != blank]
    return tmp


def ctc_decode(output, blank, name='G'):
    """
    output: shape of [-1, Net_Flag.seq_length, Net_Flag.num_class]
    name:G  method:Greedy search
    name:B  method:Beam search
    name:PB method:Prefix Beam search
    :return:char_out_lists after remove blank
    """
    assert name == 'G' or name == 'B'
    if name == 'G':
        char_out_lists = output.max(2)[1].cpu()
        score = output.softmax(2).max(2)[0].cpu().detach().numpy()
        scores_list = score
        char_out_lists = np.asarray(char_out_lists)
        char_out_lists = [_ctc_remove_blank(x, blank=blank) for x in char_out_lists]
        return char_out_lists, scores_list
    # elif name == 'B':
    #     tf_in = tf.transpose(tf.Variable(output.cpu().detach().numpy()), [1, 0, 2])
    #     tf_out = tf.nn.ctc_beam_search_decoder(tf_in, tf.Variable([tf_in.shape[0]] * tf_in.shape[1]), beam_width=5)
    #     char_out_lists = []
    #     for i in range(tf_in.shape[1]):
    #         char_out_lists.append([])
    #     point = 0
    #     for i, index in tf_out[0][0].indices:
    #         char_out_lists[i].append(int(tf_out[0][0].values[point]))
    #         point += 1
    #     return char_out_lists, None


def _cal_editing_distance(str_predict, str_label):
    """
    Calculating editing distance between two string.
    :param str_predict: the predict string
    :param str_label: the label string
    :return: replace, insert, delete
    """
    s = difflib.SequenceMatcher(None, str_predict, str_label).get_opcodes()
    replace = 0
    insert = 0
    delete = 0
    for tag, i1, i2, j1, j2 in s:
        if tag == 'replace':
            replace += min(i2 - i1, j2 - j1)
            if i2 - i1 >= j2 - j1:
                delete += ((i2 - i1) - (j2 - j1))
            else:
                insert += ((j2 - j1) - (i2 - i1))
        elif tag == 'insert':
            insert += max(i2 - i1, j2 - j1)
        elif tag == 'delete':
            delete += max(i2 - i1, j2 - j1)
    return replace, insert, delete


def cal_AR_CR_for_two_str(str_predict, str_label):
    """
    Calculating AR and CR between two string.
    :param str_predict: the predict string
    :param str_label: the label string
    :return: AR, CR
    """
    replace, insert, delete = _cal_editing_distance(str_predict, str_label)
    AR = 1 - ((replace + insert + delete) / float(max(len(str_label), len(str_predict))))
    CR = 1 - ((replace + delete) / float(max(len(str_label), len(str_predict))))
    return AR, CR
