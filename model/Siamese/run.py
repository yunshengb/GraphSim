#!/usr/bin/python3
from __future__ import division
from __future__ import print_function

from config import FLAGS
from train import train_val_loop, test
from utils_siamese import get_model_info_as_str, \
    check_flags, convert_long_time_to_str
from data_siamese import SiameseModelData
from dist_sim_calculator import DistSimCalculator
from models_factory import create_model
from saver import Saver
from eval import Eval
import tensorflow as tf
from time import time
import os, traceback


def main():
    t = time()
    check_flags()
    print(get_model_info_as_str())
    data_train = SiameseModelData(FLAGS.dataset_train)
    dist_sim_calculator = DistSimCalculator(
        FLAGS.dataset_train, FLAGS.ds_metric, FLAGS.ds_algo)
    model = create_model(FLAGS.model, data_train.input_dim(),
                         data_train, dist_sim_calculator)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = Saver(sess)
    sess.run(tf.global_variables_initializer())
    if FLAGS.dataset_val_test == FLAGS.dataset_train:
        data_val_test = data_train
    else:
        # Generalizability test: val test on unseen train and test graphs.
        data_val_test = SiameseModelData(FLAGS.dataset_val_test)
    eval = Eval(data_val_test, dist_sim_calculator)
    try:
        train_costs, train_times, val_results_dict = \
            train_val_loop(data_train, data_val_test, eval, model, saver, sess)
        best_iter, test_results = \
            test(data_val_test, eval, model, saver, sess, val_results_dict)
        overall_time = convert_long_time_to_str(time() - t)
        print(overall_time)
        saver.save_overall_time(overall_time)
    except:
        traceback.print_exc()
    else:
        return train_costs, train_times, val_results_dict, best_iter, test_results


if __name__ == '__main__':
    main()
