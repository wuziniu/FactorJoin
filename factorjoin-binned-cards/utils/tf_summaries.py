import os
import tensorflow as tf

class TensorboardSummaries(object):
    def __init__(self, tb_path):
        # config = tf.ConfigProto(device_count = {'GPU': 0})
        # self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        self.summary_ops = {}
        self.summary_vars = {}
        self.writer = tf.summary.FileWriter(tb_path)

    def add_variables(self, var_list, name='default'):
        summary_vars = []
        ops_list = []
        for var in var_list:
            tf_var = tf.Variable(0.)
            ops_list.append(
                tf.summary.scalar(var, tf_var))
            summary_vars.append(tf_var)
        summary_ops = tf.summary.merge(ops_list)
        self.summary_ops[name] = summary_ops
        self.summary_vars[name] = summary_vars

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def report(self, epoch, values, name='default'):
        feed_dict = {
            self.summary_vars[name][i] : v for (i, v) in enumerate(values)
        }
        summary_str = self.sess.run(
            self.summary_ops[name], feed_dict=feed_dict)
        self.writer.add_summary(summary_str, epoch)
        self.writer.flush()
