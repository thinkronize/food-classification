#
# Import
#
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import sys, os, time, argparse
sys.path.insert(0, os.getcwd())

from common import hasher
from common import dataset
from nets import inception_resnet_v2 

from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score 

from tensorflow.python.platform import tf_logging as logger
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step


#
# Settings
#
FS = tf.app.flags.FLAGS
slim = tf.contrib.slim
last_checkpoint = tf.train.latest_checkpoint
ref_model = inception_resnet_v2.inception_resnet_v2
arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope
streaming_accuracy = tf.contrib.metrics.streaming_accuracy


#
# Class
#
class BaseValidator:

    def __init__(self):
        self.FS = FS
        self._print_argv()
        self._print_hash()
        self._init_args()

    #
    # Core API(s)
    #
    def validate(self):
        with tf.Graph().as_default() as graph:
            tf.logging.set_verbosity(tf.logging.INFO)
            self.make_tensor_bag()
            self.validate_in_graph(graph)

    def validate_in_graph(self, graph):
        self.get_batch()
        self.output_layers()
        self.define_var_to_restore()

        metrics_op = self.define_metrics()
        summary_op = self.define_summary()
        global_step_op = self.define_global_step()

        sv = self.make_supervisor(graph)
        with sv.managed_session() as sess:
            self.sess = sess
            for step in self.step_iterator():
                self.inc_global_step(global_step_op)
                self.report_validate(step, metrics_op)
                self.run_summary(step, sv, summary_op) 
            self.finalize_validate(sv)
        return
 
    #
    # Pipeline
    #
    def get_batch(self):
        self.dataset = dataset.ImageDataset(FS.dataset_dir, 
                                            FS.record_name, 
                                            'validation',
                                            num_classes=FS.num_classes)
        self.dataset.load()
        self.num_classes = self.dataset.get_num_classes()
        print('>>> dataset load complete ...') 
        print('>>> num classes: %d\n' % self.num_classes) 

        images, raw_images, labels = self.dataset.get_batch(dynamic_pad=True)
        self.set_tensors({'images':images, 'raw_images':raw_images, 
                          'labels':labels})
        return 

    def output_layers(self):
        images = self.get_tensor('images')
 
        with slim.arg_scope(arg_scope()):
            logits, end_points = ref_model(images, 
                                           self.num_classes, 
                                           self.is_train())

        raw_predictions = end_points['Predictions']
        predictions = tf.argmax(raw_predictions, 1)

        return self.set_tensors({'logits':logits, 
                                'end_points':end_points, 
                                'predictions':predictions, 
                                'raw_predictions':raw_predictions})
    
    def define_var_to_restore(self):
        vars_to_restore = slim.get_variables_to_restore()
        return self.set_tensor('variables_to_restore', vars_to_restore)

    def make_supervisor(self, graph):
        vars_to_restore = self.get_tensor('variables_to_restore')
        saver = tf.train.Saver(vars_to_restore)

        def init_fn(sess):
            return saver.restore(sess, last_checkpoint(FS.model_log_dir))

        sv = tf.train.Supervisor(graph, summary_op=None, 
                                 init_fn=init_fn, logdir=FS.eval_log_dir)
        return sv

    def define_metrics(self):
        tensors = ['predictions', 'raw_predictions', 'labels']
        pred, raw_pred, labels = self.get_tensors(tensors)

        accuracy, accuracy_update = streaming_accuracy(pred, labels)
        top5, top5_update = tf.metrics.mean(tf.nn.in_top_k(raw_pred,labels,5))
        self.set_tensors({'accuracy':accuracy,'top5_accuracy':top5})
        metrics_op = tf.group(accuracy_update, top5_update)

        return metrics_op

    def define_summary(self):
        tensors = ['accuracy', 'top5_accuracy']
        accuracy, top5 = self.get_tensors(tensors)

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('top5_accuracy', top5)
        summary_op = tf.summary.merge_all()

        return summary_op

    def define_global_step(self):
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)

        return global_step_op

    def inc_global_step(self, global_step_op):
        self.sess.run(global_step_op)

    def report_validate(self, step, metrics_op):
        for_eval = ['accuracy', 'top5_accuracy', 
                    'logits', 'predictions', 'labels']

        tensors = [metrics_op] + list(self.get_tensors(for_eval))
        _, accuracy, top5_accuracy, logits, \
                predictions, labels = self.sess.run(tensors)

        logger.info('accuracy: %s', accuracy)
        logger.info('top5 accuracy: %s', top5_accuracy)
        return

    def run_summary(self, step, sv, summary_op):
        if step % 10 == 0:
            summaries = self.sess.run(summary_op)
            sv.summary_computed(self.sess, summaries)
        return

    def finalize_validate(self, sv):
        for_eval = ['accuracy', 'top5_accuracy']

        tensors = list(self.get_tensors(for_eval))
        accuracy, top5_accuracy = self.sess.run(tensors)

        logger.info('Final Accuracy: %s', accuracy)
        logger.info('Final Top5 Accuracy: %s', top5_accuracy)
        logger.info('Model validation has completed! Visit TensorBoard')
        return


    #
    # Accessor(s)
    #
    def make_tensor_bag(self):
        self.tensor_bag = {}  
        return self.tensor_bag
    
    def set_tensor(self, name, tensor):
        self.tensor_bag[name] = tensor

    def set_tensors(self, tensors_dic):
        self.tensor_bag.update(tensors_dic)

    def get_tensor(self, name):
        return self.tensor_bag.get(name)

    def get_tensors(self, names):
        return (self.tensor_bag[k] for k in names)
    
    def num_steps_per_epoch(self):
        return self.dataset.num_steps_per_epoch()
    
    def current_epoch(self, step):
        return int(step/self.num_steps_per_epoch()+1)

    def step_iterator(self):
        return xrange(self.num_steps_per_epoch() * FS.num_epochs)

    def is_train(self):
        return self.dataset.is_train()

    #
    # Init
    #
    def _print_argv(self):
        print('\n<<< execution summary >>>')
        print(">>> cmd:  %s" % sys.argv)
        return

    def _print_hash(self):
        print('>>> hash: [%s]' % hasher.get_hash(__file__))
        return

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_log_dir', type=str, required=True)
        parser.add_argument('--eval_log_dir', type=str, required=True)
        parser.add_argument('--dataset_dir', type=str, required=True)
        parser.add_argument('--record_name', type=str, required=True)
        parser.add_argument('--num_classes', type=int, required=True)

        parser.add_argument('--num_epochs', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--num_thread', type=int, default=16)
        parser.add_argument('--image_size', type=int, default=299)
        
        return parser.parse_args()

    def _init_args(self):
        args = self._parse_args()

        FS.record_name = args.record_name
        FS.dataset_dir = args.dataset_dir
        FS.model_log_dir = args.model_log_dir
        FS.eval_log_dir = args.eval_log_dir

        if not os.path.exists(FS.eval_log_dir):
            os.mkdir(FS.eval_log_dir)

        FS.image_size = args.image_size
        FS.num_classes = args.num_classes

        FS.num_epochs = args.num_epochs
        FS.batch_size = args.batch_size
        FS.num_thread = args.num_thread 

