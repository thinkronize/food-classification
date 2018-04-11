#
# Import
#
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import sys, os, time, argparse

from common import hasher
from common import dataset
from nets import inception_resnet_v2

from tensorflow.python.platform import tf_logging as logger
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step


#
# Settings
#
FS = tf.app.flags.FLAGS
slim = tf.contrib.slim
ref_model = inception_resnet_v2.inception_resnet_v2
arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope
streaming_accuracy = tf.contrib.metrics.streaming_accuracy
from_checkpoint = tf.contrib.framework.assign_from_checkpoint_fn


#
# Class
#
class BaseTrainer: 

    def __init__(self):
        self.FS = FS
        self._print_argv()
        self._print_hash()
        self._init_args()

    #
    # Core API(s)
    #
    def train(self):
        with tf.Graph().as_default() as graph:
            tf.logging.set_verbosity(tf.logging.INFO)
            self.make_tensor_bag()
            self.train_in_graph(graph)

    def train_in_graph(self, graph):
        self.get_batch()
        self.output_layers()
        self.define_var_to_restore()

        self.define_loss()
        self.define_optimizer()

        train_op = self.define_train_op()
        metrics_op = self.define_metrics()
        summary_op = self.define_summary()

        sv = self.make_supervisor(graph)
        with sv.managed_session() as sess:
            self.sess = sess
            for step in self.step_iterator():
                self.report_epoch(step)
                self.train_step(sv, train_op, metrics_op)
                self.run_summary(step, sv, summary_op) 
            self.finalize_train(sv)
        return

    #
    # Pipeline
    #
    def get_batch(self):
        self.dataset = dataset.ImageDataset(FS.dataset_dir, 
                                            FS.record_name, 
                                            num_classes=FS.num_classes)
        self.dataset.load()
        self.num_classes = self.dataset.get_num_classes()
        print('>>> dataset load complete ...') 
        print('>>> num classes: %d\n' % self.num_classes) 

        images, raw_images, labels = self.dataset.get_batch()
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

    def define_loss(self):
        labels, logits = self.get_tensors(['labels', 'logits'])
        one_hot_labels = slim.one_hot_encoding(labels, self.num_classes)

        loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        total_loss = tf.losses.get_total_loss()    
        
        return self.set_tensor('total_loss', total_loss)

    def define_optimizer(self):
        decay_steps = FS.num_epochs_before_decay * self.num_steps_per_epoch()

        lr = tf.train.exponential_decay(
                learning_rate=FS.initial_learning_rate,
                global_step=get_or_create_global_step(),
                decay_steps=int(decay_steps),
                decay_rate=FS.learning_rate_decay_factor,
                staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=0.1)

        return self.set_tensors({'optimizer':optimizer, 'lr':lr})

    def define_train_op(self):
        loss, optimizer = self.get_tensors(['total_loss', 'optimizer'])
        return slim.learning.create_train_op(loss, optimizer)

    def define_metrics(self):
        tensors = ['predictions', 'raw_predictions', 'labels']
        pred, raw_pred, labels = self.get_tensors(tensors)

        accuracy, accuracy_update = streaming_accuracy(pred, labels)
        top5, top5_update = tf.metrics.mean(tf.nn.in_top_k(raw_pred,labels,5))
        self.set_tensors({'accuracy':accuracy,'top5_accuracy':top5})
        metrics_op = tf.group(accuracy_update, top5_update)

        return metrics_op

    def define_summary(self):
        tensors = ['total_loss', 'accuracy', 'top5_accuracy', 'lr']
        loss, accuracy, top5, lr = self.get_tensors(tensors)

        tf.summary.scalar('losses/Total_Loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('top5_accuracy', top5)
        tf.summary.scalar('learning_rate', lr)
        summary_op = tf.summary.merge_all()

        return summary_op

    def define_var_to_restore(self):
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        vars_to_restore = slim.get_variables_to_restore(exclude=exclude)
        return self.set_tensor('variables_to_restore', vars_to_restore)

    def make_supervisor(self, graph):
        vars_to_restore = self.get_tensor('variables_to_restore')

        saver = tf.train.Saver(vars_to_restore)
        init_fn = from_checkpoint(FS.checkpoint_file, vars_to_restore, 
                                  ignore_missing_vars=True)

        sv = tf.train.Supervisor(graph, summary_op=None, 
                                 init_fn=init_fn, logdir=FS.model_log_dir)
        return sv

    def report_epoch(self, step):
        if step % self.num_steps_per_epoch() != 0:
            return

        for_eval = ['lr', 'accuracy', 'top5_accuracy', 
                    'logits', 'predictions', 'labels']

        tensors = list(self.get_tensors(for_eval))
        lr, accuracy, top5_accuracy, logits, predictions, labels = self.sess.run(tensors)

        logger.info('epoch %s/%s', self.current_epoch(step),FS.num_epochs)
        logger.info('learning rate: %s', lr)
        logger.info('accuracy: %s', accuracy)
        logger.info('top5 accuracy: %s', top5_accuracy)
        
        #print('logits: \n', logits)
        print('predictions: \n', predictions)
        print('Labels:\n:', labels)
        return

    def train_step(self, sv, train_op, metrics_op):
        start_time = time.time()

        for_eval = [train_op, sv.global_step, metrics_op]
        total_loss, global_step, _ = self.sess.run(for_eval)

        time_elapsed = time.time() - start_time

        logger.info('global step %s: loss: %.4f (%.2f sec/step)', 
                     global_step, total_loss, time_elapsed)
        return total_loss, global_step

    def run_summary(self, step, sv, summary_op):
        if step % 10 == 0:
            summaries = self.sess.run(summary_op)
            sv.summary_computed(self.sess, summaries)
        return

    def finalize_train(self, sv):
        for_eval = ['total_loss', 'accuracy', 'top5_accuracy']

        tensors = list(self.get_tensors(for_eval))
        loss, accuracy, top5_accuracy = self.sess.run(tensors)

        logger.info('Final Loss: %s', loss)
        logger.info('Final Accuracy: %s', accuracy)
        logger.info('Final Top5 Accuracy: %s', top5_accuracy)
        logger.info('Finished training! Saving model to disk now')

        sv.saver.save(self.sess, sv.save_path, global_step=sv.global_step)
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
        ckpt = '/hdd/aidentify-models/pretrained-models/inception_resnet_v2_2016_08_30.ckpt'

        # use json config file
        parser.add_argument('--model_log_dir', type=str, required=True)
        parser.add_argument('--dataset_dir', type=str, required=True)
        parser.add_argument('--record_name', type=str, required=True)
        parser.add_argument('--checkpoint_dir', type=str, default=ckpt)

        parser.add_argument('--num_classes', type=int)
        parser.add_argument('--label_path', type=str)
        parser.add_argument('--image_size', type=int, default=299)

        parser.add_argument('--num_thread', type=int, default=16)
        parser.add_argument('--init_lr', type=float, default=0.01)
        parser.add_argument('--num_epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=16)

        self._parse_args_more()
        return parser.parse_args()

    # Template Method
    def _parse_args_more(self):
        pass
 

    def _init_args(self):
        args = self._parse_args()

        FS.record_name = args.record_name
        FS.dataset_dir = args.dataset_dir
        FS.model_log_dir = args.model_log_dir
        FS.checkpoint_file = args.checkpoint_dir

        if not os.path.exists(FS.model_log_dir):
            os.mkdir(FS.model_log_dir)

        FS.image_size = args.image_size
        FS.num_classes = args.num_classes

        FS.num_epochs = args.num_epochs
        FS.batch_size = args.batch_size
        FS.num_thread = args.num_thread

        # TODO) Optimizer Parameterize
        FS.initial_learning_rate = args.init_lr
        FS.center_loss_factor = 0.002
        FS.center_loss_alfa = 0.5
        FS.num_epochs_before_decay = 2
        FS.learning_rate_decay_factor = 0.9

        self._init_args_more(args)
        return 

    # Template Method
    def _init_args_more(self, args):
        pass

