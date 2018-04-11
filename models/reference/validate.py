#
# Import
#
import sys, os
sys.path.insert(0, os.getcwd())

import numpy as np
import tensorflow as tf
from models.base import validate

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.python.platform import tf_logging as logger


#
# Settings
#
BaseValidator = validate.BaseValidator
streaming_accuracy = tf.contrib.metrics.streaming_accuracy


#
# Class
#
class Validator(BaseValidator):
    
    def __init__(self):
        BaseValidator.__init__(self)

    def define_metrics(self):
        tensors = ['predictions', 'raw_predictions', 'labels']
        pred, raw_pred, labels = self.get_tensors(tensors)

        accuracy, accuracy_update = streaming_accuracy(pred, labels)
        top5, top5_update = tf.metrics.mean(tf.nn.in_top_k(raw_pred,labels,5))
        self.set_tensors({'accuracy':accuracy,'top5_accuracy':top5})
        metrics_op = tf.group(accuracy_update, top5_update)

        # non-tensor metrics (Require dynamic_pad=True)
        self.confusion_matrix = np.empty([self.num_classes, self.num_classes])

        return metrics_op

    def report_validate(self, step, metrics_op):
        for_eval = ['accuracy', 'top5_accuracy', 
                    'logits', 'predictions', 'labels']

        tensors = [metrics_op] + list(self.get_tensors(for_eval))
        _, accuracy, top5_accuracy, logits, predictions, labels = \
                                            self.sess.run(tensors)

        curr_confusion_matrix = confusion_matrix(labels, predictions)
        self.confusion_matrix = self.confusion_matrix + curr_confusion_matrix

        logger.info('accuracy: %s', accuracy)
        logger.info('top5 accuracy: %s', top5_accuracy)
        
        logger.info('confusion mat: %s', self.confusion_matrix)
        logger.info('labels %s', labels)
        #logger.info(classification_report(labels, predictions))
        return


if __name__ == '__main__':
    v = Validator()
    v.validate()

