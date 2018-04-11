#
# Imports
#
import os
import argparse
import tensorflow as tf
import inception_preprocessing

from datetime import datetime
from tensorflow.contrib import slim
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

image_preprocessor = inception_preprocessing.preprocess_for_eval
gen_init_fn = slim.assign_from_checkpoint_fn
FS = tf.app.flags.FLAGS


#
# Model Supervisor
#
class ModelSupervisor:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    
    #
    # Export 
    #
    def export_model(self, dir_to_export):
        self.__set_checkpoint_path()
        self.__set_export_filepath(dir_to_export)

        with tf.Graph().as_default():
            processed_images = self.__input_image_acceptor()
            probabilities = self.__load_inception_network(processed_images, FS.num_classes)
            init_fn = gen_init_fn(self.checkpoint_path, slim.get_model_variables())

            with tf.Session() as sess:
                init_fn(sess)
                graph = self.__define_export_graph(sess)
                self.__write_export_model(graph, self.export_model_path)
                
    def __set_checkpoint_path(self):
        self.checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        print("Path of source checkpoint : %s " % self.checkpoint_path)
    
    def __set_export_filepath(self, dir_to_export):
        export_name = "model_%s.pb" %(datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.export_model_path = dir_to_export + "/" + export_name
        print("Path of export model : %s " % self.export_model_path)
        
    def __input_image_acceptor(self):
        input_image = tf.placeholder(tf.string, name='input_image')
        image = tf.image.decode_jpeg(input_image, channels=3)
            
        processed_images = image_preprocessor(image, FS.image_size, FS.image_size, central_fraction=None)
        processed_images = tf.expand_dims(processed_images, 0)
        return processed_images
    
    def __load_inception_network(self, processed_images, num_classes):
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            #logits, _ = inception_resnet_v2(processed_images, num_classes=num_classes,
            #                               is_training=False)
            prelogits, end_points = inception_resnet_v2(processed_images, num_classes=num_classes, is_training=False)

        embeddings = tf.nn.l2_normalize(prelogits, dim=1, epsilon=1e-10, name='embeddings')

        logits = slim.fully_connected(prelogits, num_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.001),
                                      scope='Logits', reuse=False)
        return logits
    
    def __define_export_graph(self, sess):
        target_tensor_names = ["input_image", "DecodeJpeg", "Logits/BiasAdd", "embeddings"]
        constant_graph = convert_variables_to_constants(sess, sess.graph_def, target_tensor_names)
        return constant_graph
    
    def __write_export_model(self, graph, export_filepath):
        with tf.gfile.GFile(export_filepath, "wb") as f:
            f.write(graph.SerializeToString())
        print("%d ops in the final graph." % len(graph.node))
        print("*** compelete export ***")
    
    #
    # Import 
    #
    @staticmethod
    def import_model(model_filepath):
        with tf.gfile.GFile(model_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                name = "",
                input_map = None,
                return_elements = None,
                op_dict = None,
                producer_op_list = None
            )
            
        return graph

#
# Entry Point
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',  type=str, required=True, help='model dir')
    parser.add_argument('--export_dir', type=str, required=True, help='export dir')
    parser.add_argument('--class_size', type=int, required=True, default=101, help='class size')
    parser.add_argument('--image_size', type=int, default=299, help='image size')
    args = parser.parse_args()

    # TODO : class count from label 
    # TODO : model export name

    FS.image_size = args.image_size
    FS.num_classes = args.class_size
    
    ms = ModelSupervisor(args.model_dir)
    ms.export_model(args.export_dir)

