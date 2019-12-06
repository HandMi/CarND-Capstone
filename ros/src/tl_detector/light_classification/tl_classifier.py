from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import numpy as np

SCORE_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):
        self.sess = None
        self.graph = None
        PATH_TO_SIM_GRAPH = r'models/ssd_inception_v2_coco_sim/frozen_inference_graph.pb'
        self.load_graph(PATH_TO_SIM_GRAPH)
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

    def load_graph(self, graph_file):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (300,300))
        image_expanded = np.expand_dims(image,axis=0)
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes,self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_expanded})
        if scores[0][0]>SCORE_THRESHOLD:
            top_classification = classes[0][0]
            if top_classification == 1:
               return TrafficLight.GREEN
            elif top_classification == 2:
               return TrafficLight.RED
            elif top_classification == 3:
               return TrafficLight.YELLOW
            else:
               return TrafficLight.UNKNOWN
        return TrafficLight.UNKNOWN
