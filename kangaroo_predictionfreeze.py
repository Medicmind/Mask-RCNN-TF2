import tensorflow as tf
import numpy as np
from PIL import Image
import mrcnn.model
import mrcnn.config
import os

CLASS_NAMES = ['BG', 'kangaroo']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)
    #IMAGE_MIN_DIM = 256
    #IMAGE_MAX_DIM = 512
    IMAGE_META_SIZE = 14
	
class TensorflowLiteClassificationModel:
    def __init__(self, model_path, labels, image_size=224):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        print('input0',self._input_details[0]) #,len(self._input_details)) #,self._input_details[0]['shape'])
        print('input1',self._input_details[1]) #,len(self._input_details)) #,self._input_details[0]['shape'])
        print('input2',self._input_details[2]) #,len(self._input_details)) #,self._input_details[0]['shape'])
        print('output',self._output_details[0])
        self.labels = labels
        self.image_size=image_size

    def run_from_filepath(self, image_path):
        input_data_type = self._input_details[0]["dtype"]
        image = np.array(Image.open(image_path).resize((self.image_size, self.image_size)), dtype=input_data_type)
        if input_data_type == np.float32:
            image = image / 255.

        if image.shape == (1, 224, 224):
            image = np.stack(image*3, axis=0)

        return self.run(image)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """
   
		
        model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())
        molded_images, image_metas, windows = model.mold_inputs([image])
        anchors = model.get_anchors(molded_images[0].shape)


        # Load the TFLite model and allocate tensors.
        #interpreter = tf.lite.Interpreter(model_path="model.tflite")
        #interpreter.allocate_tensors()

        # Get input and output tensors.
        #input_details = interpreter.get_input_details()
        #output_details = interpreter.get_output_details()

        # There are 3 inputs needed
        #print('shp',molded_images.shape)
        print("image",image.shape,image_metas.shape, anchors.shape)
        self.interpreter.set_tensor(self._input_details[0]['index'],  molded_images.astype('float32'))
        self.interpreter.set_tensor(self._input_details[1]['index'], image_metas.astype('float32'))
        self.interpreter.set_tensor(self._input_details[2]['index'], np.array([anchors]))


        #image= np.expand_dims(image, axis=0) #GL
        #self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output) #[0])
        print('pred',probabilities)



model = TensorflowLiteClassificationModel("kangaroo.tflite",['bg','kangaroo'],256)
model.run_from_filepath("kangaroo-transfer-learning/sample2.jpg") #00182.jpg")




	