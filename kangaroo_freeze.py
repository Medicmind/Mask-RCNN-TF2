import tensorflow as tf

from PIL import Image
import glob
import numpy as np
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import os
import random


CLASS_NAMES = ['BG', 'kangaroo']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_META_SIZE = 14
	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)
config=SimpleConfig()
print("config")
#config.display()

mm = mrcnn.model.MaskRCNN(mode="inference", 
                             config=config,
                             model_dir=os.getcwd())

# Load the weights into the model.
mm.load_weights(filepath="kangaroo-transfer-learning/Kangaro_mask_rcnn_trained.h5", 
                   by_name=True)
	
IMAGE_DIR="kangaroo-transfer-learning/kangaroo/images"	
file_names = next(os.walk(IMAGE_DIR))[2]
image = np.array(Image.open(os.path.join(IMAGE_DIR, random.choice(file_names))))
#image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = mm.detect([image], verbose=1)

# Visualize results
print(results)
r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                            class_names, r['scores'])	
#mm = nfnets.NFNetF2(num_classes=2,pretrained=None)
#mm.load_weights('checkpoints/nfnets.NFNetF2_LAMB_glaucoma75_batchsize_8_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_latest.h5') 
#mm = resnet_family.RegNetZD(num_classes=2,pretrained=None)
#mm.load_weights('checkpoints/resnet_family.RegNetZD_LAMB_diabetic_oedema2_batchsize_8_randaug_6_mixup_0.1_cutmix_1.0_RRC_0.08_lr512_0.008_wd_0.02_epoch_12_val_acc_0.9570.h5') 


# Default one
#concrete_func = mm.keras_model.signatures[
#  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
#concrete_func.inputs[0].set_shape([None, 512,512, 3])
#converter = tf.light.TFLiteConverter.from_concrete_functions([concrete_func])

converter = tf.lite.TFLiteConverter.from_keras_model(mm.keras_model) #,input_arrays_with_shape=[('input', [1,512,512,3])])
converter.allow_custom_ops=True
converter.experimental_new_converter = True
#converter.input_shape=(None,640,640,3)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
#converter.target_spec.supported_types = [tf.float16]

open("kangaroo.tflite", "wb").write(converter.convert())