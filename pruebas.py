from models import LSTM_doubleSweep, LSTM_singleSweep
model = LSTM_singleSweep(0, num_classes=10, img_width=32, img_height=32, img_channels=3)
print(model.summary())

model = LSTM_doubleSweep(0, num_classes=10, img_width=32, img_height=32, img_channels=3)
print(model.summary())
# import tensorflow as tf
# from models import PatchLayer,UnpatchLayer

# inputs = tf.random.uniform(shape=[1, 8,8,3])
# print(inputs.shape, inputs)
# x1 = PatchLayer(patch_size=2)(inputs)  ## Patching layer (no trainnable parameters) 
# print(x1.shape, x1)
# x2 = tf.reshape(x1[:,:,0,0,:],[1,16,1,1,3])
# x3 = UnpatchLayer()(x2)
# import pdb
# pdb.set_trace()
# outputs = tf.reshape(inputs,[1, 4,4, inputs.shape[3]])
# print(outputs.shape, outputs )
