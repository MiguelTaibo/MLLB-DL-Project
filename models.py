from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Input, Dense, InputLayer,BatchNormalization, Add, Activation
from keras.layers import Embedding, LSTM, ConvLSTM2D, MaxPooling3D, Layer
from keras.constraints import maxnorm
from keras.models import Model
from keras.regularizers import l2

import tensorflow as tf
from keras import applications

def VGG16(dropout, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0, batch_norm = False):
    
    input_image = Input(shape=(img_width,img_height,img_channels))

    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(input_image)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    if batch_norm:
        x1 = BatchNormalization()(x1)

    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    if batch_norm:
        x2 = BatchNormalization()(x2)

    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = MaxPooling2D((2, 2))(x3)
    if batch_norm:
        x3 = BatchNormalization()(x3)

    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = MaxPooling2D((2, 2))(x4)
    if batch_norm:
        x4 = BatchNormalization()(x4)

    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = MaxPooling2D((2, 2))(x5)
    if batch_norm:
        x5 = BatchNormalization()(x5)

    x6 = Flatten()(x5)
    x6=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x6=Dropout(dropout)(x6)
    x6=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x=Dropout(dropout)(x6)
    out= Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x)

    model = Model(inputs = input_image, outputs = out)
    
    return model

def VGG19(dropout, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0, batch_norm = False):

    input_image = Input(shape=(img_width,img_height,img_channels))
   
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(input_image)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    if batch_norm:
        x1=BatchNormalization()(x1)


    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    if batch_norm:
        x2=BatchNormalization()(x2)

    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = MaxPooling2D((2, 2))(x3)
    if batch_norm:
        x3=BatchNormalization()(x3)

    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = MaxPooling2D((2, 2))(x4)
    if batch_norm:
        x4=BatchNormalization()(x4)

    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = MaxPooling2D((2, 2))(x5)
    if batch_norm:
        x5=BatchNormalization()(x5)

    x6 = Flatten()(x5)
    x6 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x6 = Dropout(dropout)(x6)
    x6 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x  = Dropout(dropout)(x6)
    out= Dense(num_classes, activation='softmax')(x)

    model = Model(inputs = input_image, outputs = out)
    
    return model

def MLP_model(dropout, img_width=32, img_height=32, img_channels=3, num_classes=10, l2_reg=0, n_hidden=2, batch_norm = False):
    input_image = Input(shape=(img_width,img_height,img_channels))
    x = Flatten()(input_image)
    
    for layer in reversed(range(n_hidden)):
        x = Dense(int((img_height*img_width*img_channels+num_classes)*(layer+1)/(n_hidden+1)), activation='relu', 
            kernel_constraint=maxnorm(3),
            kernel_initializer="glorot_uniform",
            kernel_regularizer=l2(l2_reg))(x)
        if batch_norm:
            x=BatchNormalization()(x)
        if dropout>0.0:
            x=Dropout(dropout)(x)
        
    out= Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x)
    model = Model(inputs = input_image, outputs = out)
    
    return model


def ResNet21(dropout, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0, batch_norm = False):

    input_image = Input(shape=(img_width,img_height,img_channels))
   
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(input_image)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    if batch_norm:
        x1=BatchNormalization()(x1)


    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x2 = Conv2D(128, (3, 3),padding='same', kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)

    x22 = Conv2D(128, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)

    x2 = Add()([x2,x22])
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    if batch_norm:
        x2=BatchNormalization()(x2)

    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)

    x33 = Conv2D(256, (3, 3),padding='same', kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)

    x3 = Add()([x3, x33])
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D((2, 2))(x3)
    if batch_norm:
        x3=BatchNormalization()(x3)

    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)

    x44 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)    

    x4 = Add()([x4, x44])
    x4 = Activation('relu')(x4)
    x4 = MaxPooling2D((2, 2))(x4)
    if batch_norm:
        x4=BatchNormalization()(x4)

    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)

    x55 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)

    x5 = Add()([x5, x55])
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D((2, 2))(x5)
    if batch_norm:
        x5=BatchNormalization()(x5)

    x6 = Flatten()(x5)
    #x6 = Dropout(dropout)(x6)
    x6 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x6 = Dropout(dropout)(x6)
    x6 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x  = Dropout(dropout)(x6)
    out= Dense(num_classes, activation='softmax')(x)

    model = Model(inputs = input_image, outputs = out)
    
    return model

def ResNet25(dropout, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0, batch_norm = False):

    input_image = Input(shape=(img_width,img_height,img_channels))
   
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(input_image)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    if batch_norm:
        x1=BatchNormalization()(x1)


    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x2 = Conv2D(128, (3, 3),padding='same', kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)

    x22 = Conv2D(128, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x1)

    x2 = Add()([x2,x22])
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    if batch_norm:
        x2=BatchNormalization()(x2)

    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x3 = Conv2D(256, (3, 3),padding='same', kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)

    x33 = Conv2D(256, (3, 3),padding='same', kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x2)

    x3 = Add()([x3, x33])
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D((2, 2))(x3)
    if batch_norm:
        x3=BatchNormalization()(x3)

    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x4 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)

    x44 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)    

    x4 = Add()([x4, x44])
    x4 = Activation('relu')(x4)
    x4 = MaxPooling2D((2, 2))(x4)
    if batch_norm:
        x4=BatchNormalization()(x4)

    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)
    x5 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x5)

    x55 = Conv2D(512, (3, 3),padding='same',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x4)

    x5 = Add()([x5, x55])
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D((2, 2))(x5)
    if batch_norm:
        x5=BatchNormalization()(x5)

    x6 = Flatten()(x5)
    #x6 = Dropout(dropout)(x6)
    x6 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x6 = Dropout(dropout)(x6)
    x6 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x6)
    x  = Dropout(dropout)(x6)
    out= Dense(num_classes, activation='softmax')(x)

    model = Model(inputs = input_image, outputs = out)
    
    return model


def LSTM_singleSweep(dropout, img_width=32, img_height=32, img_channels=3, num_classes=10, l2_reg=0):

    input_image = Input(shape=(img_width, img_height, img_channels))

    x1 = PatchLayer(patch_size=2, horizontal=True)(input_image)  ## Patching layer (no trainnable parameters) 
    x1 = ConvLSTM2D(64, (2,2), return_sequences=True, activation="tanh", kernel_regularizer=l2(l2_reg), dropout=dropout)(x1)
    #x1 = UnpatchLayer()(x1)
    #x1 = MaxPooling2D((2, 2))(x1)

    x2 = Flatten()(x1)
    out= Dense(num_classes, activation='softmax')(x2)

    model = Model(inputs = input_image, outputs = out)
    return model


def LSTM_doubleSweep(dropout, img_width=32, img_height=32, img_channels=3, num_classes=10, l2_reg=0):

    input_image = Input(shape=(img_width, img_height, img_channels))

    x1 = PatchLayer(patch_size=2, horizontal=True)(input_image)  ## Patching layer (no trainnable parameters) 
    x1 = ConvLSTM2D(64, (2,2), return_sequences=True, activation="tanh", kernel_regularizer=l2(l2_reg), dropout=dropout)(x1)
    x1 = UnpatchLayer()(x1)
    #x1 = MaxPooling2D((2, 2))(x1)

    x2 = PatchLayer(patch_size=2, horizontal=False)(x1)  ## Patching layer (no trainnable parameters) 
    x2 = ConvLSTM2D(64, (2,2), return_sequences=True, activation="tanh", kernel_regularizer=l2(l2_reg), dropout=dropout)(x2)
    # x1 = MaxPooling3D((2,2,1))(x1)

    x3 = Flatten()(x2)
    out= Dense(num_classes, activation='softmax')(x3)

    model = Model(inputs = input_image, outputs = out)
    return model

class PatchLayer(Layer):

  def __init__( self , patch_size, horizontal=False):
    super(PatchLayer, self ).__init__()
    self.patch_size = patch_size
    # If horizontal or vertical
    self.horizontal = horizontal

  def call(self, inputs):
    patches = []

    if self.horizontal:
        for i in range( 0 , inputs.shape[1] , self.patch_size ):
            for j in range( 0 , inputs.shape[1] , self.patch_size ):
                patches.append( inputs[:, i : i + self.patch_size , j : j + self.patch_size , : ])
    else:
        for j in range( 0 , inputs.shape[1] , self.patch_size ):
            for i in range( 0 , inputs.shape[1] , self.patch_size ):
                patches.append( inputs[:, i : i + self.patch_size , j : j + self.patch_size , : ])

    patches = tf.transpose(tf.convert_to_tensor(patches),[1,0,2,3,4])
    return patches

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        "patch_size" : self.patch_size
    })
    return config

class UnpatchLayer(Layer):
    def __init__( self):
        super(UnpatchLayer, self ).__init__()

    def call(self, inputs):

        len = tf.cast(tf.math.sqrt(tf.cast(inputs.shape[1],tf.float64)),tf.int32)
        return tf.reshape(inputs,[-1, len,len, inputs.shape[4]])

def Transfer_model(dropout, img_width=32, img_height=32, img_channels=3, num_classes=10,l2_reg=0, batch_norm = False):
    input_image = Input(shape=(img_width,img_height,img_channels))
    train_model = applications.VGG16(weights='imagenet',input_tensor=input_image, include_top=False,input_shape=(img_width, img_height, img_channels), pooling=None)
    if batch_norm:
        x = BatchNormalization()(train_model.layers[-1].output)
        x = Flatten()(x)
    else:
        x = Flatten()(train_model.layers[-1].output)
    x=Dropout(dropout)(x)
    x=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x)
    x=Dropout(dropout)(x)
    x=Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x)
    x=Dropout(dropout)(x)
    out= Dense(num_classes, activation='softmax',kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x)

    model = Model(inputs = input_image, outputs = out)
    #Freezing top layers (Don't change the value in the trainng process)
    for layer in train_model.layers:
            layer.trainable = False

    return model
