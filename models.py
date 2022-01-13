from keras.layers import Conv2D, Dropout, Flatten, MaxPooling2D, Input, Dense, InputLayer,BatchNormalization
from keras.layers import Embedding, LSTM, ConvLSTM2D, MaxPooling3D, Layer
from keras.constraints import maxnorm
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf

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
    x6=Dropout(dropout)(x6)
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
    x6 = Dropout(dropout)(x6)
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

def LSTM_model(dropout, img_width=32, img_height=32, img_channels=3, num_classes=10,l2_reg=0,n_hidden=2, batch_norm = False):

    input_image = Input(shape=(img_width, img_height, img_channels))

    x1 = PatchLayer(patch_size=2, h_or_v=True, f_or_b=True)(input_image)
    
    # x1 = Embedding(1, 256)(x1)
    x1 = ConvLSTM2D(256, (2,2), return_sequences=True, activation="tanh", kernel_regularizer=l2(l2_reg))(x1)
    
    # x1 = MaxPooling3D(pool_size=(1, 2, 2)) (x1)

    # x2 = ConvLSTM2D(128, (3,3), padding="same", return_sequences=True, activation="tanh", kernel_regularizer=l2(l2_reg))(x1)
    # x2 = ConvLSTM2D(128, (3,3), padding="same", return_sequences=True, activation="tanh", kernel_regularizer=l2(l2_reg))(x2)
    # x2 = MaxPooling3D(pool_size=(1, 2, 2)) (x2)

    x1 = Flatten()(x1)
    # x3 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    # x3 = Dropout(dropout)(x3)
    # x3 = Dense(512, activation='relu', kernel_constraint=maxnorm(3),kernel_initializer="glorot_uniform",kernel_regularizer=l2(l2_reg))(x3)
    # x  = Dropout(dropout)(x3)
    out= Dense(num_classes, activation='softmax')(x1)

    model = Model(inputs = input_image, outputs = out)
    return model

class PatchLayer(Layer):

  def __init__( self , patch_size, h_or_v = False, f_or_b = False ):
    super(PatchLayer, self ).__init__()
    self.patch_size = patch_size
    self.h_or_v = h_or_v
    self.f_or_b = f_or_b

  def call(self, inputs):
    patches = []

    for i in range( 0 , inputs.shape[1] , self.patch_size ):
        for j in range( 0 , inputs.shape[1] , self.patch_size ):
            patches.append( inputs[:, i : i + self.patch_size , j : j + self.patch_size , : ] )
    patches = tf.transpose(tf.convert_to_tensor(patches),[1,0,2,3,4])
    return patches

    # if self.h_or_v: ## Horizontal patching
    #     iterator = range( 0 , inputs.shape[2] , self.patch_size ) if self.f_or_b else reversed(range( 0 , inputs.shape[2] , self.patch_size ))

    #     for i in range( 0 , inputs.shape[1] , self.patch_size ):
    #         for j in iterator:
    #             patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )

    # else: ## Vertical patching
    #     iterator = range( 0 , inputs.shape[1] , self.patch_size ) if self.f_or_b else reversed(range( 0 , inputs.shape[1] , self.patch_size ))

    #     for j in range( 0 , inputs.shape[2] , self.patch_size ):
    #         for i in iterator:
    #             patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
