import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from game2048.agents import Agent


class NNAgent(Agent):
    def define_model(self):
        tf.keras.backend.clear_session()
        if self.training:
            train_input=keras.Input(shape=(None,4,4),name='float_input')
            cat_input=keras.Input(shape=(None,4,4,16),name='cat_input')
            stateful=False
        else:
            if self.vote:
                batch_size=8
            else:
                batch_size=1
            train_input=keras.Input(batch_shape=(batch_size,1,4,4),name='float_input')
            cat_input=keras.Input(batch_shape=(batch_size,1,4,4,16),name='cat_input')
            stateful=True

        mask_value=np.ones((4,4),dtype=int)
        mask=layers.Masking(mask_value=mask_value)(train_input)
            
        stdreshape=layers.Reshape((-1,16))(train_input)
        floatConv=layers.Reshape((-1,4,4,1))(train_input)
        trainable=True
        Conved_1=layers.TimeDistributed(layers.Conv2D(222, (2, 2),padding="same"
                                      ,activation="relu",kernel_initializer='he_uniform',trainable=trainable))(cat_input)
        Conved_2=layers.TimeDistributed(layers.Conv2D(222, (2, 2),padding="same"
                                      ,activation="relu",kernel_initializer='he_uniform',trainable=trainable))(Conved_1)
        Conved_3=layers.TimeDistributed(layers.Conv2D(222, (2, 2),padding="same"
                                      ,activation="relu",kernel_initializer='he_uniform',trainable=trainable))(Conved_2)
        Conved_4=layers.TimeDistributed(layers.Conv2D(222, (2, 2),padding="same"
                                      ,activation="relu",kernel_initializer='he_uniform',trainable=trainable))(Conved_3)
        Conved_5=layers.TimeDistributed(layers.Conv2D(222, (2, 2),padding="same"
                                      ,activation="relu",kernel_initializer='he_uniform',trainable=trainable))(Conved_4)
        pooled=layers.TimeDistributed(layers.GlobalMaxPool2D())(Conved_5)
        train_out=layers.Dense(4, activation='softmax',name='Dense')(pooled)
        
        model=keras.Model([train_input,cat_input],train_out)
        if self.training:
            model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.RMSprop(),
                              metrics=['CategoricalAccuracy'])
        return model
    
    def __init__(self, game, display=None, training=False,modelname='2048_6_18_CNN5_pool_whatever_data_I_have',debug=False,vote=True):
        self.vote=vote
        self.debug=debug
        self.training=training
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            if not training:
                 tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    #             tf.config.gpu.set_per_process_memory_fraction(0.75)
            else:
                tf.config.experimental.set_memory_growth(gpu_devices[0], True)  #strange workaround for a bug on RTX card
        if debug:
            from game2048.agents import ExpectiMaxAgent
            from .expectimax import board_to_move
            self.search_func = board_to_move
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        
        
        self.model=self.define_model()
        if training==False:
            self.model.load_weights(modelname).expect_partial()
            

         
    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        n_bad_decision =0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            if self.debug:
                good_direction,t= self.search_func(self.game.board)
                if (good_direction!=direction):
                    n_bad_decision+=1
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.debug:
                    print("======CorrectDirection: {}======".format(
                        ["left", "down", "right", "up"][good_direction]))
                if self.display is not None:
                    self.display.display(self.game)
        if self.debug:
            if self.display is not None:
                    self.display.display(self.game)
            return 1-n_bad_decision/n_iter
    
    def encode(self):
        x=self.game.board
        return np.log2(x + (x == 0))

    def step(self):
        formatted=np.empty((1,1,4,4),dtype=float)
        formatted[0,0]=self.encode()
        if self.vote:
            X=np.zeros((8,1,4,4),dtype=int)
#             Y=np.zeros((8,4,4),dtype=float)
            for k in range(0,4):
                X[k,0]=np.rot90(formatted,k=k,axes=(2,3))
                X[k+4,0]=np.flip(X[k,0],1)
            X_cat=np_utils.to_categorical(X,num_classes=16)    
            Y=self.model.predict([X,X_cat])
            
            Y=Y.reshape((8,4))
            for k in range(0,4):
                Y[k,0:4]=np.hstack([Y[k,k:],Y[k,:k]])
                Y[k+4,[0,2]]=Y[k+4,[0,2]].copy()
                Y[k+4,0:4]=np.hstack([Y[k+4,k:],Y[k+4,:k]])
            Y_p=Y
            Y=np.argmax(Y,axis=1)
#             for k in range(0,4):
#                 if Y[k+4]==0 or Y[k+4]==2:
#                     Y[k+4]=2-Y[k+4]
#                 Y[k]=(Y[k]-k)%4
#                 Y[k+4]=(Y[k+4]-k)%4
            Y_cat=np_utils.to_categorical(Y,num_classes=4)  
            Y_sum=np.sum(Y_cat,axis=0)
            max_direction=np.where(Y_sum==np.max(Y_sum))
            if not np.size(max_direction[0])==1:
                probability=np.sum(Y_p,axis=0)
                t=np.zeros((4,),dtype=float)
                for direction in range(np.size(max_direction[0])):
                    t[max_direction[0][direction]]=probability[max_direction[0][direction]]
                direction=np.argmax(t)                 
            else:
                direction=max_direction[0][0]
#             np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#             print (Y_p,direction)
        else:
            cat_formatted=np_utils.to_categorical(formatted,num_classes=16)
            formatted[0,0]=formatted[0,0]/np.max(formatted[0,0])
            Y=self.model.predict([formatted,cat_formatted],batch_size=1)
            direction=np.argmax(Y)
        
        return direction