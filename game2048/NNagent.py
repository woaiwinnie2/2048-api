import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from game2048.agents import Agent

class NNAgent(Agent):
    def __init__(self, game, display=None, training=False,modelname='2048_5_24',stateful=False,debug=False):
        self.debug=debug
        if debug:
            from game2048.agents import ExpectiMaxAgent
            from .expectimax import board_to_move
            self.search_func = board_to_move
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        
        tf.keras.backend.clear_session()
        if stateful==False:
            train_input=keras.Input(shape=(None,4,4),name='float_input')
            cat_input=keras.Input(shape=(None,4,4,15),name='cat_input')
        else:
            train_input=keras.Input(batch_shape=(1,1,4,4),name='float_input')
            cat_input=keras.Input(batch_shape=(1,1,4,4,15),name='cat_input')
            
        stdreshape=layers.Reshape((-1,16))(train_input)
        floatConv=layers.Reshape((-1,4,4,1))(train_input)
        
        conv_2d_layer_1 = layers.Conv2D(128, (1, 2))
        conv_2d_layer_2 = layers.Conv2D(128, (2, 1))
        conv_2d_layer_3 = layers.Conv2D(128, (1, 2))
        conv_2d_layer_4 = layers.Conv2D(128, (2, 1))
        conv_2d_layer_5 = layers.Conv2D(128, (1, 2))
        conv_2d_layer_6 = layers.Conv2D(128, (2, 1))
        Conved_1=layers.TimeDistributed(conv_2d_layer_1)(cat_input)
        Conved_2=layers.TimeDistributed(conv_2d_layer_2)(cat_input)
        Conved_3=layers.TimeDistributed(conv_2d_layer_3)(Conved_1)
        Conved_4=layers.TimeDistributed(conv_2d_layer_4)(Conved_1)
        Conved_5=layers.TimeDistributed(conv_2d_layer_5)(Conved_2)
        Conved_6=layers.TimeDistributed(conv_2d_layer_6)(Conved_2)
        flat_Conved_1=layers.TimeDistributed(layers.Flatten())(Conved_1)
        flat_Conved_2=layers.TimeDistributed(layers.Flatten())(Conved_2)
        flat_Conved_3=layers.TimeDistributed(layers.Flatten())(Conved_3)
        flat_Conved_4=layers.TimeDistributed(layers.Flatten())(Conved_4)
        flat_Conved_5=layers.TimeDistributed(layers.Flatten())(Conved_5)
        flat_Conved_6=layers.TimeDistributed(layers.Flatten())(Conved_6)
        combined=K.concatenate([flat_Conved_3,flat_Conved_4,flat_Conved_5,flat_Conved_6])
#         LSTM=layers.LSTM(16,
#             return_sequences=True,
#             stateful=stateful)(combined)
#         combined=K.concatenate([combined,LSTM])
        combined=layers.Dense(192, activation='relu',name='DenseH')(combined)
        train_out=layers.Dense(4, activation='relu',name='DenseL')(combined)
        
        self.model=keras.Model([train_input,cat_input],train_out)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.RMSprop(),
                          metrics=['CategoricalAccuracy'])
        
        if training==False:
            self.model.load_weights(modelname).expect_partial()
            

         
    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        n_bad_decision =0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            if self.debug:
                good_direction= self.search_func(self.game.board)
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
            return n_bad_decision/n_iter
    
    def encode(self):
        list=[0,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]
        code=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        output=np.zeros((4,4),dtype=int)
        for i in code:
            output[np.where(self.game.board==list[i])]=code[i]
        return output

    def step(self):
        formatted=np.empty((1,1,4,4),dtype=float)
        formatted[0,0]=self.encode()
        cat_formatted=np_utils.to_categorical(formatted,num_classes=15)
        
        formatted[0,0]=formatted[0,0]/np.max(formatted[0,0])
        Y=self.model.predict([formatted,cat_formatted],batch_size=1)
        direction=np.argmax(Y)
        
        return direction