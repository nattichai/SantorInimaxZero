import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from .cycliclr import CyclicLR

class SantoriniNet:
    def __init__(self, game, args):
        self.args = args
        self.nnet = Backbone(game, args)
        self.val_size = self.args.val_size
        self.board_x, self.board_y = game.board_dim
        self.action_size = game.action_size
        self.frozen_func = None
        
    def prepare(self, state):
        return np.concatenate([[state[0] == i] for i in range(5)] + [[state[1] == p] for p in [1, 2, -1, -2]] + [[np.full((self.board_x, self.board_y), p)] for p in state[2][1:]]).astype(np.float32)

    def train(self, train_data):
        input_states, target_pis, target_vs = list(zip(*train_data))
        input_states = np.asarray([self.prepare(board) for board in input_states])
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        callbacks = []
        
        if self.args.cyclic_lr:
            clr = CyclicLR(mode='triangular2',
                           base_lr=self.args.base_lr,
                           max_lr=self.args.max_lr,
                           step_size=((1. - self.val_size) * input_states.shape[0] // self.args.n_bs) * 4)
            callbacks.append(clr)
        
        if self.args.has_val and self.val_size > 1e-5:
            mc = ModelCheckpoint('./checkpoint/best_train.h5', verbose=0, save_best_only=True, save_weights_only=True)
            callbacks.append(mc)
            
            self.nnet.model.fit(x=input_states,
                                y=[target_pis, target_vs],
                                batch_size=self.args.n_bs,
                                epochs=self.args.n_ep,
                                verbose=2,
                                validation_split=self.val_size,
                                callbacks=callbacks)
            self.nnet.model.load_weights('./checkpoint/best_train.h5')
            
            self.val_size = max(0, self.val_size - self.args.val_decay)
        else:
            self.nnet.model.fit(x=input_states,
                                y=[target_pis, target_vs],
                                batch_size=self.args.n_bs,
                                epochs=self.args.n_ep,
                                verbose=2,
                                callbacks=callbacks)
            
        self.convert_to_tensorrt()
        K.clear_session()

    def predict(self, board):
        input_board = self.prepare(board)[None]
        if self.frozen_func is None:
            self.convert_to_tensorrt()
        if self.frozen_func is not None:
            pi, v = self.frozen_func(tf.convert_to_tensor(input_board))
        else:
            pi, v = self.nnet.model.predict_on_batch(input_board)
        return pi.numpy()[0], v.numpy()[0, 0]

    def save(self, filepath):
        self.nnet.model.save_weights(filepath)

    def load(self, filepath):
        self.nnet.model.load_weights(filepath)
        self.convert_to_tensorrt()
    
    def convert_to_tensorrt(self):
        self.nnet.model.save('./checkpoint/save_model', save_format='tf')
        if self.args.tensorrt_convert:
            converter = trt.TrtGraphConverterV2(input_saved_model_dir='./checkpoint/save_model')
            converter.convert()
            converter.save('./checkpoint/tensorrt_model')

            loaded_model = tf.saved_model.load('./checkpoint/tensorrt_model')
            self.frozen_func = convert_variables_to_constants_v2(loaded_model.signatures['serving_default'])
            
        else:
            loaded_model = tf.saved_model.load('./checkpoint/save_model')
            self.frozen_func = convert_variables_to_constants_v2(loaded_model.signatures['serving_default'])
        
        
class ChannelFirstToLast(Layer):
    def __init__(self):
        super(ChannelFirstToLast, self).__init__()
        
    def call(self, input):
        return tf.transpose(input, [0, 2, 3, 1])
    
class Backbone:
    def __init__(self, game, args):
        self.board_x, self.board_y = game.board_dim
        self.action_size = game.action_size
        self.args = args

        input_boards = Input(shape=(13, self.board_x, self.board_y))
        x_image = ChannelFirstToLast()(input_boards)

        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.n_ch, 3, padding='same', use_bias=False)(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.n_ch, 3, padding='same', use_bias=False)(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.n_ch, 3, padding='valid', use_bias=False)(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(args.n_ch, 3, padding='valid', use_bias=False)(h_conv3)))
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(Dense(args.n_denses[0], use_bias=False)(h_conv4_flat)))
        s_fc2 = Dropout(args.dropout)(Activation('relu')(Dense(args.n_denses[1], use_bias=False)(s_fc1)))
        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=input_boards, outputs=[pi, v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.base_lr))