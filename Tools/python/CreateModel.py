import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.data_flow_ops import StagingArea

@tf.custom_gradient
def gradient_reversal_op(x, grw=1.0):
    y = tf.identity(x)

    def grad(dy):
        return -dy * grw, None

    return y, grad
        
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, grw=1.0, **kwargs):
        super().__init__(**kwargs)
        self.gradientReversalWeight = grw

    def call(self, x):
        return gradient_reversal_op(x, self.gradientReversalWeight)

    def get_config(self):
        config = super().get_config()
        config.update({'gradientReversalWeight': self.gradientReversalWeight})
        return config

class LossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        y_true, y_pred, p_true, p_pred, wgt, w_fc, reg = inputs
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        cross_entropy = loss_fn(
            y_true=y_true,
            y_pred=y_pred,
            #sample_weight=tf.reduce_sum(tf.cast(y_true, dtype=tf.float32), axis=1, keepdims=True)*wgt,
            sample_weight=tf.reduce_sum(y_true, axis=1, keepdims=True)*wgt,
        )

        cross_entropy_d = loss_fn(
            y_true=p_true,
            y_pred=p_pred,
            #sample_weight=tf.reduce_sum(tf.cast(p_true, dtype=tf.float32), axis=1, keepdims=True)*wgt,
            sample_weight=tf.reduce_sum(p_true, axis=1, keepdims=True)*wgt,
        )

        l2_norm = tf.constant(0.0)
        for w in w_fc.values():
            l2_norm += tf.nn.l2_loss(w)
        loss_final = cross_entropy + cross_entropy_d + l2_norm*reg
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        total_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.add_loss(loss_final)
        self.add_metric(loss_final, name='loss')
        self.add_metric(total_accuracy, name='accuracy')
        return y_pred

class CreateModel:
    def __init__(self, options, nnStruct, convLayers, rnnNodes, rnnLayers, offset_initial, scale_initial, l2reg, keep_prob, training, grw):
        self.options = options        

        self.nnStruct = nnStruct
        self.convLayers = convLayers
        self.rnnNodes = rnnNodes
        self.rnnLayers = rnnLayers
        self.offset_initial = offset_initial
        self.scale_initial = scale_initial
        self.daLayer = options.netOp.daLayer
        self.reg = l2reg
        self.keep_prob = keep_prob
        self.training = training
        self.gradientReversalWeight = grw



        self.custom_layers_dict = {'GradientReversalLayer': GradientReversalLayer, 'LossLayer': LossLayer}

    def get_layers(self):
        return self.custom_layers_dict

    def createDenseNetwork(self, denseInputLayer, nnStruct, w_fc, b_fc, keep_prob=1.0, gradientReversalWeight=1.0, training = False, domainAdaptionConnectionLayer = 9999, NDomains = 2, prefix=""):
        #constants 
        NLayer = len(nnStruct)
        share = len(prefix) > 0
        
        #variables
        h_fc = {}
        addResult = {}

        # Fully connected input layer
        batch_normalizer = tf.keras.layers.BatchNormalization(trainable=(not share), name="layer0_bn")
        h_fc[0] = batch_normalizer(denseInputLayer, training=training)
        addResult_layers = {}

        # create hidden layers 
        for layer in range(1, NLayer - 1):
            #use relu for hidden layers as this seems to give best result

            if layer == 1:
                kernel_shape = int(denseInputLayer.shape[1])
            else:
                kernel_shape = nnStruct[layer]
            addResult_layers[layer - 1] = tf.keras.layers.Dense(nnStruct[layer], kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2.0/kernel_shape), name='addResult_layer'+str(layer-1))
            addResult[layer - 1] = addResult_layers[layer - 1](h_fc[layer - 1])

            if not (layer - 1) in w_fc:
                w_fc[layer - 1], b_fc[layer - 1] = addResult_layers[layer - 1].get_weights()
            #add batch normalization 
            new_batch_normalizer = tf.keras.layers.BatchNormalization(trainable=(not share), name="layer%i_bn"%layer)
            batchNormalizedLayer = new_batch_normalizer(addResult[layer - 1], training=training)
            if self.options.netOp.denseActivationFunc == "none":
                layerOutput = tf.keras.layers.Dense(batchNormalizedLayer.shape[1], activation='relu', use_bias=False, kernel_initializer=tf.keras.initializers.Identity(), name="h_fc%i%s"%(layer,prefix))(batchNormalizedLayer)
                #layerOutput = batchNormalizedLayer
            else:
                layerOutput = tf.keras.layers.Dense(batchNormalizedLayer.shape[1], activation=self.options.netOp.denseActivationFunc, use_bias=False, kernel_initializer=tf.keras.initializers.Identity(), name="h_fc%i%s"%(layer,prefix))(batchNormalizedLayer)
            #add dropout 
            h_fc[layer] = tf.keras.layers.Dropout(rate=1-keep_prob)(layerOutput)
                
        #create yt for input to the softmax cross entropy for classification (this should not have softmax applied as the loss function will do this)
        # for primary class classification

        addResult_layers[NLayer - 2] = tf.keras.layers.Dense(nnStruct[NLayer - 1], kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2.0/nnStruct[NLayer - 2]), name="yt"+prefix)
        yt = addResult_layers[NLayer - 2](h_fc[NLayer - 2])
        if not (NLayer - 2) in w_fc:
            w_fc[NLayer - 2], b_fc[NLayer - 2] = addResult_layers[NLayer - 2].get_weights()

        #create pt for domain classification
        layer = NLayer - 1
        connectionLayer = min(domainAdaptionConnectionLayer, NLayer - 2)
        pt_input = GradientReversalLayer(grw=gradientReversalWeight)(h_fc[connectionLayer])
        addResult_layers[layer] = tf.keras.layers.Dense(NDomains, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2.0/nnStruct[connectionLayer]), name="pt"+prefix)
        pt = addResult_layers[layer](pt_input)
        if not layer in w_fc:
            w_fc[layer], b_fc[layer] = addResult_layers[layer].get_weights()

        return yt, pt

    ### createMLP
    # This fucntion is designed to create a MLP for classification purposes (using softmax_cross_entropy_with_logits)
    # inputs 
    #  nnStruct - a list containing the number of nodes in each layer, including the input and output layers 
    #  offset_initial - a list of offsets which will be applied to the initial input features, they are stored in the tf model
    #  scale_initial - a list of scales which will be applied to each input feature after the offsets are subtracted, they are stored in the tf model
    def createMLP(self):
        #constants 
        NLayer = len(self.nnStruct)
    
        if len(self.nnStruct) < 2:
            #throw
            raise
        
        #Define inputs and training inputs
        self.x = tf.keras.Input(shape=(self.nnStruct[0],), name="x")
        self.p_ = tf.keras.Input(shape=(self.nnStruct[NLayer - 1],), name="p_")
        self.wgt = tf.keras.Input(shape=(1,), name="wgt")
        self.y_ = tf.keras.Input(shape=(self.nnStruct[NLayer - 1],), name="y_")

        #variables for pre-transforming data
        
        #input variables after rescaling 
        subtractX = self.x-self.offset_initial
        denseInputLayer = subtractX*self.scale_initial

        #variables for weights and activation functions 
        self.w_fc = {}
        self.b_fc = {}

        #create dense network
        self.yt,    self.pt    = self.createDenseNetwork(denseInputLayer,    self.nnStruct, self.w_fc, self.b_fc, keep_prob=self.keep_prob, gradientReversalWeight=self.gradientReversalWeight, training=self.training, domainAdaptionConnectionLayer = self.daLayer, NDomains = int(self.p_.shape[1]))
    
        #final answer with softmax applied for the end user
        self.y = LossLayer(name='y')([self.y_, self.yt, self.p_, self.pt, self.wgt, self.w_fc, self.reg])
                
        self.model = tf.keras.Model(inputs=[self.x,self.p_,self.wgt,self.y_], outputs=self.y)

    def get_model(self):
        self.createMLP()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-4),
            loss=None,
            )
        return self.model
