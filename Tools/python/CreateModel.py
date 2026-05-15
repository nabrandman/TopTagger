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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_index = self.add_weight(
            name='batch_index',
            shape=(),
            initializer='zeros',
            trainable=False,
            dtype=tf.float32
            )
        self.gradientReversalWeight = 0

    def call(self, x, training=None):
        self.gradientReversalWeight = 2/(1+tf.math.exp(-self.batch_index/10000.0)) - 1
        if training:
            self.batch_index.assign_add(1.0)
        return gradient_reversal_op(x, self.gradientReversalWeight)

    #def get_config(self):
        #config = super().get_config()
        #config.update({'gradientReversalWeight': self.gradientReversalWeight})
        #config.update({'batch_index': self.batch_index})
        #return config

class CustomLoss(tf.keras.layers.Layer):
    def call(self, inputs):
        y_true, y_pred, p_true, p_pred, wgt, w_fc, reg = inputs
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        cross_entropy = loss_fn(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=tf.reduce_sum(y_true, axis=1)*tf.reshape(wgt, [-1])
        )

        cross_entropy_d = loss_fn(
            y_true=p_true,
            y_pred=p_pred,
            sample_weight=tf.reduce_sum(p_true, axis=1)*tf.reshape(wgt, [-1]),
        )

        l2_norm = tf.constant(0.0)
        for w in w_fc.values():
           l2_norm += tf.nn.l2_loss(w[0])
        loss_final = cross_entropy + cross_entropy_d + l2_norm*reg
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        correct_prediction_d = tf.equal(tf.argmax(p_pred, 1), tf.argmax(p_true, 1))
        total_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        total_accuracy_d = tf.reduce_mean(tf.cast(correct_prediction_d, tf.float32))
        self.add_loss(loss_final)
        #self.add_metric(loss_final, name='loss')
        #self.add_metric(cross_entropy, name='label_loss')
        #self.add_metric(cross_entropy_d, name='domain_loss')
        #self.add_metric(l2_norm*reg, name='l2_norm_loss')
        #self.add_metric(total_accuracy, name='accuracy')
        return loss_final, total_accuracy, total_accuracy_d, cross_entropy, cross_entropy_d, l2_norm*reg

class CustomModel(tf.keras.Model):
    def __init__(self, l2Reg=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name='La')
        self.accuracy_tracker_domain = tf.keras.metrics.Mean(name='Da')
        self.label_loss_tracker = tf.keras.metrics.Mean(name='Ll')
        self.domain_loss_tracker = tf.keras.metrics.Mean(name='Dl')
        self.l2_loss_tracker = tf.keras.metrics.Mean(name='L2l')
        self.mae_labels = tf.keras.metrics.MeanAbsoluteError(name="Lm")
        self.mae_domains = tf.keras.metrics.MeanAbsoluteError(name='Dm')
        self.loss_fn = CustomLoss()
        self.reg = l2Reg
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_pred, p_pred = self(data, training=True)
            w_fc = {}
            for layer in self.layers:
                if 'addResult' in layer.name:
                    w_fc[layer.name] = layer.weights
            loss, accuracy, accuracy_domain, label_loss, domain_loss, l2_loss = self.loss_fn([data['y_'], y_pred, data['p_'], p_pred, data['wgt'], w_fc, self.reg])
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.loss_tracker.update_state(loss)
        self.mae_labels.update_state(data['y_'], y_pred)
        self.mae_domains.update_state(data['p_'], p_pred)
        self.label_loss_tracker.update_state(label_loss)
        self.domain_loss_tracker.update_state(domain_loss)
        self.l2_loss_tracker.update_state(l2_loss)
        self.accuracy_tracker.update_state(accuracy)
        self.accuracy_tracker_domain.update_state(accuracy_domain)
        return {
            'loss': self.loss_tracker.result(),
            'Ll': self.label_loss_tracker.result(),
            'Dl': self.domain_loss_tracker.result(),
            'L2l': self.l2_loss_tracker.result(),
            'La': self.accuracy_tracker.result(),
            'Da': self.accuracy_tracker_domain.result(),
            'Lm': self.mae_labels.result(),
            'Dm': self.mae_domains.result(),
        }

    def test_step(self, data):
        y_pred, p_pred = self(data, training=False)
        w_fc = {}
        for layer in self.layers:
            if 'addResult' in layer.name:
                w_fc[layer.name] = layer.weights
        loss, accuracy, accuracy_domain, label_loss, domain_loss, l2_loss = self.loss_fn([data['y_'], y_pred, data['p_'], p_pred, data['wgt'], w_fc, self.reg])
        
        self.loss_tracker.update_state(loss)
        self.mae_labels.update_state(data['y_'], y_pred)
        self.mae_domains.update_state(data['p_'], p_pred)
        self.label_loss_tracker.update_state(label_loss)
        self.domain_loss_tracker.update_state(domain_loss)
        self.l2_loss_tracker.update_state(l2_loss)
        self.accuracy_tracker.update_state(accuracy)
        self.accuracy_tracker_domain.update_state(accuracy_domain)
        return {
            'loss': self.loss_tracker.result(),
            'Ll': self.label_loss_tracker.result(),
            'Dl': self.domain_loss_tracker.result(),
            'L2l': self.l2_loss_tracker.result(),
            'La': self.accuracy_tracker.result(),
            'Da': self.accuracy_tracker_domain.result(),
            'Lm': self.mae_labels.result(),
            'Dm': self.mae_domains.result(),
        }

        
    @property
    def metrics(self):
        return [self.loss_tracker, self.label_loss_tracker, self.domain_loss_tracker, self.l2_loss_tracker, self.accuracy_tracker, self.accuracy_tracker_domain, self.mae_labels, self.mae_domains]

class CreateModel:
    def __init__(self, options, nnStruct, convLayers, rnnNodes, rnnLayers, offset_initial, scale_initial, l2reg, keep_prob, training):
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

        self.custom_layers_dict = {'CustomModel': CustomModel, 'GradientReversalLayer': GradientReversalLayer, 'CustomLoss': CustomLoss}

    def createDenseNetwork(self, denseInputLayer, nnStruct, w_fc, b_fc, keep_prob=1.0, training = False, domainAdaptionConnectionLayer = 9999, NDomains = 3, prefix=""):
        #constants 
        NLayer = len(nnStruct)
        share = len(prefix) > 0
        
        #variables
        h_fc = {}
        addResult = {}

        # Fully connected input layer
        print('not share:', not share)
        batch_normalizer = tf.keras.layers.BatchNormalization(name="layer0_bn")
        h_fc[0] = batch_normalizer(denseInputLayer)
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

            #if not (layer - 1) in w_fc:
            #    w_fc[layer - 1], b_fc[layer - 1] = addResult_layers[layer - 1].get_weights()
            #add batch normalization 
            new_batch_normalizer = tf.keras.layers.BatchNormalization(name="layer%i_bn"%layer)
            batchNormalizedLayer = new_batch_normalizer(addResult[layer - 1])
            if self.options.netOp.denseActivationFunc == "none":
                layerOutput = tf.keras.layers.Dense(batchNormalizedLayer.shape[1], activation='relu', use_bias=False, kernel_initializer=tf.keras.initializers.Identity(), name="h_fc%i%s"%(layer,prefix))(batchNormalizedLayer)
            else:
                layerOutput = tf.keras.layers.Dense(batchNormalizedLayer.shape[1], activation=self.options.netOp.denseActivationFunc, use_bias=False, kernel_initializer=tf.keras.initializers.Identity(), name="h_fc%i%s"%(layer,prefix))(batchNormalizedLayer)
            #add dropout 
            h_fc[layer] = tf.keras.layers.Dropout(rate=1-keep_prob)(layerOutput)
                
        #create yt for input to the softmax cross entropy for classification (this should not have softmax applied as the loss function will do this)
        # for primary class classification
        addResult_layers[NLayer - 2] = tf.keras.layers.Dense(nnStruct[NLayer - 1], kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2.0/nnStruct[NLayer - 2]), name="addResult_layer_yt"+prefix)
        yt = addResult_layers[NLayer - 2](h_fc[NLayer - 2])
        #if not (NLayer - 2) in w_fc:
        #    w_fc[NLayer - 2], b_fc[NLayer - 2] = addResult_layers[NLayer - 2].get_weights()

        #create pt for domain classification
        layer = NLayer - 1
        connectionLayer = min(domainAdaptionConnectionLayer, NLayer - 2)
        pt_input = GradientReversalLayer(name='gradientReversal_layer_%i'%(connectionLayer))(h_fc[connectionLayer])
        addResult_layers[layer] = tf.keras.layers.Dense(NDomains, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2.0/nnStruct[connectionLayer]), name="addResult_layer_pt"+prefix)
        pt = addResult_layers[layer](pt_input)
        #if not layer in w_fc:
        #    w_fc[layer], b_fc[layer] = addResult_layers[layer].get_weights()
            
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

        print('\nnnStruct')
        print(self.nnStruct)
        if len(self.nnStruct) < 2:
            raise

        #Define inputs and training inputs
        self.x = tf.keras.Input(shape=(self.nnStruct[0],), name="x")
        self.p_ = tf.keras.Input(shape=(3,), name="p_")
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
        self.yt_,    self.pt_    = self.createDenseNetwork(denseInputLayer,    self.nnStruct, self.w_fc, self.b_fc, keep_prob=self.keep_prob, training=self.training, domainAdaptionConnectionLayer = self.daLayer, NDomains = int(self.p_.shape[1]))
    
        #final answer with softmax applied for the end user
        self.yt = tf.keras.layers.Softmax()(self.yt_)
        self.pt = tf.keras.layers.Softmax()(self.pt_)
        self.model = CustomModel(inputs=[self.x,self.p_,self.wgt,self.y_], outputs=[self.yt, self.pt], l2Reg=self.reg)

    def get_model(self):
        self.createMLP()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-4),
        )
        return self.model

    def get_layers(self):
        return self.custom_layers_dict
