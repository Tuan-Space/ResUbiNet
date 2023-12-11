# https://kexue.fm/archives/5112
from keras import activations
from keras import backend as K
from keras.layers import Layer
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) # S向量模长的平方
    scale = (s_squared_norm+K.epsilon())/(0.5 + s_squared_norm)/K.sqrt(s_squared_norm)
    return scale * x

# def squash(x, axis=-1):
#     s_squared_norm = K.sum(K.square(x), axis, keepdims=True) # S向量模长的平方
#     scale = (K.sqrt(s_squared_norm)+K.epsilon())/(0.5 + s_squared_norm)/K.sqrt(s_squared_norm)
#     return scale * x

# define our own softmax function instead of K.softmax
# keepdims=True保持矩阵维数不变(被减少的那个轴会以维度1保留在结果中)
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):  # u_vecs是输入
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)  # 使用一维卷积替代全连接
            print(u_vecs.shape)
            print(u_vecs)
            print(self.W)

        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]  # 批次数目
        input_num_capsule = K.shape(u_vecs)[1]  # 输入胶囊的个数
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))  # 交换轴
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0])  # shape = [None, num_capsule, input_num_capsule]  # 步骤1，将b初始化为0
        # 动态路由部分
        for i in range(self.routings):
            c = softmax(b, 1)  # 步骤2，对b进行softmax
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])  # 批量矩阵相乘 原注释
            # print('动态路由前c的shape', c.shape)
            # print('动态路由前u_hat_vecs的shape', u_hat_vecs.shape)
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)  # 原代码
            # o = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            # print('动态路由前o的shape', o.shape)
            o = self.activation(o)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)

            if i < self.routings - 1:
                # o = K.l2_normalize(o, -1)  # 原代码
                # print('动态路由后o的shape', o.shape)
                # print('动态路由后u_hat_vecs的shape', u_hat_vecs.shape)
                # b += K.batch_dot(o, u_hat_vecs, [2, 3])  # 原注释
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)  # 原代码
                # print('动态路由后b的shape', b.shape)
                if K.backend() == 'theano':
                    # b = K.sum(b, axis=1)  # 原代码
                    o = K.sum(o, axis=1)

        # return self.activation(o)  # 原代码
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
