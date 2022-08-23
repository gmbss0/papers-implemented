#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import MultiHeadAttention, Dropout, Embedding, LayerNormalization
import numpy as np


# %%
# ResNet like image-encoder
def build_ResNet(input_shape, kernel_size=3, blocks=[32, 64, 128], z_dim=128):
    def residual_block(z, filters, residual=False):
        # batch norm -> conv 
        z_res = BatchNormalization()(Conv2D(filters, kernel_size, padding="same",
                                          activation="relu", kernel_initializer="he_uniform")(z))
        # batch norm -> conv 
        z_res = BatchNormalization()(Conv2D(filters, kernel_size, padding="same",
                                          activation="relu", kernel_initializer="he_uniform")(z_res))
        if residual:
            z_res = z + z_res
        return z_res
    def conv_block(z, filters, n_residual_blocks=3):
        for n in range(n_residual_blocks):
            if n == 0:
                z = residual_block(z, filters)
            else:
                z = residual_block(z, filters, residual=True)
        return z
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    z = inputs
    # loop over blocks defined in list with increasing amount of filters
    for filters in blocks:
        z = conv_block(z, filters)
    # flatten and compute representation
    avg_pooled = GlobalAveragePooling2D()(z)
    output = Dense(z_dim)(avg_pooled)
    return tf.keras.models.Model(inputs=inputs, outputs=output)


# %%
# expander
def build_expander(embedding_dim=256, expander_layers=3):
    return tf.keras.Sequential([Dense(embedding_dim, activation="relu", kernel_initializer="he_uniform") for num in range(expander_layers)])


# %%
# Transformer
class MLP(tf.keras.layers.Layer):
    def __init__(self, output_dim, internal_dim):
        super(MLP, self).__init__()
        self.non_linear_dense = Dense(internal_dim, activation="relu", kernel_initializer="he_uniform")
        self.dense = Dense(output_dim)
    def call(self, x):
        out = self.non_linear_dense(x)
        return self.dense(out)
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = MLP(d_model, mlp_dim)
        self.att_layernorm = LayerNormalization(epsilon=1e-6)
        self.ffn_layernorm = LayerNormalization(epsilon=1e-6)
        self.att_dropout = Dropout(dropout_rate)
        self.ffn_dropout = Dropout(dropout_rate)

    def call(self, x, training):
        bs = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        # attention block
        mask = tf.ones([bs, seq_len, seq_len], dtype=tf.float32) # self-attention attends to entire sequence
        att_out = self.mha(x, x, x, attention_mask=mask)
        att_out = self.att_dropout(att_out, training=training)
        att_out = self.att_layernorm(x + att_out)
        # dense block
        ffn_out = self.ffn(att_out) 
        ffn_out = self.ffn_dropout(ffn_out, training=training)
        ffn_out = self.ffn_layernorm(att_out + ffn_out)
        return ffn_out

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def PositionalEncoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
    
class TextEncoder(tf.keras.Model):
    def __init__(self, n_layers, seq_len, vocab_size, d_model, num_heads, mlp_dim, dropout_rate):
        super(TextEncoder, self).__init__()
        self.d_model = d_model
        self.emb = Embedding(vocab_size, d_model)  # Only use if input provided as integers
        self.pos_emb = PositionalEncoding(seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, mlp_dim, dropout_rate=dropout_rate) for _ in range(n_layers)]
    def call(self, x):
        # embed tokens
        input_emb = self.emb(x)
        input_emb *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # positional encoding
        pos_emb = self.pos_emb
        text_emb = input_emb + pos_emb
        # transformer layers
        for layer in self.enc_layers:
            text_emb = layer(text_emb)
        # mean pooling 
        text_emb = tf.math.reduce_mean(text_emb, axis=1)
        return text_emb


# %%
# VICreg losses
def V_loss(Z1, Z2, target_std=1., epsilon=1e-6):
    def compute_v_loss(Z):
        std = tf.math.sqrt(tf.math.reduce_variance(Z, axis=0) + epsilon)  # compute std along batch dimension
        std = tf.nn.relu(target_std - std)  # hinge loss
        return tf.math.reduce_mean(std)
    return compute_v_loss(Z2) + compute_v_loss(Z2)

def I_loss(Z1, Z2):
    return tf.math.reduce_mean((Z1 - Z2) ** 2)

def C_loss(Z1, Z2):
    def compute_c_loss(Z):
        bs = tf.cast(tf.shape(Z)[0], dtype=tf.float32)
        z_dim = tf.cast(tf.shape(Z)[1], dtype=tf.float32)
        Z_ = Z - tf.math.reduce_mean(Z, axis=0)  # center representation
        # compute covariance matrix
        cov = tf.linalg.matmul(Z_, Z_, transpose_a=True) / (bs - tf.cast(1, dtype=tf.float32))  # normalise by N - 1
        # set diagonal elements to 0 as they should not contribute to loss 
        cov = tf.linalg.set_diag(cov, tf.zeros(shape=[z_dim]))
        return tf.math.reduce_sum(cov ** 2) / z_dim
    return compute_c_loss(Z1) + compute_c_loss(Z2)


# %%
# VICReg
class VICReg(tf.keras.Model):
    def __init__(self, img_encoder, txt_encoder, img_expander, txt_expander, params):
        super(VICReg, self).__init__()
        self.img_encoder = img_encoder
        self.img_expander = img_expander
        self.txt_encoder = txt_encoder
        self.txt_expander = txt_expander
        
        self.var_loss_weight = params["V_loss_weight"]
        self.inv_loss_weight = params["I_loss_weight"]
        self.cov_loss_weight = params["C_loss_weight"]
        
    def compile(self, optimizer, variance_loss, invariance_loss, covariance_loss):
        super(VICReg, self).compile()
        self.optimizer = optimizer
        self.var_loss = variance_loss
        self.inv_loss = invariance_loss
        self.cov_loss = covariance_loss
        
    def train_step(self, batch):
        image, text = batch
        with tf.GradientTape() as tape:
            # encode data
            Y_img = self.img_encoder(image)
            Y_txt = self.txt_encoder(text)
            # expand representations
            Z_img = self.img_expander(Y_img)
            Z_txt = self.txt_expander(Y_txt)
            # compute losses
            var_loss = self.var_loss(Z_img, Z_txt) # V
            inv_loss = self.inv_loss(Z_img, Z_txt) # I
            cov_loss = self.cov_loss(Z_img, Z_txt) # C
            loss = self.var_loss_weight * var_loss + self.inv_loss_weight * inv_loss + self.cov_loss_weight * cov_loss
        # get gradients and perform optimizer step
        img_vars = self.img_encoder.trainable_variables + self.img_expander.trainable_variables
        txt_vars = self.txt_encoder.trainable_variables + self.txt_expander.trainable_variables
        grads = tape.gradient(loss, img_vars + txt_vars)
        self.optimizer.apply_gradients(zip(grads, img_vars + txt_vars))
        return {"loss" : loss, "variance_loss" : var_loss,
                "invariance_loss" : inv_loss, "covariance_loss" : cov_loss}
    def call(self, batch):
        image, text = batch
        # encode data
        Y_img = self.img_encoder(image)
        Y_txt = self.txt_encoder(text)
        # expand representations
        Z_img = self.img_expander(Y_img)
        Z_txt = self.txt_expander(Y_txt)
        return Z_img, Z_txt
