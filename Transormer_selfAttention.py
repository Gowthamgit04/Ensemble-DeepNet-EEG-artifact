import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, matthews_corrcoef

# -------------------------------
# Patch Embedding Layer
# -------------------------------
class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=15, embed_dim=128, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.projection = layers.Dense(self.embed_dim, name="patch_projection")

    def call(self, inputs):
        # Extract non-overlapping patches and project
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # patches shape: (batch, nH, nW, patch_dim)
        batch = tf.shape(patches)[0]
        patch_dim = tf.shape(patches)[-1]
        patches_flat = tf.reshape(patches, (batch, -1, patch_dim))  # (batch, num_patches, patch_dim)
        return self.projection(patches_flat)  # -> (batch, num_patches, embed_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return cfg


# -------------------------------
# Transformer Encoder with Cross-Attention (graph-safe)
# -------------------------------
class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, ff_dim=256, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.ff_dim = int(ff_dim)
        self.dropout = float(dropout)

        # Self-attention components
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="norm1")
        self.mha_self = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim,
                                                  dropout=self.dropout, name="mha_self")

        # Cross-attention components
        self.norm_cross_q = layers.LayerNormalization(epsilon=1e-6, name="norm_cross_q")
        self.norm_cross_kv = layers.LayerNormalization(epsilon=1e-6, name="norm_cross_kv")
        self.mha_cross = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim,
                                                   dropout=self.dropout, name="mha_cross")

        # Feed-forward
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm2")
        self.ffn_dense1 = layers.Dense(self.ff_dim, activation="relu", name="ffn_dense1")
        # output back to embedding dimension; we'll infer embed_dim during call
        self.ffn_dense2 = layers.Dense(self.key_dim * self.num_heads, name="ffn_dense2")

    def build(self, input_shape):
        # input_shape: (batch, seq_len, embed_dim) or (None, None, embed_dim)
        # Safely get embed_dim if available, else leave layers to build lazily
        if len(input_shape) >= 3 and input_shape[2] is not None:
            embed_dim = int(input_shape[2])
            # ensure last dense maps correctly
            self.ffn_dense2 = layers.Dense(embed_dim, name="ffn_dense2")
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs, context=None):
        """
        inputs: (batch, seq_len_q, embed_dim)
        context: either
           - (seq_len_ctx, embed_dim)  OR
           - (batch, seq_len_ctx, embed_dim)
        This function avoids Python-level tensor truth checks and uses only static
        shape inspection where safe.
        """
        # --- Self-attention (explicit query/key/value) ---
        x_norm = self.norm1(inputs)
        attn_self = self.mha_self(query=x_norm, value=x_norm, key=x_norm)
        x = layers.Add()([inputs, attn_self])

        # --- Cross-attention if context provided ---
        if context is not None:
            # Use static shape rank if available (safe in graph mode)
            context_rank = context.shape.rank  # returns int or None (no tensor op)
            if context_rank == 2:
                # context is (seq_len_ctx, embed_dim) -> expand & broadcast to (batch, seq_len_ctx, embed_dim)
                context_exp = tf.expand_dims(context, axis=0)  # (1, seq_len_ctx, emb)
                context_broadcast = tf.repeat(context_exp, tf.shape(x)[0], axis=0)
            else:
                # assume context already (batch, seq_len_ctx, emb)
                context_broadcast = context

            # normalize q and kv explicitly
            q = self.norm_cross_q(x)
            kv = self.norm_cross_kv(context_broadcast)

            # explicit query/key/value args to improve shape inference
            cross_attn = self.mha_cross(query=q, value=kv, key=kv)
            x = layers.Add()([x, cross_attn])

        # --- Feed-forward ---
        x_norm2 = self.norm2(x)
        ffn = self.ffn_dense2(self.ffn_dense1(x_norm2))
        return layers.Add()([x, ffn])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_heads": self.num_heads, "key_dim": self.key_dim,
            "ff_dim": self.ff_dim, "dropout": self.dropout
        })
        return cfg


# -------------------------------
# Vision Transformer Model (full)
# -------------------------------
def build_vit(input_shape=(1000, 400, 3), num_classes=5,
              patch_size=15, embed_dim=128, num_layers=1,
              num_heads=8, ff_dim=256, dropout=0.1):

    inputs = Input(shape=input_shape, name="input_image")

    # Patch embedding
    x = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim, name="patch_emb")(inputs)

    # compute number of patches
    n_h = input_shape[0] // patch_size
    n_w = input_shape[1] // patch_size
    num_patches = int(n_h * n_w)

    # positional embedding as context (shape: (num_patches, embed_dim))
    positions = tf.range(num_patches)
    pos_embedding_layer = layers.Embedding(input_dim=num_patches, output_dim=embed_dim, name="pos_embedding")
    pos_embed = pos_embedding_layer(positions)  # (num_patches, embed_dim)

    # Transformer blocks (each accepts context)
    for i in range(num_layers):
        x = TransformerEncoder(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_encoder_{i}"
        )(x, context=pos_embed)

    # classification head
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.5, name="drop1")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VisionTransformer_CrossAttn")
    return model


# -------------------------------
# Build, compile and show summary
# -------------------------------
model = build_vit(input_shape=(1000, 400, 3), num_classes=5,
                  patch_size=15, embed_dim=128, num_layers=4,
                  num_heads=8, ff_dim=256, dropout=0.1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


# -------------------------------
# Data Generators (unchanged)
# -------------------------------
base_dir = "./new_data_split5"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

train_datagen = ImageDataGenerator(rescale=1.0/255)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(1000, 400), batch_size=12, class_mode="categorical", shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(1000, 400), batch_size=12, class_mode="categorical", shuffle=False
)


# -------------------------------
# Training (unchanged)
# -------------------------------
checkpoint = ModelCheckpoint(
    "best_model_transformer_cross_safe.keras", monitor="val_accuracy", verbose=1,
    save_best_only=True, mode="max"
)

history = model.fit(
    train_generator,
    steps_per_epoch=int(np.ceil(train_generator.samples / train_generator.batch_size)),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=int(np.ceil(validation_generator.samples / validation_generator.batch_size)),
    callbacks=[checkpoint]
)
