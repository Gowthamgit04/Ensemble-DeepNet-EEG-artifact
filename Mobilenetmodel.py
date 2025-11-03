import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, matthews_corrcoef
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, Input
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
base_dir = './new_data_split5'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Define Squeeze-and-Excite block
def squeeze_excite_block(input_tensor, ratio=16):
    """Squeeze-and-Excitation block implementation."""
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

    x = layers.Multiply()([init, se])
    return x

# Define Depthwise Separable Convolution with SE block
def depthwise_separable_conv(input_tensor, filters, strides, use_se=True):
    """Depthwise Separable Convolution with optional SE block."""
    # Depthwise Convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pointwise Convolution
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Add SE block if required
    if use_se:
        x = squeeze_excite_block(x)
    return x

# Build MobileNet from scratch with SE blocks
def build_mobilenet_with_se(input_shape=(1000, 400, 3), num_classes=5):
    """Build MobileNet from scratch with Squeeze-and-Excitation blocks."""
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise Separable Convolution Layers
    x = depthwise_separable_conv(x, filters=64, strides=(1, 1), use_se=True)
    x = depthwise_separable_conv(x, filters=128, strides=(2, 2), use_se=True)
    x = depthwise_separable_conv(x, filters=128, strides=(1, 1), use_se=True)
    x = depthwise_separable_conv(x, filters=256, strides=(2, 2), use_se=True)
    x = depthwise_separable_conv(x, filters=256, strides=(1, 1), use_se=True)
    x = depthwise_separable_conv(x, filters=512, strides=(2, 2), use_se=True)

    # Global Average Pooling and Dense Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs, outputs)
    return model

# Build the model
model = build_mobilenet_with_se(input_shape=(1000, 400, 3), num_classes=5)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(1000, 400),  # Input size for MobileNet
    batch_size=20,
    class_mode='categorical',
    shuffle=True  # Shuffle to improve training performance
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(1000, 400),
    batch_size=20,
    class_mode='categorical',
    shuffle=False
)

# Model checkpoint callback
checkpoint = ModelCheckpoint(
    'best_model_mobilenet_se_scratch_s5.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=int(np.ceil(train_generator.samples / train_generator.batch_size)),
    epochs=150,
    validation_data=validation_generator,
    validation_steps=int(np.ceil(validation_generator.samples / validation_generator.batch_size)),
    callbacks=[checkpoint]
)

# Load the best model after training
best_model = models.load_model('best_model_mobilenet_se_scratch_s5.keras')

# Evaluation function
def evaluate_model(model, generator, dataset_name):
    y_true = generator.classes
    y_pred_prob = model.predict(generator, steps=int(np.ceil(generator.samples / generator.batch_size)))
    y_pred = np.argmax(y_pred_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average='macro')
    specificity = recall_score(y_true, y_pred, average='macro', pos_label=0)
    fscore = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)

    csi = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))
    csi = np.mean(csi)

    # Save metrics to CSV
    metrics_dict = {
        'Accuracy': [accuracy],
        'Sensitivity': [sensitivity],
        'Specificity': [specificity],
        'F1 Score': [fscore],
        'MCC': [mcc],
        'CSI': [csi]
    }
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_filename = f'evaluation_metrics_{dataset_name}_mobilenet_se_scratch_s5.csv'
    metrics_df.to_csv(metrics_filename, index=False)

    # Save the confusion matrix to CSV
    cm_df = pd.DataFrame(cm, index=[f"Actual_{i}" for i in range(cm.shape[0])],
                         columns=[f"Predicted_{i}" for i in range(cm.shape[1])])
    cm_filename = f'confusion_matrix_{dataset_name}_mobilenet_se_scratch_s5.csv'
    cm_df.to_csv(cm_filename)

    # Save predicted probabilities and true labels to CSV
    probabilities_df = pd.DataFrame(y_pred_prob, columns=[f"Class_{i}_Probability" for i in range(y_pred_prob.shape[1])])
    probabilities_df['True_Label'] = y_true
    probabilities_filename = f'{dataset_name}_probabilities_mobilenet_se_scratch_s5.csv'
    probabilities_df.to_csv(probabilities_filename, index=False)

    print(f"{dataset_name.capitalize()} Metrics saved to '{metrics_filename}', confusion matrix saved to '{cm_filename}', and probabilities saved to '{probabilities_filename}'.")

    return cm, accuracy, sensitivity, specificity, fscore, mcc, csi

# Evaluate on train and validation data
print("Evaluating on Training Data:")
evaluate_model(best_model, train_generator, 'train')

print("Evaluating on Validation Data:")
evaluate_model(best_model, validation_generator, 'validation')