import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os
import pickle  

def build_efficientnetb3_model(input_shape=(300, 300, 3), num_classes=6):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    return model

def train_model(train_ds, val_ds, input_shape=(300, 300, 3), num_classes=6, epochs=10, lr=1e-3):
    model = build_efficientnetb3_model(input_shape, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return model, history

if __name__ == "__main__":
    data_dir = "F:\\k\\DACN\\CNN_EfficientNetB3\\Garbage_classification"  # sửa lại đường dẫn cho đúng
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục: {data_dir}")

    img_size = (300, 300)
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int"
    )

    model, history = train_model(train_ds, val_ds, epochs=50)
    print("Train xong, kiểm tra model summary:")
    model.summary()

    # Lưu mô hình
    model_save_path = "cnn_model\\cnn_best_model50_1.h5"
    model.save(model_save_path)
    print(f"Đã lưu mô hình tại: {model_save_path}")

    # Lưu history
    history_save_path = "cnn_model\\result\\history.pkl"
    with open(history_save_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"Đã lưu history tại: {history_save_path}")
