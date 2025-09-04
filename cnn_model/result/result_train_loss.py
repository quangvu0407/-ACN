import matplotlib.pyplot as plt
import pickle

# Đường dẫn tới file history.pkl đã lưu khi train
history_path = "cnn_model\\result\\history.pkl"

# Load history
with open(history_path, 'rb') as f:
    history = pickle.load(f)

# Lấy dữ liệu
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(1, len(acc) + 1)

# Vẽ Accuracy
plt.figure(figsize=(8, 6))
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('cnn_model\\result\\accuracy.png')
plt.close()

# Vẽ Loss
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('cnn_model\\result\\loss.png')
plt.close()

print("✅ Đã lưu ảnh accuracy.png và loss.png")
