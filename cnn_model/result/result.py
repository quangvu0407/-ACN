import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

def preprocess_image(img, target_size=(300, 300)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # chuẩn hóa theo EfficientNetB3
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 1. Load model đã lưu
model = load_model('cnn_model\\cnn_best_model50_1.h5')  # Đường dẫn tới file model, chỉnh lại nếu cần

# 2. Load dữ liệu test bằng generator từ thư mục chứa 6 class
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# test_generator sẽ lấy toàn bộ ảnh từ tất cả các class trong ../Garbage_classification để test
test_generator = test_datagen.flow_from_directory(
    'Garbage_classification',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# 3. Dự đoán và lấy nhãn thực tế
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# 4. Tính confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# 5. Vẽ confusion matrix và lưu file
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('cnn_model\\result\\result.png')
plt.close()

# 6. In classification report (nếu muốn)
print(classification_report(y_true, y_pred_classes))

# --- Hướng dẫn sử dụng ---
# - Đảm bảo bạn đã chuẩn bị X_test, y_test hoặc test_generator phù hợp với model.
# - Bỏ comment các đoạn code phù hợp với dữ liệu của bạn.
# - Chạy script để tạo file result.png.