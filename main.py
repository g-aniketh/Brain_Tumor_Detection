import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA


DATASET_LABELS = {
    'no_tumor': 0,
    'pituitary_tumor': 1,
    'meningioma_tumor': 2,
    'glioma_tumor': 3,
}
LABEL_TO_NAME = {value: key for key, value in DATASET_LABELS.items()}
IMAGE_SIZE = (200, 200)


def _load_dataset():
    images = []
    labels = []

    for class_name, class_id in DATASET_LABELS.items():
        class_path = f'./{class_name}'
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_id)

    if not images:
        raise ValueError('No valid training images were found in dataset folders.')

    return np.array(images), np.array(labels)


def _train_model():
    x, y = _load_dataset()
    x_update = x.reshape(len(x), -1)
    x_train, _, y_train, _ = train_test_split(x_update, y, random_state=10, test_size=0.3)

    x_train = x_train / 255

    pca = PCA(0.98)
    pca_train = pca.fit_transform(x_train)

    svm_model = SVC(probability=True)
    svm_model.fit(pca_train, y_train)
    return pca, svm_model


PCA_MODEL, SVM_MODEL = _train_model()


def _preprocess_uploaded_image(file):
    image_buffer = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Unable to decode uploaded image.')

    image = cv2.resize(image, IMAGE_SIZE)
    image = image.reshape(1, -1) / 255
    return image


def get_tumor_prediction(file):
    processed_image = _preprocess_uploaded_image(file)
    transformed = PCA_MODEL.transform(processed_image)

    prediction = SVM_MODEL.predict(transformed)
    probabilities = SVM_MODEL.predict_proba(transformed)[0]

    predicted_index = int(prediction[0])
    predicted_label = LABEL_TO_NAME[predicted_index]
    confidence = float(np.max(probabilities))

    return {
        'tumor_type': predicted_label,
        'confidence': confidence,
    }


def get_tumor_type(file):
    result = get_tumor_prediction(file)
    return result['tumor_type']
