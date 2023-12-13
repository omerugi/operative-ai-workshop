import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from io import BytesIO

def validate_dataset_info(train_shape, num_train_images, train_image_height, train_image_width, train_channels, 
                          test_shape, num_test_images, test_image_height, test_image_width, test_channels, 
                          num_classes, train_images, test_images,train_labels,test_labels, class_names):
    if not all([
        train_shape == train_images.shape,
        num_train_images == train_images.shape[0],
        train_image_height == train_images.shape[1],
        train_image_width == train_images.shape[2],
        train_channels == train_images.shape[3],
        test_shape == test_images.shape,
        num_test_images == test_images.shape[0],
        test_image_height == test_images.shape[1],
        test_image_width == test_images.shape[2],
        test_channels == test_images.shape[3],
        num_classes == len(class_names)
    ]):
        raise ValueError("One or more dataset dimensions do not match the actual dataset dimensions.")
    print("All dataset dimensions are as expected. Good job!")
    print(f"Training images shape: {train_shape} meaning that we have {train_shape[0]} training images, each image is of shape {train_shape[1]}x{train_shape[2]}, in {train_shape[3]} channels")
    print("Training labels shape:", train_labels.shape)
    print(f"Test images shape: {test_shape} meaning that we have {test_shape[0]} test images, each image is of shape {test_shape[1]}x{test_shape[2]}, in {test_shape[3]} channels")
    print("Test labels shape:", test_labels.shape)
    print(f"Number of classes: {num_classes} and they are: {class_names}")


def find_unique_images(train_images, train_labels, class_names):
    unique_labels = set()
    images_to_display = []
    labels_to_display = []

    for i in range(len(train_labels)):
        label = train_labels[i][0]
        if label not in unique_labels:
            unique_labels.add(label)
            images_to_display.append(train_images[i])
            labels_to_display.append(label)
        if len(unique_labels) == len(class_names):
            break
    return images_to_display, labels_to_display

def plot_sample_images(images, labels, class_names):
    plt.figure(figsize=(15, 2))
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(1, 10, i+1)
        plt.imshow(image)
        plt.title(class_names[label])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_class_distribution(labels, class_names):
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels.flatten(), order=np.arange(len(class_names)))
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.title('Distribution of Classes in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_tsne(data, labels, class_names, n_samples=5000, perplexity=10, n_iter=1000, learning_rate=200):
    # Select a subset of data and labels
    indices = np.random.choice(range(len(data)), n_samples, replace=False)
    X_subset = data[indices].reshape(n_samples, -1)

    # Optionally, scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_subset = scaler.fit_transform(X_subset)

    y_subset = labels[indices].flatten()

    # Apply t-SNE with adjusted parameters
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate, init='pca')
    X_tsne = tsne.fit_transform(X_subset)

    # Plot
    plt.figure(figsize=(12, 8))
    for i in range(len(class_names)):
        indices = y_subset == i
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=class_names[i], alpha=0.5)
    plt.legend()
    plt.title('t-SNE visualization')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()

def validate_dataset_loading(iris_dataset):
    try:
        assert iris_dataset is not None, "Dataset not loaded. Use the load_iris function."
        assert hasattr(iris_dataset, 'data') and hasattr(iris_dataset, 'target'), "Dataset seems incomplete."
        return "Dataset loaded correctly."
    except AssertionError as error:
        return error

def validate_type(train,test):
    try:
        if str(train.dtype) == 'float32' and str(test.dtype) == 'float32':
            return "Type changed succefully!"
        raise Exception(f"Wrong type: train type {str(train.dtype)} test type {str(test.dtype)}")
    except Exception as e: 
        return e


def validate_norm_rage(train_images,test_images):
    if train_images.min() == 0.0 and train_images.max() == 1.0 and test_images.min() == 0.0 and test_images.max() == 1.0:
        return "Normalization succeeded"
    raise Exception(f"Wrong values: train max/min {train_images.max()}{train_images.min()} test max/min {test_images.max()}{test_images.min()}")

def load_and_preprocess_image(url):
    response = requests.get(url)
    img = load_img(BytesIO(response.content), target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array