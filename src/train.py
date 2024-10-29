import argparse
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='MLP Model Training')
    parser.add_argument('--layer', type=int, nargs='+', help='Layer sizes', default=[64, 64, 64])
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--loss', type=str, help='Loss function', default='binary_crossentropy')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("Number of epochs must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be a positive float.")
    
    if len(args.layer) != 3:
        raise ValueError("You must provide exactly three layer sizes for hidden layers.")

    return args


def preprocess_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def main():
    """Main function to parse arguments and initiate model training."""
    args = parse_args()
    logging.info(f'Parsed arguments: {args}')

    # Load training data
    df_train = pd.read_csv('../data/processed/training.csv')
    logging.info(f'Read training data: {df_train.shape}')

    X_train = df_train.drop('M', axis=1)
    y_train = df_train['M']

    # Load validation data
    df_val = pd.read_csv('../data/processed/validation.csv')  # Adjust the path as needed
    logging.info(f'Read validation data: {df_val.shape}')

    X_val = df_val.drop('M', axis=1)
    y_val = df_val['M']

    # Preprocess training and validation data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)

    layer1, layer2, layer3 = args.layer
    loss = args.loss
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size

    model = models.Sequential([
        layers.Dense(32, activation='sigmoid', input_shape=(X_train.shape[1],)), 
        layers.Dense(layer1, activation='sigmoid', kernel_initializer='he_uniform'),
        layers.Dense(layer2, activation='sigmoid', kernel_initializer='he_uniform'),
        layers.Dense(layer3, activation='sigmoid', kernel_initializer='he_uniform'),
        layers.Dense(1, activation='sigmoid', kernel_initializer='he_uniform')  # Change to sigmoid for binary classification
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
    
    # Add model summary
    model.summary()

    # Train the model with validation data
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Print final training and validation accuracy and loss
    final_train_loss, final_train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    final_val_loss, final_val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    logging.info(f'Final Training Loss: {final_train_loss:.4f}, Final Training Accuracy: {final_train_accuracy:.4f}')
    logging.info(f'Final Validation Loss: {final_val_loss:.4f}, Final Validation Accuracy: {final_val_accuracy:.4f}')

    plot_training_history(history)

if __name__ == '__main__':
    main()
