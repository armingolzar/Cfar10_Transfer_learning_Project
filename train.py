import os
import matplotlib.pyplot as plt
from src.data_loader import data_preparing
from src.model import build_network

# Configuration
EPOCHS = 50
BATCH_SIZE = 32
ASSETS_DIR = "assets"
MODELS_DIR = "models"

def plot_training_history(history, save_dir=ASSETS_DIR):
    """Plot and save training accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    
    plt.close()

def main():
    print("🚀 Loading and preparing CIFAR-10 data (64x64)...")
    x_train, y_train, x_test, y_test = data_preparing(target_size=64)
    print(f"✅ Data loaded: {x_train.shape[0]} training, {x_test.shape[0]} test samples")

    print("🧠 Building model with VGG16 (frozen up to block3_pool) + augmentation...")
    model = build_network(input_shape=(64, 64, 3), num_classes=10)
    model.summary()

    print(f"🏋️ Starting training for {EPOCHS} epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Saving model
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("\n 🚀 Start saving model...")
    model.save(os.path.join(MODELS_DIR, 'cifar10_transfer_learning_model.h5'))
    print(f"\n ✅ Saving complete! Model saved to '{MODELS_DIR}/cifar10_transfer_learning_model.h5'.")

    # Evaluate final accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"\n🎯 Final Test Loss: {test_loss:.4f}")

    print("📊 Saving training curves to assets/...")
    plot_training_history(history)

    print("✅ Training complete! Plots saved in 'assets/' folder.")

if __name__ == "__main__":
    main()