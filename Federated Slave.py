import socket
import pickle
import struct
import tensorflow as tf
import numpy as np

# Function to load and preprocess MNIST data
def load_mnist_data():
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0  # Normalize pixel values
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    return x_train, y_train

# Function to create the MNIST model
def create_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Helper function to send data with a size header
def send_data(conn, data):
    serialized_data = pickle.dumps(data)
    data_size = len(serialized_data)
    conn.sendall(struct.pack("!I", data_size))  # Send size of data first
    conn.sendall(serialized_data)  # Send actual serialized data

# Helper function to receive data with a size header
def receive_data(conn):
    raw_size = conn.recv(4)
    if not raw_size:
        raise ConnectionError("Connection closed by server (no size received)")
   
    data_size = struct.unpack("!I", raw_size)[0]
    data = b""
    while len(data) < data_size:
        packet = conn.recv(4096)
        if not packet:
            raise ConnectionError(f"Connection closed while receiving data: received {len(data)} of {data_size} bytes")
        data += packet

    return pickle.loads(data)

# Custom callback to send model weights after each epoch
class FederatedCallback(tf.keras.callbacks.Callback):
    def __init__(self, local_model):
        super(FederatedCallback, self).__init__()
        self.local_model = local_model  # Renamed to avoid conflict with `model` property

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} completed. Sending local weights to server...\n")

        # Get local model weights
        local_weights = self.local_model.get_weights()

        # Ensure all weights are NumPy arrays before sending
        if not all(isinstance(layer, np.ndarray) for layer in local_weights):
            raise ValueError("Local model weights must be a list of NumPy arrays before sending.")

        # Send weights to server
        data_to_send = {"dataset": "mnist", "weights": local_weights}
       
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_sock:
            client_sock.connect(('127.0.0.1', 5000))  # Connect to the master server
            send_data(client_sock, data_to_send)

            # Receive updated global weights
            response = receive_data(client_sock)
            if "weights" not in response:
                raise ValueError("Server response did not contain updated weights.")

            global_weights = response["weights"]

            print("Received updated global weights. Updating local model...\n")

            # Update local model with global weights
            self.local_model.set_weights(global_weights)

def mnist_client(local_epochs=5):
    x_train, y_train = load_mnist_data()
    model = create_mnist_model()

    # Train the model with the custom federated callback
    model.fit(x_train, y_train, epochs=local_epochs, batch_size=64, verbose=1, callbacks=[FederatedCallback(model)])

if __name__ == "__main__":
    local_epochs = int(input("Enter number of local epochs for MNIST: "))
    mnist_client(local_epochs=local_epochs)

On Sat, Mar 15, 2025 at 11:34 AM Varsh Gandhi <varshgandhi.ai23@bmsce.ac.in> wrote:
import socket
import pickle
import struct
import tensorflow as tf
import numpy as np

# Function to load and preprocess MNIST data
def load_mnist_data():
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0  # Normalize pixel values
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    return x_train, y_train

# Function to create the MNIST model
def create_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Helper function to send data with a size header
def send_data(conn, data):
    serialized_data = pickle.dumps(data)
    data_size = len(serialized_data)
    conn.sendall(struct.pack("!I", data_size))  # Send size of data first
    conn.sendall(serialized_data)  # Send actual serialized data

# Helper function to receive data with a size header
def receive_data(conn):
    raw_size = conn.recv(4)
    if not raw_size:
        raise ConnectionError("Connection closed by server (no size received)")
   
    data_size = struct.unpack("!I", raw_size)[0]
    data = b""
    while len(data) < data_size:
        packet = conn.recv(4096)
        if not packet:
            raise ConnectionError(f"Connection closed while receiving data: received {len(data)} of {data_size} bytes")
        data += packet

    return pickle.loads(data)

# Custom callback to send model weights after each epoch
class FederatedCallback(tf.keras.callbacks.Callback):
    def __init__(self, local_model):
        super(FederatedCallback, self).__init__()
        self.local_model = local_model  # Renamed to avoid conflict with `model` property

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} completed. Sending local weights to server...\n")

        # Get local model weights
        local_weights = self.local_model.get_weights()

        # Ensure all weights are NumPy arrays before sending
        if not all(isinstance(layer, np.ndarray) for layer in local_weights):
            raise ValueError("Local model weights must be a list of NumPy arrays before sending.")

        # Send weights to server
        data_to_send = {"dataset": "mnist", "weights": local_weights}
       
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_sock:
            client_sock.connect(('127.0.0.1', 5000))  # Connect to the master server
            send_data(client_sock, data_to_send)

            # Receive updated global weights
            response = receive_data(client_sock)
            if "weights" not in response:
                raise ValueError("Server response did not contain updated weights.")

            global_weights = response["weights"]

            print("Received updated global weights. Updating local model...\n")

            # Update local model with global weights
            self.local_model.set_weights(global_weights)

def mnist_client(local_epochs=5):
    x_train, y_train = load_mnist_data()
    model = create_mnist_model()

    # Train the model with the custom federated callback
    model.fit(x_train, y_train, epochs=local_epochs, batch_size=64, verbose=1, callbacks=[FederatedCallback(model)])

if __name__ == "__main__":
    local_epochs = int(input("Enter number of local epochs for MNIST: "))
    mnist_client(local_epochs=local_epochs)
