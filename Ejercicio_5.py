import DnnLib as Dn
import numpy as np 
import json
import argparse
    
def crear_modelo(input_dim=784, hidden_units=128, output_units=10):
        layer1 = Dn.DenseLayer(input_dim, hidden_units, Dn.ActivationType.RELU)
        layer2 = Dn.DenseLayer(hidden_units, output_units, Dn.ActivationType.SOFTMAX)
        
        return layer1, layer2
    
def calcular_accuracy(output, labels):
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
def entrenar_modelo(layer1, layer2, X_train, y_train,epochs=30, batch_size=64, lr=0.01,lambda_reg=0.001, dropout_rate=0.5):

    layer1.set_regularizer(Dn.RegularizerType.L2, lambda_reg)
    layer2.set_regularizer(Dn.RegularizerType.L2, lambda_reg)
    dropout = Dn.Dropout(dropout_rate=dropout_rate)
    optimizer = Dn.Adam(learning_rate=lr)

    Mean_acc = []

    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            h1 = layer1.forward(X_batch)
            dropout.training = True
            h1 = dropout.forward(h1)
            output = layer2.forward(h1)
            
            loss = Dn.cross_entropy(output, y_batch)
            grad = Dn.cross_entropy_gradient(output, y_batch)
            grad = layer2.backward(grad)
            grad = dropout.backward(grad)   
            grad = layer1.backward(grad)

            optimizer.update(layer2)
            optimizer.update(layer1)

        dropout.training = False
        train_out = layer2.forward(dropout.forward(layer1.forward(X_train)))
        train_loss = Dn.cross_entropy(train_out, y_train)
        train_preds = np.argmax(train_out, axis=1)
        train_acc = np.mean(train_preds == np.argmax(y_train, axis=1))
        Mean_acc.append(train_acc)

        if epoch % 5 == 0:
            acc = np.mean(Mean_acc)
            print(f"Epoch {epoch}, "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Acc: {acc:.4f} ")
            evaluar_modelo(layer1,layer2)
            Mean_acc.clear()

    return layer1, layer2

def exportar_modelo(layer1, layer2, filename="Modelo_fashion_MNIST.json"):
    model_dict = {
        "layers": [
            {
                "type": "Dense",
                "input_dim": int(layer1.weights.shape[1]),
                "output_dim": int(layer1.weights.shape[0]),
                "activation": "ReLU",
                "weights": layer1.weights.tolist(),
                "bias": layer1.bias.tolist()
            },
            {
                "type": "Dense",
                "input_dim": int(layer2.weights.shape[1]),
                "output_dim": int(layer2.weights.shape[0]),
                "activation": "Softmax",
                "weights": layer2.weights.tolist(),
                "bias": layer2.bias.tolist()
            }
        ]
    }

    with open(filename, "w") as f:
        json.dump(model_dict, f, indent=4)
    print(f"Modelo exportado en {filename}")


def cargar_modelo(filename="Modelo_fashion_MNIST.json"):
    with open(filename, "r") as f:
        model_dict = json.load(f)

    layer1_info = model_dict["layers"][0]
    layer2_info = model_dict["layers"][1]

    act_map = {
        "ReLU": Dn.ActivationType.RELU,
        "Sigmoid": Dn.ActivationType.SIGMOID,
        "Tanh": Dn.ActivationType.TANH,
        "Softmax": Dn.ActivationType.SOFTMAX
    }

    layer1 = Dn.DenseLayer(layer1_info["input_dim"], layer1_info["output_dim"],
                           act_map.get(layer1_info["activation"], Dn.ActivationType.RELU))
    layer1.weights = np.array(layer1_info["weights"], dtype=np.float64)
    layer1.bias = np.array(layer1_info["bias"], dtype=np.float64)

    layer2 = Dn.DenseLayer(layer2_info["input_dim"], layer2_info["output_dim"],
                           act_map.get(layer2_info["activation"], Dn.ActivationType.SOFTMAX))
    layer2.weights = np.array(layer2_info["weights"], dtype=np.float64)
    layer2.bias = np.array(layer2_info["bias"], dtype=np.float64)

    print(f"Modelo cargado desde {filename}")
    return layer1, layer2


    
def evaluar_modelo(layer1, layer2):
    Eval = np.load("fashion_mnist_test.npz")
    images_eval = Eval["images"]
    labels_eval = Eval["labels"]
    datos_eval = (images_eval / 255.0).reshape(images_eval.shape[0], -1)
    h1 = layer1.forward(datos_eval)
    output = layer2.forward(h1)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions== labels_eval)
    print("Precision del modelo Testeada: ",accuracy)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Entrenar o cargar un modelo MNIST con DnnLib")
        parser.add_argument("--epochs", type=int, default=30, help="Número de épocas de entrenamiento")
        parser.add_argument("--batch_size", type=int, default=64, help="Tamaño de batch")
        parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
        parser.add_argument("--hidden_units", type=int, default=128, help="Unidades de la capa oculta")
        parser.add_argument("--train", type=str, default="mnist_train.npz", help="Archivo de entrenamiento")
        parser.add_argument("--output", type=str, default="Modelo_MNIST.json", help="Archivo para exportar el modelo")
        parser.add_argument("--load", type=str, help="Ruta a un modelo JSON previamente guardado")
        args = parser.parse_args()

        if args.load:
            layer1, layer2 = cargar_modelo(args.load)
            evaluar_modelo(layer1, layer2)
        else:
            data = np.load(args.train)
            images = data["images"]
            labels = data["labels"]
            datos = (images / 255.0).reshape(images.shape[0], -1)
            y_train_onehot = np.eye(10)[labels]

            layer1, layer2 = crear_modelo(input_dim=datos.shape[1],hidden_units=args.hidden_units,output_units=10)
            layer1, layer2 = entrenar_modelo(layer1, layer2, datos, y_train_onehot,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr)
            exportar_modelo(layer1, layer2, args.output)
            evaluar_modelo(layer1, layer2, args.test)