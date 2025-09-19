import DnnLib as Dn
import numpy as np 
import json
    
def crear_modelo(input_dim=784, hidden_units=128, output_units=10):
        layer1 = Dn.DenseLayer(input_dim, hidden_units, Dn.ActivationType.RELU)
        layer2 = Dn.DenseLayer(hidden_units, output_units, Dn.ActivationType.SOFTMAX)
        
        return layer1, layer2
    
def calcular_accuracy(output, labels):
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy
    
def entrenar_modelo(layer1, layer2, X_train, y_train, epochs=30, batch_size=64, lr=0.01):
        optimizer = Dn.Adam(learning_rate=lr)
        Mean_acc = []
        for epoch in range(epochs):
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                h1 = layer1.forward(X_batch)
                output = layer2.forward(h1)
                loss = Dn.cross_entropy(output, y_batch)
                loss_grad = Dn.cross_entropy_gradient(output,y_batch)
                grad2 = layer2.backward(loss_grad)
                grad1 = layer1.backward(grad2)
                optimizer.update(layer2)
                optimizer.update(layer1)
                
            train_out = layer2.forward(layer1.forward(X_train))
            train_loss = Dn.cross_entropy(train_out, y_train)
            train_preds = np.argmax(train_out, axis=1)
            train_acc = np.mean(train_preds == np.argmax(y_train, axis=1))
            Mean_acc.append(train_acc)
            if epoch % 5 == 0:
                acc = np.mean(Mean_acc)
                print(f"Epoch {epoch}, "
                       f"Train Loss: {train_loss:.6f},  Acc: {acc:.4f}, ")
                Mean_acc.clear()
        return layer1, layer2

    
    
def exportar_modelo(layer1, layer2, filename="Modelo_MNIST.json"):
        model_dict = {
            "layers": [
                {
                    "type": "Dense",
                    "input_dim": layer1.weights.shape[1],
                    "output_dim": layer1.weights.shape[0],
                    "activation": "ReLU",
                    "weights": layer1.weights.tolist(),
                    "bias": layer1.bias.tolist()
                },
                {
                    "type": "Dense",
                    "input_dim": layer2.weights.shape[1],
                    "output_dim": layer2.weights.shape[0],
                    "activation": "Softmax",
                    "weights": layer2.weights.tolist(),
                    "bias": layer2.bias.tolist()
                }
            ]
        }
        with open(filename, "w") as f:
            json.dump(model_dict, f, indent=4)
        print(f"Modelo exportado")
    
def cargar_modelo(filename="modelo.json"):
        with open(filename, "r") as f:
            model_dict = json.load(f)
    
        layer1_info = model_dict["layers"][0]
        layer2_info = model_dict["layers"][1]
    
        layer1 = Dn.DenseLayer(layer1_info["input_dim"], layer1_info["output_dim"], Dn.ActivationType.RELU)
        layer1.weights = np.array(layer1_info["weights"])
        layer1.bias = np.array(layer1_info["bias"])
        
        layer2 = Dn.DenseLayer(layer2_info["input_dim"], layer2_info["output_dim"], Dn.ActivationType.SOFTMAX)
        layer2.weights = np.array(layer2_info["weights"])
        layer2.bias = np.array(layer2_info["bias"])
    
        print(f"Modelo cargado")
        return layer1, layer2

    
def evaluar_modelo(layer1, layer2,):
    Eval = np.load("mnist_test.npz")
    images_eval = Eval["images"]
    labels_eval = Eval["labels"]
    h1 = layer1.forward(datos_eval)
    output = layer2.forward(h1)
    predictions = np.argmax(output, axis=1)
    accuracy = np.mean(predictions== labels_eval)
    print("Precision del modelo Testeada: ",accuracy)

if __name__ == "__main__":
    data = np.load("mnist_train.npz")
    images = data["images"] 
    labels = data["labels"]
    datos = (images / 255.0).reshape(images.shape[0], -1)
    y_train_onehot = np.eye(10)[labels]

    layer1, layer2 = crear_modelo()
    layer1, layer2 = entrenar_modelo(layer1, layer2, datos, y_train_onehot)
    exportar_modelo(layer1, layer2)
    