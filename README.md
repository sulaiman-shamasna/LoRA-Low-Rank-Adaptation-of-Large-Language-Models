# Low Rank Adaptation - LoRA
---

**L**ow-**r**ank **a**daptation(LoRA) is a machine learning technique that modifies a pretrained model(e.g., an LLM) to better suit a specific, often smaller, dataset by adjusting only a small, low-rank subset of the model's parameters.

This approach is important because it allows for efficient finetuning of large models on task-specific data significantly reducing the computational cost and time required for finetuning.

Since LLMs are large, updating all model weights during training can be expensive due to GPU memory limitations. Suppoese we have a large weight matrix *W* for a given layer During backpropagation, we learn a *ΔW* matrix, which contains information on how much we want to update the original weights to minimize the loss function during training.

In regular training and finetuning, the weight update is defined as follows:

$$
W_{updated} = W + ΔW
$$

This [paper](https://arxiv.org/abs/2106.09685), where *LoRA* is proposed offers a more efficient alternative to computing the weight updates *ΔW*
by learning an approximation of it,  
*ΔW ≈ AB*. In other words, in LoRA, we have the following, where *A* and *B* are two small weight matrices:
$$
W_{updated} = W + A.B
$$
The figure below illustrates these formulas for full finetuning and LoRA side by side.

<!-- ![LoRA](images/LoRA.png) -->
### How does LoRA save GPU memory

For example, if a pretrained weight matrix W is a *1000 X 1000* matrix. Then the weight update matrix ΔW in regular finetuning is a *1000 X 1000* matrix as well. In this case, Delta W has *1,000,000* parameters. In case *LoRA* is cosidered, and a rank of *2* is used, then *A* is a *1000 X 2* matrix, and *B* is a *2 X 1000* matrix. Hence, the result will be *2 X 2 X 1000 = 4000* parameters, which to be updated when *LoRA* is used. This implies, that we've got *250 times* fewer parameters.

Of course, *A* and *B* can't capture all the information that *ΔW* could capture, but this is by design. When using LoRA, we hypothesize that the model requires *W* to be a large matrix with full rank to capture all the knowledge in the pretraining dataset. However, when we finetune an LLM, we don't need to update all the weights and capture the core information for the adaptation in a smaller number of weights than *ΔW* would; hence, we have the low-rank updates via *AB*.

If we paid close attention, the full finetuning and LoRA depictions in the figure above look slightly different from the formulas I have shown earlier. That's due to the distributive law of matrix multiplication: we don't have to add the weights with the updated weights but can keep them separate. For instance, if *x* is the input data, then we can write the following for regular finetuning:

$$
x.(W + Delta W) = x.W + x.Delta W
$$

Similarly, we can write the following for *LoRA*:
$$
x.(W + A.B) = x.W + x.A.B
$$

The fact that we can keep the *LoRA* weight matrices separate makes *LoRA* especially attractive. In practice, this means that we don't have to modify the weights of the pretrained model at all, as we can apply the *LoRA* matrices on the fly. This is especially useful if you are considering hosting a model for multiple customers. Instead of having to save the large updated models for each customer, you only have to save the small set of *LoRA* weights alongside the original pretrained model.

### LoRA: A step-by-step implementation
We begin by initializing the ```LoRALayer``` that creates the matrices *A* and *B*, along with the alpha scaling hyperparameter and the rank hyperparameters. This layer can accept an input and compute the corresponding output, as illustrated in the figure below (The *LoRA* matrices *A* and *B* with rank *r*).

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
...
```

In this code above, *rank* is the hyperparameter that controls the inner dimension of the matrices *A* and *B*. In other words, this parameter controls the number of additional parameters introduced by *LoRA* and is a key factor in determining the balance between model adaptability and parameter efficiency.

The second hyperparameter, *alpha* is a scaling hyperparameter applied to the output of the low-rank adaptation. It essentially controls the extent to which the adapted layer's output is allowed to influence the original output of the layer being adapted. This can be seen as a way to regulate the impact of the low-rank adaptation on the layer's output.

So far, the ```LoRALayer``` class we implemented above allows us to transform the layer inputs *x*. However, in LoRA, we are usually interested in replacing existing ```Linear``` layers so that the weight update is applied to the existing pretrained weights, as shown in the figure below (LoRA applied to an existing linear layer):

IMAGE

To incorporate the original Linear layer weights as shown in the figure above, we will implement a ```LinearWithLoRA``` layer that uses the previously implemented ```LoRALayer``` and can be used to replace existing ```Linear``` layers in a neural network, for example; self-attention module or feed forward modules in an LLM:

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
...
```

Note that since we initialize the weight matrix *B* (```self.b``` in ```LoRALayer```) with zero values in the *LoRA* layer, the matrix multiplication between *A* and *B* results in a matrix consisting of *0's* and doesn't affect the original weights (since adding *0* to the original weights does not modify them).

### Applying LoRA to Linear Layer
Let's apply *LoRA* to the Linear layer, we see that the results are the same since we haven't trained the *LoRA* weights yet. In other words, everything works as expected:

```python
layer_lora_1=LinearWithLoRA(layer, rank=2, alpha=4)
print(layer_lora_1(x))
"""
tensor([[0.6639, 0.4487]], grad_fn=<AddBackward0>)
"""
```

### Merging *LoRA* matrices and Original Weights
As we mentioned above, the distributive law of matrix multiplication
$$
x(W + A.B) = x.W + x.A.B
$$
This means that we can also combine or merge the *LoRA* matrices and original weights, which should result in an equivalent implementation. In code, this alternative implementation to the LinearWithLoRA layer looks as follows:

```python
class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
...
```

```LinearWithLoRAMerged``` calculates the left side of the equation above, whereas ```LinearWithLoRA``` calculates the right side.

### Applying *LoRA* Layers to *LLM*
Using **PyTorch** enables us to easily replace a linear layer in an existing neural network (e.g., the feed forward or attention modules of a *LLM*) with our new ```LinearWithLoRA``` and ```LinearWithLoRAMerged```.

#### Multilayer Perceptron Model (without *LoRA*)
For simplicity, let's focus on a small 3-layer multilayer perception instead of an *LLM* for now, which is illustrated in the figure below:
<!-- ![Multilayer_perceptron](Multilayer_perceptron.png) -->
```python
class MultilayerPerceptron(nn.Module):
    def __init__(self, num_features, num_hidden_1, num_hidden_2, num_classes):
        super().__init__()
...
```

#### Defining Hyperparameters
```python
num_features=784
num_hidden_1=128
num_hidden_2=256
num_classes=10

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate=0.005
num_epochs=10
```

#### Loading dataset
```python
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

BATCH_SIZE=64

# Note: transforms.ToTensor() scales input images to 0-1 range
train_dataset=datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset=datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader=DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader=DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
```

#### Implementing Evaluation Metric(s)
```python
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples=0,0
    with torch.no_grad():
        for features, targets in data_loader:
            features=features.view(-1, 28*28).to(device)
            targets=targets.to(device)
            logits=model(features)
            _, predicted_labels=torch.max(logits,1)
            num_examples+=targets.size(0)
            correct_pred+=(predicted_labels==targets).sum()
        return correct_pred.float()/num_examples*100
```

#### Model Training
```python
import time

def train(num_epochs, model, optimizer, train_loader, device):
    start_time=time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features=features.view(-1, 28*28).to(device)
            targets=targets.to(device)
            
            # forward and back propagation
            logits=model(features)
            loss=F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            loss.backward()
            
            # update model parameters
            optimizer.step()
...
```

### Replacing *Linear* with *LoRA* Layer

Using ```LinearWithLoRA```, we can then add the *LoRA* layers by replacing the original ```Linear``` layers in the multilayer perception model:
```python
import copy

model_lora=copy.deepcopy(model)

model_lora.layers[0]=LinearWithLoRAMerged(model_lora.layers[0], rank=4, alpha=8)
model_lora.layers[2]=LinearWithLoRAMerged(model_lora.layers[2], rank=4, alpha=8)
model_lora.layers[4]=LinearWithLoRAMerged(model_lora.layers[4], rank=4, alpha=8)
model_lora.to(DEVICE)
optimizer_lora=torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
print(model_lora)

"""
MultilayerPerceptron(
  (layers): Sequential(
    (0): LinearWithLoRAMerged(
      (linear): Linear(in_features=784, out_features=128, bias=True)
      (lora): LoRALayer()
    )
    (1): ReLU()
    (2): LinearWithLoRAMerged(
      (linear): Linear(in_features=128, out_features=256, bias=True)
      (lora): LoRALayer()
    )
    (3): ReLU()
    (4): LinearWithLoRAMerged(
      (linear): Linear(in_features=256, out_features=10, bias=True)
      (lora): LoRALayer()
    )
  )
)
"""
```

### Freezing the Original *Linear* Layers
Then, we can freeze the original ```Linear``` layers and only make the ```LoRALinear``` layers trainable, as follows:
```python
def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad=False
        else:
            # recursively freeze linear layers in children modules
            freeze_linear_layers(child)

freeze_linear_layers(model_lora)
for name, param in model_lora.named_parameters():
    print(f'{name}:{param.requires_grad}')

"""
layers.0.linear.weight:False
layers.0.linear.bias:False
layers.0.lora.A:True
layers.0.lora.B:True
layers.2.linear.weight:False
layers.2.linear.bias:False
layers.2.lora.A:True
layers.2.lora.B:True
layers.4.linear.weight:False
layers.4.linear.bias:False
layers.4.lora.A:True
layers.4.lora.B:True
"""
```

