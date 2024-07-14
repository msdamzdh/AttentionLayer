# Attention Layer for MATLAB

This MATLAB script defines a custom attention layer class `attentionLayer` that can be used in deep learning models, particularly for sequence-to-sequence tasks or transformer-based architectures.

## Features

- Implements a multi-head attention mechanism
- Supports various input formats
- Optional causal masking
- Compatible with MATLAB's Deep Learning Toolbox

## Class Definition

The `attentionLayer` class is defined as a subclass of `nnet.layer.Layer` and `nnet.layer.Formattable`.

### Properties

- `Nhead`: Number of attention heads
- `InFormat`: Input format specification
- `UseMask`: Flag for using causal masking

### Learnable Parameters

- `Wq`: Query weight matrix
- `Wk`: Key weight matrix
- `Wv`: Value weight matrix
- `Wo`: Output weight matrix
  
### Parameters

- `InputDim`: Dimension of the input
- `QueryDim`: Dimension of the query
- `ValueDim`: Dimension of the value
- `OutputDim`: Dimension of the output
- `NumberOfHead`: Number of attention heads
- `InputFormat`: Format of the input tensor (e.g., "CBT", "CTB", "BTC", etc.)
- `UseMask`: Boolean flag for causal masking (default: false)
- `Name`: Name of the layer (default: "attentionLayer")

### Methods
#### predict
The predict method performs the forward pass of the attention layer:
1. Reshapes the input tensor based on the specified input format
2. Computes query, key, and value matrices
3. Applies multi-head attention
4. Produces the output

### Dependencies

- MATLAB
- Deep Learning Toolbox

### Notes

The layer supports various input tensor formats and automatically reshapes the input accordingly.
Causal masking can be enabled for autoregressive models.
The implementation uses MATLAB's dlarray for GPU compatibility.

Example

## Usage

To create an instance of the `attentionLayer`:

```matlab
layer = attentionLayer(InputDim, QueryDim, ValueDim, OutputDim, NumberOfHead, InputFormat, UseMask, Name)
```
1. The layer supports various input tensor formats and automatically reshapes the input accordingly.
2. Causal masking can be enabled for autoregressive models.
3. The implementation uses MATLAB's dlarray for GPU compatibility.

### Example 
```matlab
% Create an attention layer
attLayer = attentionLayer(512, 64, 64, 512, 8, "CBT", true, "MyAttentionLayer");

% Use the layer in a network
% ... (add other layers as needed)
layers = [ ...
    % ... previous layers
    attLayer
    % ... subsequent layers
];

% Create and train the network
net = dlnetwork(layers);
% ... (training code)
```
For more information on using custom layers in MATLAB, refer to the MATLAB Deep Learning Toolbox documentation.
