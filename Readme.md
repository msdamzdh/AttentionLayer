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

## Usage

To create an instance of the `attentionLayer`:

```matlab
layer = attentionLayer(InputDim, QueryDim, ValueDim, OutputDim, NumberOfHead, InputFormat, UseMask, Name)
