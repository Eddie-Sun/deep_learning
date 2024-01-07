# Model Components

###### [Progress] January 6, 2024
###### [Progress] Elaborated notes from [Little Book of Deep Learning ](https://fleuret.org/public/lbdl.pdf?fbclid=IwAR3jmeQf1k6Q6Qbp6fDmEtklfqo3XMNrHSoIE_2m8By8cpF2sPZjghuq-Zg)

## What is a layer?
### Linear Layers
"Linear" in deep learning != [linear in math](https://en.wikipedia.org/wiki/Linearity#:~:text=In%20mathematics%2C%20a%20linear%20map,(x)%20for%20all%20%CE%B1.). The linear in layers, refers to an [affine operation](https://youtu.be/E3Phj6J287o?si=YW0ya5B9iY3OtiQb) (linear transformation): 

$$ y = Ax + b $$

Where: 
- y = is the transformed vector.
- A = is a linear transformation matrix.
- x = is the original vector.
- b = is a translation vector.

### Dense Layers/fully connected layers
The [fully connected](https://www.youtube.com/watch?v=Tsvxx-GGlTg) layer (every neuron is connected) is simple affine transformation that can work with input/outputs from multi-dimensional tensors. The mathematical expression is:

$$ Y[d_1, d_2, \ldots, d_K] = W \cdot X[d_1, d_2, \ldots, d_K] + b $$

The input tensor X with dimensions D<sub>1</sub> × D<sub>2</sub> × … × D<sub>K</sub> × D, resulting in an output tensor Y with dimensions D<sub>1</sub> × D<sub>2</sub> × … × D<sub>K</sub> × D'. The weight matrix W is of size D' × D, and b is the bias vector of dimension D'.

<details>
  <summary>Python Function to Visualize Fully Connected Layer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_affine_transformation(input_dims, output_dim):
    """
    Visualize the affine transformation performed by a fully connected layer.

    :param input_dims: Dimensions of the input tensor (excluding the last dimension).
    :param output_dim: Dimension of the output tensor's last dimension.
    """
    # Generate a random input tensor with the specified dimensions
    input_tensor = np.random.randn(*input_dims)

    # Initialize the weight matrix and bias vector
    W = np.random.randn(output_dim, input_dims[-1])
    b = np.random.randn(output_dim)

    # Applying affine transformation for each vector in the input tensor
    reshaped_input = input_tensor.reshape(-1, input_dims[-1])  # Flatten input except for the last dimension
    transformed = np.dot(reshaped_input, W.T) + b  # Apply affine transformation
    output_tensor = transformed.reshape(*input_dims[:-1], output_dim)  # Reshape to the output tensor

    # Visualization
    plt.figure(figsize=(12, 6))

    # Visualizing the input tensor
    plt.subplot(1, 2, 1)
    plt.imshow(input_tensor.reshape(-1, input_dims[-1]), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Input Tensor")

    # Visualizing the output tensor
    plt.subplot(1, 2, 2)
    plt.imshow(output_tensor.reshape(-1, output_dim), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Output Tensor (After Affine Transformation)")

    plt.show()

# Example usage: visualizing a 2D input tensor transforming to a different dimension
visualize_affine_transformation(input_dims=(10, 20), output_dim=5)
```
<img src="image-1.png" alt="Alt text" width="500"/>  
</details>
</br>

[Projections](https://www.cuemath.com/geometry/projection-vector/) can be used for dimension reduction or singal filtering.
<details>
  <summary>Python Function to Visualize Dimensionality Reduction</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

def visualize_before_after_pca(data, n_components=2):
    """
    Visualize the data before and after applying PCA for dimensionality reduction.

    :param data: The input data as a 2D numpy array.
    :param n_components: The number of dimensions to reduce the data to.
    """
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    # Visualization
    fig = plt.figure(figsize=(15, 6))

    # Before PCA: Plotting the original data in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.7, color='blue')
    ax1.set_title("Original Data (3D)")
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')

    # After PCA: Plotting the reduced data in 2D
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, color='red')
    ax2.set_title("Data after PCA (2D)")
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')

    plt.show()

# Generate a smaller synthetic dataset for clear visualization
np.random.seed(0)
X_small = np.random.rand(30, 3)  # 30 samples with 3 features

# Visualize before and after PCA
visualize_before_after_pca(X_small, n_components=2)

```
<img src="image-2.png" alt="Alt text" width="500"/> 
</details>

### Convolutional Layers

<!--START OF FOOTER-->
<hr style="margin-top:9px;height:1px;border: 0;background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0), rgba(0, 0, 0, 0.5),rgba(0, 0, 0, 0.0));">
<!--START OF ISSUE NAVIGATION LINKS-->
<!--START OF ISSUE NAVIGATION LINKS-->
<!--END OF FOOTER-->
