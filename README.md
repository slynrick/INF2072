# INF2072
INF2072 - MultiAgent

## Environment Installation Guide (PyTorch + TensorFlow)

Below are the detailed steps to set up a development environment with the correct versions of PyTorch, TensorFlow, and other essential data science libraries, ensuring CUDA compatibility.

---

#### **Step 1: Install PyTorch (with the correct command)**

First, install PyTorch, torchvision, and torchaudio using the official command that points to the binaries compiled for CUDA 12.1. This is compatible with a system that has CUDA 12.1 installed, as the important thing is the NVIDIA driver version.

Run this command in your terminal:
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

#### **Step 2: Create and use the `requirements.txt` file**

Now, create a file named `requirements.txt` with the following content. This file includes TensorFlow 2.15 (which is compatible with CUDA 12.1 and cuDNN 8.9) and the other data science libraries.

```plaintext
# requirements.txt

# TensorFlow compatible with CUDA 12.1 and cuDNN 8.9
tensorflow==2.15.0

# Standard Machine Learning and visualization libraries
scikit-learn
matplotlib
seaborn
pandas
numpy
```

#### **Step 3: Install the remaining dependencies**

With the `requirements.txt` file in the folder, run the following command in your terminal to install the rest of the libraries:

```bash
pip install -r requirements.txt
```

---

#### **Process Summary**

> 1.  **Install PyTorch** with the command from Step 1.
> 2.  **Create the `requirements.txt` file** with the content from Step 2.
> 3.  **Install the rest of the libraries** with the command from Step 3.

By following these steps, you will have an environment with all the requested libraries, configured to use your GPU with the CUDA and cuDNN versions you specified. ✅

