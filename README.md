# INF2072
INF2072 - MultiAgent

## Environment Installation Guide (PyTorch + TensorFlow)

Below are the detailed steps to set up a development environment with the correct versions of PyTorch, TensorFlow, and other essential data science libraries, ensuring CUDA compatibility.

---

#### **Step 1: Install PyTorch (with the correct command)**

First, install PyTorch, torchvision, and torchaudio using the official command that points to the binaries compiled for CUDA 12.1. This is compatible with a system that has CUDA 12.1 installed, as the important thing is the NVIDIA driver version.

Run this command in your terminal:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

#### **Step 4: Connect your environment to Jupyter (Optional)**

To use this environment within Jupyter Notebook or JupyterLab, you need to install `ipykernel` and register it.

1.  **Install ipykernel**:
    Inside your active environment, install the `ipykernel` package. This allows Jupyter to communicate with your environment's Python interpreter.
    ```bash
    # If you are using conda
    conda install ipykernel
    # Or if you are using pip/venv
    pip install ipykernel
    ```

2.  **Register the kernel with Jupyter**:
    Now, you need to register this environment as a new kernel. Replace `your_env_name` with the actual name of your environment.
    ```bash
    python -m ipykernel install --user --name=your_env_name --display-name="Python (your_env_name)"
    ```
    -   `--name=your_env_name`: This is the internal name Jupyter will use.
    -   `--display-name="Python (your_env_name)"`: This is the name you will see in Jupyter's kernel menu.

---

#### **Process Summary**

> 1.  **Install PyTorch** with the command from Step 1.
> 2.  **Create the `requirements.txt` file** with the content from Step 2.
> 3.  **Install the rest of the libraries** with the command from Step 3.
> 4.  **Connect to Jupyter** by installing and registering `ipykernel` as shown in Step 4.

By following these steps, you will have an environment with all the requested libraries, configured to use your GPU and ready to be used in Jupyter. ✅

