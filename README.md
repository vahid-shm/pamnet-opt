# PAMNet-QM9

## Run in Google Colab

Create a new Google Colab notebook and select a GPU runtime from **Runtime > Change runtime type**. Then run the following cells in order.

### 1. Clone the repository

```python
!git clone https://github.com/vahid-shm/pamnet-opt
%cd pamnet-opt
```

### 2. Install the requirements

```python
!pip install -r requirements.txt
```

### 3. Run the experiment

```python
!python -u main_qm9.py --model 'PAMNet' --variant 'original' --run_name 'pamnet_original'
```

Replace the three argument values in the command as needed:

| Argument | Accepted value | Description |
| --- | --- | --- |
| `--model` | `PAMNet` | Full PAMNet model |
|  | `PAMNet_s` | Simplified PAMNet model |
| `--variant` | `original` | Original architecture |
|  | `ws` | Weight sharing only |
|  | `br` | Basis reduction only |
|  | `wsbr` | Weight sharing and basis reduction |
| `--run_name` | Any custom name | Identifier for the run, such as `pamnet_ws` |

All other settings use the default values defined in `main_qm9.py`. To change them, edit those defaults directly in the code.
