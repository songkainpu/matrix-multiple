# Input Data
All input matrices are in the 'matrices' folder located in the project directory.
You don't need to worry about the input; we have already set the input path in the code. If you want to use a different data source, please modify the global variable MATRICES_FILE_FOLDER in the file.
# Set up
Recommend python interpreter version should be >= 3.8.1
mac or linux run:
```bash
python -m venv myenv
source myenv/bin/activate
pip install requirements.txt -r
```
windows run:
```bash
python -m venv myenv
myenv/bin/activate
pip install requirements.txt -r
```
Due to some algorithms used multiple-processes, Please run on Mac or Linux as much as possible to avoid compatibility issues
# Statement for files
## coppersmith-winograd.py
This script contains the Coppersmith-Winograd matrix algorithm, and the main function has already been written for you. You can execute this script directly using the Python command line.
```bash
python coppersmith-winograd.py
```
## fox_matrix_algorithm.py

```bash
python fox_matrix_algorithm.py
```
This script provides both sequential and parallel matrix multiplication for the Fox algorithm. Sequential multiplication is handled by the compute_single method, while parallel multiplication is handled by the _compute_matrices method. You can directly execute this script using the Python command line. It is already set up to use a single thread for sequential computation and to utilize multiple processes for parallel computation.

The number of processes in the process pool and the block sizes for Fox matrix decomposition will be automatically determined based on the number of CPU cores available on your system.
## GPU_matrix_algorithm_synchronize.py
```bash
python GPU_matrix_algorithm_synchronize.py
```
This script utilizes the OpenCL library to perform matrix operations on the GPU using a brute-force approach. You can execute this script directly using the Python command line. The script is already set up with a default method that will use your first available GPU device for matrix multiplication calculations.
## GPU_matrix_algorithm_fox.py
```bash
python GPU_matrix_algorithm_fox.py
```
This script utilizes the OpenCL library to perform brute-force matrix operations on the GPU. You can execute this script directly using the Python command line. The script is already set up with a default method that will utilize all the cores of your first available GPU device, using a Fox-like matrix decomposition approach for matrix calculations. Additionally, it leverages coroutines for GPU scheduling.
## sp1.py sp2.py
```bash
python sp1.py
python sp2.py
```
This script contains the Cannon Sequential matrix algorithm, and the main function has already been written for you. You can execute this script directly using the Python command line. sp1.py is used to compute data for only 'group0,' while sp2。py is used to compute matrices for all 10 groups.
## mp.py
```bash
python mp.py
```
This script contains the Cannon parallel matrix algorithm, and the main function has already been written for you. You can execute this script directly using the Python command line
