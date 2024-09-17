# Quantum Reservoir Computing Surrogate

Surrogate: A 'quantum neural network' imitation of the Siemens reactor.

Quantum Reservoir Computing: A fixed quantum circuit and a trained classical linear layer.

Training (fitting) is done with a quadratic loss and ordinary least squares (OLS).

Contains three main QRC methods:
- QRewindingStatevectorRC (works well)
- QExtremeLearningMachine (works but no quantum advantage)
- QContinuousRC (not working yet)

And three main tasks:
- Surrogate from Siemens reactor data (QLindA)
- Nonlinear autoregressive moving average (NARMA) 
- Short term memory (STM)

## Quickstart

How to use the code: [./notebooks/try_me.ipynb](./notebooks/try_me.ipynb)

Plots for the QLindA report: [./experiments/plotting/plot_rewinding_reactor.ipynb](./experiments/plotting/plot_rewinding_reactor.ipynb)

`/notebooks/` examples for benchmark tasks and QRC methods

`/experiments/plotting/` plot results from experiments

`/src/` contains the QRC implementation

## Documentation

Open [./src/docs/build/html/index.html](./src/docs/build/html/index.html) in your browser

or `<your_code_folder>/quantum-reinforcement-learning/qrc_surrogate/src/docs/build/html/index.html`

## Setup: Conda (MacOS)

```bash
conda env create --name envrl --file=env_rl.yml

conda activate envrl
```

## Setup: Conda (Windows or Linux)

```bash
conda create --name "envrl" python=3.11
conda activate envrl

conda install jupyter==1.0.0 notebook=6.5.4 matplotlib=3.7.1 ipykernel=6.19.2 seaborn=0.12.2 -y
conda install -c anaconda ipython=8.12.0 statsmodels=0.13.5 scikit-learn=1.2.2 pytables=3.8.0 sphinx=5.0.2 -y
conda install -c conda-forge ipympl=0.9.3 gymnasium=0.26.3 tqdm=4.65.0 fastparquet=2023.4.0 sphinx-rtd-theme=1.2.2 readthedocs-sphinx-ext=2.2.1 sphinxcontrib-napoleon=0.7 -y
conda install -c pytorch pytorch=2.0.1 -y

conda install pip
python -m pip install qiskit==0.43.0 qiskit[providers]
python -m pip install qiskit==0.43.0 qiskit[visualization]
```

---

## Setup: Windows (alternative).

It may be the case that following the above instructions do not allow the code to run successfully;
In this case,the recommended approach is to install Windows Subsystem for Linux and Ubuntu (via the Microsoft store). One can then use e.g. VSCode with the WSL extension.



## What I did (for reproducability)

### Conda 

Export the conda environment

```bash
conda env export | grep -v "^prefix: " > env_rl.yml
```

### Documentation (Sphinx)

```bash
mkdir docs
cd docs

sphinx-quickstart --ext-autodoc
```

Separate source and build folders: yes

Edit conf.py and index.rst

Redo this if you change the comments in the code:

```bash
conda activate envrl
sphinx-apidoc -f -o ../docs/source/ ../src/
make clean
make html
```
