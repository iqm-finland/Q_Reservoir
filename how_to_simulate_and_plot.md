# Simulations and plots for the QLindA report

## Simulate

1. open a terminal
2. type line-by-line

```bash
cd ~/RLCode/quantum-reinforcement-learning/qrc_surrogate/experiments/simulating/
conda activate envrl 
python3.11 sweeps_run.py
```

## Plot

after simulations are finished
1. open
`RLCode/quantum-reinforcement-learning/qrc_surrogate/experiments/plotting/plot_rewinding_reactor.ipynb`
2. Run All

## Upload new plots to Overleaf
upload all plots in
`RLCode/quantum-reinforcement-learning/qrc_surrogate/plots/plots_reactor/`
to `/plots/` in Overleaf

## Commit changes to gitlab
 
1. open a terminal
2. type line-by-line

```bash
cd ~/RLCode/quantum-reinforcement-learning
git add * 
git commit -am 'simulations for QLindA report'
# git pull
git push
```
