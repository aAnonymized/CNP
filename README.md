## CNP: Counterfactual-Calibrated Nuisance Purification for Long-Tailed Medical Image Classification

## Installation
### Environment set up
Install required packages: see CNP.yaml


## Run Experiments

### Stage1: train baseline and Frozen Class Anchoring.

**Training:**

```bash
python CNP.py --config ./config/isic/100/CNP.py
```
- The ./config/isic/100/CNP.py `stage='IM'`
- The ./config/isic/100/CNP.py `method='CNP'` VSLoss, CB, Focal, LEAD ...

### Stage1: Nuisance Purification

**Training:**

```bash
python CNP.py --config ./config/isic/100/CNP.py
```
- The ./config/isic/100/CNP.py `stage='Purification'`
- The ./config/isic/100/CNP.py `method='CNP'`
- The CNP.py `prototype_k_factors=[2, 3, 4]` ：It denotes selecting the number of clusters from the range of `2 × class_num` to `4 × class_num`.
- The CNP.py `decision_threshold=0.1` : eval by NRG
