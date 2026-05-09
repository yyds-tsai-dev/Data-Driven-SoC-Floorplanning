# ICCAD 2026 FloorSet Optimizer

This package contains my optimizer files for the ICCAD 2026 FloorSet Challenge.

## Files

Please put all files under:
FloorSet/iccad2026contest/

Required files:

my_optimizer.py
my_optimizer_frame_first_best.py
my_optimizer_anchor_v2_best.py
gnn_model.py
checkpoints/gnn_best.pt

File descriptions:

my_optimizer.py

Official entry file loaded by iccad2026_evaluate.py.

my_optimizer_frame_first_best.py

Main optimizer wrapper. It applies the frame-first post-repair optimizer, including MIB, boundary, and cluster/grouping repair.

my_optimizer_anchor_v2_best.py

Base optimizer. It uses GNN anchors if the trained checkpoint is available. If the checkpoint is missing, it can still fall back to non-GNN anchors, but the score may be different.

gnn_model.py

GNN model definition and feature-building functions.

checkpoints/gnn_best.pt

Trained GNN checkpoint. Keep this exact path:

checkpoints/gnn_best.pt

Do not place gnn_best.pt directly in iccad2026contest/.

Folder structure

The final structure should look like this:

FloorSet/
└── iccad2026contest/
    ├── iccad2026_evaluate.py
    ├── my_optimizer.py
    ├── my_optimizer_frame_first_best.py
    ├── my_optimizer_anchor_v2_best.py
    ├── gnn_model.py
    └── checkpoints/
        └── gnn_best.pt

## Required FloorSet repository files

This package only contains the optimizer files.  
Your local `FloorSet/iccad2026contest/` folder must already contain the original contest framework files and validation dataset.

Required original contest files:

```text
iccad2026_evaluate.py
litetestLoader.py
liteLoader.py
lite_dataset.py
cost.py
utils.py
```
Required validation dataset:

LiteTensorDataTest/

Usually, these files and folders already exist if you cloned the official FloorSet repository and downloaded the validation dataset correctly.

# How to run

Go to the contest folder:

cd ~/FloorSet/iccad2026contest

Validate the optimizer:
python3 iccad2026_evaluate.py --validate my_optimizer.py

Run one test case:
python3 iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 91

Run full validation:
python3 iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions

Re-score saved solutions:
python3 iccad2026_evaluate.py --score my_optimizer_solutions.json


## Optional training reference

The following file is optional:

```text
train_gnn_anchor_v2.py

This file is included only as a reference to show how the GNN checkpoint was trained.

You do not need this file to run evaluation.
You do not need to retrain the model.