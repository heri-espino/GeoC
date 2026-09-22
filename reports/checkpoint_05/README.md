# Checkpoint 05 report namespace

This directory is reserved for the final global-local stacking experiment.

The workstation runner is:

    python tools\run_checkpoint_05.py

Partial files beginning with _partial_ are written after every completed outer split. They
exist only to make long workstation runs resumable and are deleted after a successful run.

The canonical 05 output will be final_predictions.csv only if the configured split-excluded
promotion gate is evaluated. Until then, Checkpoint 04F remains the frozen incumbent.
