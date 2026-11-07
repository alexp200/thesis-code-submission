#!/usr/bin/env bash

# Copy this file to config.local.sh on the cluster and fill in these values.
export PROJECT_ACCOUNT="replace-with-project-account"
export TOOLCHAIN_MODULE="replace-with-required-toolchain-module"
export PYTHON_MODULE="replace-with-python-3.12-module"
export GPU_GRES="gpu:rtx6000:1"
export REPO_DIR="${HPC_SCRATCH}/thesis_code"
