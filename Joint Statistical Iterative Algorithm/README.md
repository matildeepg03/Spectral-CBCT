# Image Reconstruction for Spectral Cone Beam Computed Tomography

Simulation, reconstruction and SPR extraction pipeline for 3D Cone Beam CT projections using OpenGATE 10 and ASTRA Toolbox.

---

## Overview

This repository contains code for:

- CT simulations using OpenGATE
- CT simulations using ASTRA Toolbox
- Generating 3D sinograms
- Reconstruction and visualization utilities

---

## Features

- Load volumetric phantom data
- Generate projection geometries
- Create 3D sinograms
- GPU-accelerated forward projection
- Reconstruction utilities
- Visualization tools

---

## Repository Structure

```text
project/
│
├── data/               # Input phantom or projection data
├── ROOT2/              # ROOT phantom files
├── scripts/            # Main processing scripts
├── notebooks/          # Jupyter notebooks
├── outputs/            # Generated sinograms/reconstructions
├── utils/              # Helper functions
└── README.md

