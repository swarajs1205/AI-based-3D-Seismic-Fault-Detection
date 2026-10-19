# AI Seismic Dip & Fault Extraction

Advanced seismic dip and azimuth analysis using industry-standard Weighted Structural Tensor method. This project transforms raw seismic data into detailed geological maps that reveal rock layer orientations, dip angles, and structural features through sophisticated mathematical analysis.

## Features

Implements cutting-edge approach for seismic attribute extraction using Weighted Structural Tensor methodology:
- Gradient-based directional analysis for 3D geological structure identification
- Energy-based weighting using Hilbert transform for noise suppression
- Uniform filtering for smooth, natural results
- Eigenvalue decomposition for dominant direction extraction
- Memory-efficient chunked processing for large datasets

## Applications

Used in oil and gas exploration, geothermal energy mapping, carbon sequestration monitoring, academic research, and geological surveys. The extracted attributes help identify reservoirs, fault systems, and subsurface structures with professional accuracy.

## Installation

```bash
pip install numpy scipy matplotlib segyio
```

## Quick Start

```bash
python seismic_analysis.py
```

## Methodology

The project implements the Weighted Structural Tensor approach with memory-efficient chunked processing:

### Data Ingestion
SEG-Y file reading using segyio library with robust error handling. Reshapes flat trace data into 3D grid (Inlines × Crosslines × Samples) representing the complete subsurface volume. Processes full F3 dataset (631×951×462) in 10,000 trace chunks to manage memory efficiently.

### Gradient Calculation
Mathematical gradient computation using central finite differences:
```
∇u = (∂u/∂x, ∂u/∂y, ∂u/∂z)
```
Where gradients Vx, Vy, Vz represent directional changes in three dimensions, creating a 3D compass for every subsurface point.

### Weighted Filtering
Energy-based weighting using complex analytic signal:
```
w = |H(u)|⁴ = |u + i·H(u)|⁴
```
Where H(u) is the Hilbert transform, creating high contrast between real geological features (high energy) and random noise (low energy).

### Structural Tensor Construction
3×3 symmetric tensor from weighted dyadic products:
```
T = w × [∇u × ∇uᵀ]
```
Components include wVxx, wVyy, wVzz (diagonal) and wVxy, wVxz, wVyz (off-diagonal), smoothed with 5×5×5 uniform filter.

### Eigenvalue Decomposition
Solves the eigenvalue problem:
```
T × v = λ × v
```
Extracts dominant eigenvector v_max representing the most consistent structural orientation.

### Dip Angle Calculation
Converts eigenvector to geological dip angle:
```
dip = arctan2(√(vₓ² + vᵧ²), |v_z|)
```
Results in degrees from 0° (horizontal) to 90° (vertical).

## Memory Management

Optimized chunked processing for large datasets:
- Data loading in 10,000 trace chunks
- Float32 precision to reduce memory by 50%
- Progress indicators for monitoring
- Peak memory usage ~500 MB for full F3 dataset

## Output

Professional 2-panel visualization:
- Left panel: Original seismic slice (grayscale)
- Right panel: Dip angle map (jet colormap, 0-30°)
- Colorbar shows dip angle scale in degrees
- High-resolution 12×6 inch figure

## Geological Interpretation

Results enable detailed geological interpretation:
- Red areas (25-30°): Steep structures like salt domes
- Blue/green areas (0-10°): Flat, undisturbed layers  
- Sharp boundaries: Faults and discontinuities
- Smooth transitions: Natural geological formations
