import numpy as np
import segyio
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

def process_f3_weighted_tensor(file_path):
    print(f"--- Loading File: {file_path} ---")
    
    with segyio.open(file_path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size
        print(f"Total Traces in file: {n_traces}, Samples per Trace: {n_samples}")
        
        # We have choose the exact dimensions of F3 Dataset because it was showing error.
        inlines = 631
        xlines = 951
        required_traces = inlines * xlines
        
        print(f"Required traces for grid: {required_traces}")
        # Reading required traces for Grid Formation with chunking
        print("Loading data in chunks to manage memory...")
        chunk_size = 10000
        data = np.zeros((inlines, xlines, n_samples), dtype=np.float32)
        
        for chunk_start in range(0, required_traces, chunk_size):
            chunk_end = min(chunk_start + chunk_size, required_traces)
            chunk_traces = chunk_end - chunk_start
            
            print(f"Loading traces {chunk_start}-{chunk_end} of {required_traces}")
            chunk_data = segyio.tools.collect(f.trace[chunk_start:chunk_end])
            
            # Place chunk in 3D grid
            for i, trace_offset in enumerate(range(chunk_traces)):
                inline_idx = (chunk_start + i) // xlines
                xline_idx = (chunk_start + i) % xlines
                data[inline_idx, xline_idx, :] = chunk_data[i]
        
        print("Successfully loaded and reshaped data into 3D grid.")
    
    print("--- Calculating Gradients & Weights ---")
    # Process in smaller chunks to manage memory
    print("Processing gradients in chunks...")
    
    # Gradient calculation [cite: 25, 26]
    Vx, Vy, Vz = np.gradient(data)
    
    # Weight computation: squared power of complex traces [cite: 61, 105]
    print("Computing analytic signal...")
    analytic_signal = hilbert(data, axis=2)
    weight = (np.abs(analytic_signal)**2)**2
    
    print("--- Computing Weighted Structural Tensor ---")
    # Smoothing dyadic products in a 3D window [cite: 31, 53]
    window = 5
    
    # Process tensor components in chunks
    print("Computing tensor components...")
    wVxx = uniform_filter(weight * Vx * Vx, size=window)
    wVyy = uniform_filter(weight * Vy * Vy, size=window)
    wVzz = uniform_filter(weight * Vz * Vz, size=window)
    wVxy = uniform_filter(weight * Vx * Vy, size=window)
    wVxz = uniform_filter(weight * Vx * Vz, size=window)
    wVyz = uniform_filter(weight * Vy * Vz, size=window)
    
    print("--- Processing Dip for Visualization ---")
    # Eigenvalue logic for a representative slice [cite: 35, 37]
    slice_idx = 300 # Slice just in the middle.
    dip_slice = np.zeros((xlines, n_samples))
    
    for i in range(xlines):
        for j in range(n_samples):
            # Matrix construction [cite: 53, 58]
            ST = np.array([[wVxx[slice_idx,i,j], wVxy[slice_idx,i,j], wVxz[slice_idx,i,j]],
                           [wVxy[slice_idx,i,j], wVyy[slice_idx,i,j], wVyz[slice_idx,i,j]],
                           [wVxz[slice_idx,i,j], wVyz[slice_idx,i,j], wVzz[slice_idx,i,j]]])
            
            # Largest eigenvector of an eigenvalue. [cite: 37, 123]
            evals, evecs = np.linalg.eigh(ST)
            v_max = evecs[:, -1]
            
            # Dip angle calculation
            dip_slice[i, j] = np.degrees(np.arctan2(np.sqrt(v_max[0]**2 + v_max[1]**2), np.abs(v_max[2])))
    
    return data[slice_idx], dip_slice

# --- Execution ---
try:
    original_slice, dip_result = process_f3_weighted_tensor("Seismic_data.sgy")
    if original_slice is not None:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_slice.T, cmap='gray', aspect='auto')
        plt.title("Original Seismic Slice")
        
        plt.subplot(1, 2, 2)
        plt.imshow(dip_result.T, cmap='jet', aspect='auto', vmin=0, vmax=30)
        plt.title("Weighted Structural Tensor Dip")
        plt.colorbar(label="Degrees")
        
        plt.tight_layout()
        plt.show()
    else:
        print("Data couldn't be processed.")
except Exception as e:
    print(f"Run-time error: {e}")
