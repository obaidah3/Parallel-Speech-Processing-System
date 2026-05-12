# Hybrid MPI + OpenMP Audio Signal Processing System

A high-performance, academic-grade audio signal processing system implementing hybrid MPI + OpenMP parallelism for distributed computing environments.

## Overview

This project demonstrates advanced HPC concepts through a complete audio processing pipeline, featuring:

- **Sequential Baseline**: Reference implementation for performance comparison
- **OpenMP Parallelization**: Thread-level parallelism for shared-memory systems
- **MPI Distribution**: Process-level parallelism for distributed-memory clusters
- **STFT/FFT Processing**: Frequency-domain analysis with optimized FFTW3
- **Real-time Filtering**: Speech band-pass and noise reduction
- **Feature Extraction**: MFCC, spectral features for audio analysis
- **Performance Benchmarking**: Scalability analysis and efficiency metrics

## Architecture

### System Components

```
audio-project/
├── Core Processing Pipeline
│   ├── WAV I/O (libsndfile)
│   ├── DSP Preprocessing (normalization, DC removal, pre-emphasis)
│   ├── STFT/FFT (FFTW3 optimized)
│   ├── Frequency Filtering
│   ├── Feature Extraction
│   └── ISTFT Reconstruction
├── Parallel Frameworks
│   ├── OpenMP (threading)
│   └── MPI (distributed processing)
├── Benchmarking & Analysis
│   ├── Timing utilities
│   ├── Scalability metrics
│   └── Visualization scripts
```

### MPI Architecture

- **Master Process (Rank 0)**: Audio I/O, preprocessing, chunk distribution
- **Worker Processes**: Parallel STFT, filtering, feature extraction
- **Load Balancing**: Dynamic chunk sizing for optimal distribution

### OpenMP Integration

- **STFT Parallelization**: Frame-level parallelism
- **Filtering Operations**: Frequency bin parallel processing
- **Feature Extraction**: Independent frame analysis

## Technical Specifications

- **Language**: C++17
- **Parallelism**: MPI + OpenMP hybrid
- **FFT Library**: FFTW3 (optimized plans)
- **Audio I/O**: libsndfile (PCM WAV support)
- **Build System**: GNU Make with automatic dependencies
- **Visualization**: Python (matplotlib, numpy, pandas)

## Performance Characteristics

### STFT Parameters
- FFT Size: 2048 samples
- Hop Size: 1024 samples (50% overlap)
- Window: Hann window

### Filtering
- Speech Band-pass: 300Hz - 3400Hz
- Frequency-domain implementation for efficiency

### Features Extracted
- RMS Energy
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Flatness
- MFCC (13 coefficients)

## Build Instructions

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential
sudo apt install libopenmpi-dev openmpi-bin
sudo apt install libsndfile1-dev
sudo apt install libfftw3-dev libfftw3-single3
sudo apt install python3 python3-pip python3-matplotlib python3-numpy python3-pandas scipy
```

### Compilation

```bash
make clean
make
```

### Execution Modes

The system supports four execution modes selected via command-line arguments:

#### Sequential Mode
```bash
./audio_app sequential
```
- Single-threaded, no parallelism
- Reference implementation for performance comparison
- Output: `data/output_sequential.wav`, `data/results/features_sequential.csv`

#### OpenMP Mode
```bash
./audio_app omp [num_threads]
```
- Thread-level parallelism using OpenMP
- Default: 4 threads
- Example: `./audio_app omp 8`
- Output: `data/output_omp.wav`, `data/results/features_omp.csv`

#### MPI Mode
```bash
mpirun -np <processes> ./audio_app mpi
```
- Process-level parallelism using MPI
- Workers process chunks sequentially
- Example: `mpirun -np 4 ./audio_app mpi`
- Output: `data/output_mpi.wav`, `data/results/features_mpi.csv`

#### Hybrid MPI + OpenMP Mode
```bash
mpirun -np <processes> ./audio_app hybrid <processes> <threads>
```
- Combined MPI + OpenMP parallelism
- Workers use OpenMP for intra-process threading
- Example: `mpirun -np 2 ./audio_app hybrid 2 4`
- Output: `data/output_hybrid.wav`, `data/results/features_hybrid.csv`

## Benchmarking

### Automated Benchmarking

```bash
python3 scripts/benchmark.py
```

### Performance Analysis

```bash
python3 scripts/plot_results.py
```

### Audio Visualization

```bash
python3 scripts/visualize_audio.py
```

## Scalability Analysis

### Speedup Calculation
```
S(p) = T_sequential / T_parallel(p)
```

### Efficiency Metrics
```
E(p) = S(p) / p
```

### Expected Results

- **MPI Scaling**: Near-linear speedup on distributed systems
- **OpenMP Scaling**: Diminishing returns beyond 8-16 threads
- **Hybrid Efficiency**: Optimal performance combining both paradigms

## File Structure

```
data/
├── raw/           # Input audio files
├── processed/     # Intermediate results
├── chunks/        # MPI chunk data
└── results/       # Benchmarks and features

src/
├── io/            # Audio I/O modules
├── dsp/           # Signal processing
├── parallel/      # MPI/OpenMP implementations
├── sequential/    # Baseline implementation
└── utils/         # Helper functions

scripts/           # Python analysis tools
include/           # Header files
```

## HPC Best Practices Implemented

- **Memory Management**: FFTW optimized plans, buffer reuse
- **Load Balancing**: Dynamic chunk distribution in MPI
- **Thread Safety**: Critical sections for shared resources
- **Error Handling**: Exception-based error propagation
- **Modular Design**: Clean separation of concerns
- **Performance Profiling**: Comprehensive timing utilities

## Academic Relevance

This project serves as a comprehensive example of:

- **Parallel Computing**: MPI process management, OpenMP threading
- **Signal Processing**: Real-time audio analysis techniques
- **High-Performance Computing**: Scalability, efficiency metrics
- **Software Engineering**: Modular architecture, build systems
- **Scientific Computing**: FFT algorithms, feature extraction

## Future Enhancements

- GPU acceleration (CUDA/OpenCL)
- Real-time streaming processing
- Advanced ML-based feature extraction
- Distributed file systems integration
- Containerized deployment (Docker/Singularity)

## License

Academic use permitted. Commercial licensing available upon request.

## Contact

For questions or contributions, please open an issue or submit a pull request.