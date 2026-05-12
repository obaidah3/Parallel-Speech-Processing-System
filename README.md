# Hybrid MPI + OpenMP Audio Signal Processing Framework

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5+-green.svg)](https://www.openmp.org/)
[![MPI](https://img.shields.io/badge/MPI-3.1+-red.svg)](https://www.mpi-forum.org/)
[![FFTW](https://img.shields.io/badge/FFTW-3.3+-purple.svg)](http://www.fftw.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade, high-performance computing framework for distributed audio signal processing, implementing hybrid MPI + OpenMP parallelism with advanced DSP algorithms and comprehensive benchmarking.

## Overview

This project implements a complete HPC audio processing pipeline that demonstrates state-of-the-art parallel computing techniques for signal processing applications. The system processes WAV audio files through a multi-stage pipeline including preprocessing, frequency-domain analysis, filtering, and feature extraction, while supporting multiple parallel execution models.

### Why This Project Exists

The framework addresses critical challenges in modern audio processing:

- **HPC Relevance**: Demonstrates scalable parallel algorithms for signal processing workloads
- **DSP Relevance**: Implements research-grade audio analysis techniques with real-time performance
- **Distributed Systems Relevance**: Shows hybrid shared/distributed memory parallelism for cluster computing

The system serves as both an educational tool for HPC concepts and a foundation for research in distributed audio processing, machine learning feature extraction, and real-time signal analysis.

## Key Features

### Core DSP Pipeline
- **Sequential Baseline**: Reference implementation for baseline performance and correctness validation
- **OpenMP Parallelization**: Thread-level parallelism for shared-memory systems with dynamic scheduling
- **MPI Distributed Processing**: Process-level parallelism for cluster environments with load balancing
- **Hybrid Architecture**: Combined MPI + OpenMP for maximum scalability across nodes and cores

### Audio Processing Capabilities
- **WAV I/O**: High-performance reading/writing using libsndfile with mono/stereo support
- **Preprocessing**: Normalization, DC offset removal, pre-emphasis filtering
- **STFT/ISTFT**: Overlap-add algorithm with configurable window size and hop length
- **Frequency Filtering**: Speech-band filtering with configurable cutoff frequencies
- **Feature Extraction**: MFCC coefficients, spectral centroid, bandwidth, flatness, RMS, ZCR
- **Audio Reconstruction**: Perfect reconstruction via inverse STFT with overlap-add

### Performance & Analysis
- **Comprehensive Benchmarking**: Runtime measurement, throughput, speedup, efficiency metrics
- **Scalability Analysis**: Weak/strong scaling studies across different problem sizes
- **Memory Optimization**: Flat buffer storage for cache-efficient spectrogram processing
- **Load Balancing**: Size-aware task distribution for optimal resource utilization

## System Architecture

### Module Hierarchy

```
audio-project/
├── Core DSP Engine
│   ├── Audio I/O (libsndfile integration)
│   ├── Preprocessing (normalization, filtering)
│   ├── STFT Engine (FFTW3 optimized transforms)
│   ├── Filtering (frequency-domain processing)
│   ├── Feature Extraction (MFCC, spectral features)
│   └── Reconstruction (ISTFT with overlap-add)
├── Parallel Processing Framework
│   ├── Sequential Pipeline (reference implementation)
│   ├── OpenMP Pipeline (shared-memory parallelism)
│   ├── MPI Pipeline (distributed-memory parallelism)
│   └── Hybrid Pipeline (MPI + OpenMP combination)
├── Dataset Management
│   ├── Recursive file scanning
│   ├── Task generation and distribution
│   └── Output hierarchy management
└── Benchmarking & Analysis
    ├── Timing utilities (high-resolution)
    ├── Performance metrics calculation
    ├── CSV output generation
    └── Python visualization scripts
```

### Pipeline Flow

```
Input WAV → Preprocessing → STFT → Filtering → Feature Extraction → ISTFT → Output WAV
                    ↓           ↓           ↓           ↓           ↓
              Benchmarking  Benchmarking  Benchmarking  Benchmarking  Benchmarking
```

### Dataset Workflow

1. **File Discovery**: Recursive scanning of dataset directory for WAV files
2. **Task Generation**: Creation of processing tasks with input/output path mapping
3. **Load Balancing**: Size-aware distribution across MPI ranks (greedy assignment)
4. **Parallel Processing**: Independent task execution with local I/O
5. **Result Aggregation**: Distributed benchmark collection and global statistics

### Execution Modes

- **Sequential**: Single-threaded processing for baseline performance
- **OpenMP**: Multi-threaded shared-memory processing with thread-local FFT plans
- **MPI**: Distributed processing with file-level parallelism with non-blocking communication
- **Hybrid**: Combined distributed + shared-memory parallelism (planned)

### Communication Model

- **MPI**: Non-blocking receives with `MPI_Irecv`/`MPI_Wait` for overlapped computation
- **Load Balancing**: File-size based greedy assignment minimizing maximum rank load
- **Result Collection**: Distributed I/O with `MPI_Allreduce` for global statistics
- **Synchronization**: Barrier-free design with independent rank operation

## Technology Stack

### C++17
**Role**: Core implementation language providing modern features for HPC development
**Advantages**:
- `std::filesystem` for portable file operations
- Structured bindings for tuple returns
- `thread_local` for thread-safe FFT plan management
- Template metaprogramming for generic DSP algorithms
**Performance Implications**: Zero-cost abstractions, efficient memory management
**HPC Relevance**: Standard in scientific computing with excellent compiler optimization

### FFTW3
**Role**: High-performance FFT library for frequency-domain processing
**Advantages**:
- Optimized for various architectures with SIMD support
- Thread-safe plan caching with `fftw_plan` reuse
- Real/complex transforms with in-place operations
- Extensive algorithm selection for different transform sizes
**Performance Implications**: 2-10x speedup over naive DFT implementations
**HPC Relevance**: De facto standard for scientific FFT computations

### OpenMP 4.5+
**Role**: Shared-memory parallelization for intra-node parallelism
**Advantages**:
- `#pragma omp parallel for` for loop parallelization
- `schedule(dynamic)` for load balancing across frames
- Thread-local storage for FFT plan isolation
- Nested parallelism support for hybrid execution
**Performance Implications**: Near-linear scaling to multi-core systems
**HPC Relevance**: Essential for exploiting modern multi-core processors

### MPI 3.1+
**Role**: Distributed-memory parallelization for cluster computing
**Advantages**:
- `MPI_Irecv`/`MPI_Wait` for non-blocking communication
- `MPI_Allreduce` for collective operations
- Process-based parallelism with message passing
- Scalable to thousands of nodes
**Performance Implications**: Enables processing of massive datasets across clusters
**HPC Relevance**: Standard for distributed scientific computing

### libsndfile
**Role**: Audio file I/O with format support
**Advantages**:
- Wide format support (WAV, AIFF, FLAC, etc.)
- Efficient memory-mapped I/O
- Sample rate/format conversion
- Thread-safe operations
**Performance Implications**: Minimal I/O overhead in processing pipeline
**HPC Relevance**: Reliable data ingestion for signal processing workflows

### Python Ecosystem
**Role**: Benchmarking analysis and visualization
**Advantages**:
- `pandas` for CSV processing and statistical analysis
- `matplotlib` for performance plotting and scalability graphs
- `numpy` for numerical computations
- `scipy` for signal processing validation
**Performance Implications**: Post-processing analysis without affecting runtime
**HPC Relevance**: Standard tools for performance analysis in scientific computing

## DSP Concepts

### STFT (Short-Time Fourier Transform)
The STFT decomposes a time-domain signal into time-frequency representation using:
- **Window Function**: Hann window for minimal spectral leakage
- **Frame Size**: 2048 samples (default) for frequency resolution
- **Hop Size**: 1024 samples (50% overlap) for time resolution
- **Zero Padding**: None (rectangular transform)

### ISTFT (Inverse STFT)
Perfect reconstruction using overlap-add algorithm:
- **Window Compensation**: Normalization by window sum for perfect reconstruction
- **Overlap Handling**: 50% overlap reconstruction with proper phase alignment
- **Boundary Conditions**: Zero-padding for edge effects

### Spectral Processing
- **Magnitude Spectrum**: `|X[k]|` for energy distribution analysis
- **Power Spectrum**: `|X[k]|²` for energy conservation
- **Phase Information**: Preserved for reconstruction quality

### Speech-Band Filtering
Frequency-domain filtering with configurable cutoffs:
- **Low Cutoff**: 80 Hz (removes infrasound)
- **High Cutoff**: 8000 Hz (telephone bandwidth)
- **Filter Response**: Sharp cutoff with minimal ripple

### Feature Extraction
- **RMS Energy**: Root mean square for signal power
- **ZCR**: Zero crossing rate for voiced/unvoiced detection
- **Spectral Centroid**: Center of mass of spectrum
- **Spectral Bandwidth**: Spread around centroid
- **Spectral Flatness**: Tonality measure (noise vs tone)
- **MFCC**: 13 mel-frequency cepstral coefficients using:
  - Mel filterbank (26 filters)
  - Log compression
  - Discrete cosine transform

### Preprocessing
- **Normalization**: Peak normalization to [-1, 1] range
- **DC Removal**: High-pass filtering at 0 Hz
- **Pre-emphasis**: `y[n] = x[n] - 0.97*x[n-1]` for high-frequency boost

## HPC Concepts

### Shared-Memory Parallelism (OpenMP)
- **Thread Creation**: `#pragma omp parallel` regions
- **Work Distribution**: `#pragma omp for schedule(dynamic)` for load balancing
- **Data Sharing**: Private variables for thread-local buffers
- **Synchronization**: Implicit barriers at parallel region ends

### Distributed-Memory Parallelism (MPI)
- **Process Management**: Rank-based process identification
- **Communication**: Non-blocking receives with request handling
- **Collective Operations**: `MPI_Allreduce` for global statistics
- **Load Balancing**: Size-based task assignment

### Hybrid Parallelism
- **MPI + OpenMP**: Distributed nodes with shared-memory cores
- **Scalability**: Thousands of cores across hundreds of nodes
- **Communication**: Minimal inter-node data transfer

### Performance Metrics
- **Speedup**: `T₁/Tₚ` ratio vs sequential baseline
- **Efficiency**: `Speedup/P` normalized by processor count
- **Throughput**: Tasks/second processing rate
- **Scalability**: Performance vs problem size analysis

### Load Balancing
- **Static**: Round-robin for equal-sized tasks
- **Dynamic**: Size-aware greedy assignment for variable tasks
- **Quality Metric**: Minimize maximum rank completion time

### Memory Management
- **Flat Buffers**: Contiguous storage for cache efficiency
- **Thread-Local**: FFT plans isolated per thread
- **Distributed**: Per-rank memory allocation

## Directory Structure

```
audio-project/
├── LICENSE                 # MIT License
├── Makefile               # GNU Make build system with VPATH
├── README.md              # This documentation
├── main.cpp               # Command-line interface and mode dispatch
├── include/               # Header files
│   ├── audio.h           # WAV I/O declarations
│   ├── audio_features.h  # Feature extraction API
│   ├── compiler_workaround.h  # Platform compatibility
│   ├── dataset_manager.h # Dataset scanning and task generation
│   ├── dsp.h             # DSP algorithm declarations
│   ├── fft_utils.h       # FFTW wrapper with plan pooling
│   ├── parallel.h        # Parallel pipeline declarations
│   └── timing.h          # High-resolution timing utilities
├── src/                  # Implementation files
│   ├── dataset_manager.cpp    # File scanning and task creation
│   ├── io/                     # Audio I/O
│   │   ├── wav_reader.cpp     # WAV file reading
│   │   └── wav_writer.cpp     # WAV file writing
│   ├── dsp/                    # Digital signal processing
│   │   ├── preprocessing.cpp  # Audio preprocessing
│   │   ├── stft.cpp           # STFT implementation
│   │   ├── istft.cpp          # ISTFT implementation
│   │   ├── filters.cpp        # Frequency filtering
│   │   └── feature_extraction.cpp  # MFCC and spectral features
│   ├── parallel/               # Parallel processing
│   │   ├── omp_pipeline.cpp   # OpenMP implementation
│   │   ├── mpi_pipeline.cpp   # MPI implementation
│   │   └── chunk_manager.cpp  # Data chunking utilities
│   ├── sequential/            # Sequential baseline
│   │   └── sequential_pipeline.cpp  # Reference implementation
│   └── utils/                 # Utilities
│       ├── timing.cpp         # Timing implementation
│       ├── logger.cpp         # Logging utilities
│       └── helpers.cpp        # Helper functions
├── scripts/               # Python analysis tools
│   ├── benchmark.py       # Performance analysis
│   ├── plot_results.py    # Visualization scripts
│   └── visualize_audio.py # Audio signal plotting
├── data/                 # Sample datasets
│   └── chunks/           # Processing output
├── build/                # Build artifacts (generated)
└── test_*.cpp            # Test programs
```

## Build Instructions

### Prerequisites (Ubuntu/Debian)

```bash
# Update package lists
sudo apt update

# Install build tools
sudo apt install build-essential cmake git

# Install FFTW3
sudo apt install libfftw3-dev libfftw3-single3

# Install libsndfile
sudo apt install libsndfile1-dev

# Install OpenMPI
sudo apt install libopenmpi-dev openmpi-bin

# Install Python dependencies
pip3 install numpy pandas matplotlib scipy
```

### Build Process

```bash
# Clone repository
git clone <repository-url>
cd audio-project

# Build the project
make clean && make

# Verify build
ls -la audio_app
```

### Execution Examples

#### Sequential Mode
```bash
./audio_app --dataset data/ --mode sequential --output results/
```

#### OpenMP Mode
```bash
./audio_app --dataset data/ --mode omp --output results/ --threads 8
```

#### MPI Mode
```bash
mpirun -np 4 ./audio_app --dataset data/ --mode mpi --output results/
```

#### Hybrid Mode (Planned)
```bash
mpirun -np 2 ./audio_app --dataset data/ --mode hybrid --output results/ --threads 4
```

## Dataset Processing

### Input Structure
```
dataset/
├── audio1.wav
├── audio2.wav
├── subdirectory/
│   └── audio3.wav
└── ...
```

### Output Hierarchy
```
results/
├── mode/                    # sequential, omp, mpi, hybrid
│   ├── rank_0/             # MPI rank directory
│   │   ├── results/        # Processed WAV files
│   │   │   └── audio1.wav
│   │   └── features/       # Feature CSV files
│   │       └── audio1.csv
│   └── rank_1/             # Additional ranks
└── benchmark_results_mode_rank_*.csv  # Per-rank benchmarks
```

### Task Generation
- Recursive WAV file discovery
- Automatic output path generation
- Size-based load balancing for MPI
- Directory structure preservation

## Benchmarking System

### Metrics Collected
- **Runtime**: Wall-clock time per task
- **File Size**: Input file size for throughput analysis
- **Throughput**: Files/second processing rate
- **Efficiency**: Parallel efficiency vs ideal scaling

### Output Formats
- **CSV Files**: Per-rank benchmark results with timing data
- **Summary Files**: Global statistics across all ranks
- **Visualization**: Python scripts for scalability plots

### Analysis Scripts
```bash
# Generate performance plots
python3 scripts/plot_results.py results/

# Analyze benchmark data
python3 scripts/benchmark.py results/benchmark_results_mpi_rank_*.csv
```

## Current Implementation State

### Fully Implemented ✅
- **Sequential Pipeline**: Complete reference implementation
- **OpenMP Pipeline**: Thread-parallel STFT, filtering, feature extraction
- **Basic MPI Pipeline**: File-level distribution with load balancing
- **DSP Algorithms**: STFT/ISTFT, filtering, MFCC feature extraction
- **Audio I/O**: WAV reading/writing with libsndfile
- **Benchmarking**: Timing, CSV output, basic analysis
- **Dataset Management**: Recursive scanning, task generation

### Partially Implemented 🚧
- **MPI Communication**: Non-blocking receives implemented, but hybrid mode incomplete
- **Load Balancing**: Size-aware distribution working, but dynamic rebalancing absent
- **Memory Optimization**: Flat buffers implemented, but memory pooling incomplete

### Planned Features 📋
- **Hybrid MPI + OpenMP**: Full integration of distributed + shared memory
- **Advanced Load Balancing**: Dynamic task migration based on runtime
- **Memory Pooling**: Custom allocators for FFT buffers
- **GPU Acceleration**: CUDA/OpenCL FFT integration
- **Real-time Processing**: Streaming audio pipeline
- **Advanced DSP**: Wavelet transforms, neural network features

### Known Limitations ⚠️
- **Hybrid Mode**: Command-line parsing exists, implementation incomplete
- **Memory Usage**: Per-rank loading of all assigned files (no streaming)
- **Scalability**: Tested up to 100 files, not validated at 1000+ scale
- **Error Handling**: Basic error checking, no recovery mechanisms
- **Configuration**: Hard-coded parameters (FFT size, filter cutoffs)

## Project Evolution

### Phase 1: Sequential DSP (Completed)
- Basic WAV I/O implementation
- Sequential STFT/ISTFT pipeline
- Fundamental DSP algorithms
- Reference performance baseline

### Phase 2: OpenMP Parallelization + Dataset Processing (Completed)
- OpenMP thread-level parallelism
- Dataset scanning and task generation
- Parallel STFT with thread-local FFT plans
- Shared-memory optimization

### Phase 3: MPI Dataset Distribution (Partially Completed)
- MPI process-level parallelism
- File-level task distribution
- Non-blocking communication
- Size-aware load balancing

### Phase 4: Hybrid MPI + OpenMP (Planned)
- Nested parallelism integration
- MPI ranks with OpenMP threads
- NUMA-aware memory allocation
- Advanced synchronization

### Phase 5: Optimization + Research-Grade System (In Progress)
- FFT plan precomputation
- Memory pooling and reuse
- Advanced benchmarking
- Scalability validation

## Engineering Challenges

### STFT Overlap-Add Complexity
- **Problem**: Maintaining phase coherence across overlapping frames
- **Solution**: Careful window normalization and boundary handling
- **Impact**: Perfect reconstruction quality

### MPI Communication Overhead
- **Problem**: Latency in rank 0 bottleneck for result collection
- **Solution**: Distributed I/O with MPI_Allreduce
- **Impact**: Improved scalability for large file counts

### FFT Memory Usage
- **Problem**: Large FFT sizes require significant memory
- **Solution**: Thread-local plan pooling and buffer reuse
- **Impact**: Reduced memory footprint and allocation overhead

### Synchronization Issues
- **Problem**: Race conditions in shared data structures
- **Solution**: Mutex protection for FFT plan pool
- **Impact**: Thread-safe parallel execution

### Scaling Limitations
- **Problem**: Memory growth with dataset size
- **Solution**: Streaming processing and memory pooling
- **Impact**: Support for massive datasets

## Research Value

### HPC Contributions
- Demonstrates hybrid parallelism patterns for signal processing
- Provides benchmark data for parallel algorithm efficiency
- Shows scalability techniques for distributed DSP workloads

### AI Infrastructure Value
- Foundation for distributed feature extraction pipelines
- Scalable audio preprocessing for machine learning
- Performance baselines for GPU-accelerated audio processing

### DSP Research Value
- Research-grade MFCC implementation with mel filterbanks
- Optimized STFT/ISTFT for real-time applications
- Comparative analysis of parallel DSP algorithms

### Distributed Systems Relevance
- Load balancing strategies for heterogeneous workloads
- Communication optimization for HPC applications
- Scalability patterns for scientific computing

## Conclusion

This framework represents a comprehensive implementation of modern HPC techniques applied to audio signal processing. By combining shared-memory and distributed-memory parallelism with optimized DSP algorithms, it provides both educational value for understanding parallel computing concepts and practical utility for research in distributed audio processing.

The modular architecture enables easy extension to new algorithms and execution models, while the comprehensive benchmarking system supports rigorous performance analysis. As a research-oriented project, it demonstrates the intersection of high-performance computing, digital signal processing, and distributed systems engineering.

---

*Developed as an academic HPC project demonstrating advanced parallel computing concepts in audio signal processing.*

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