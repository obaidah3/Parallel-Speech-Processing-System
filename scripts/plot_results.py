#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_speedup_efficiency():
    df = pd.read_csv('data/results/benchmark_results.csv')

    # Plot speedup
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    omp_data = df[df['Mode'] == 'OpenMP']
    plt.plot(omp_data['Threads'], omp_data['Speedup'], 'o-', label='OpenMP', color='blue')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('OpenMP Speedup')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    mpi_data = df[df['Mode'] == 'MPI']
    plt.plot(mpi_data['Processes'], mpi_data['Speedup'], 's-', label='MPI', color='red')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('MPI Speedup')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    hybrid_data = df[df['Mode'] == 'Hybrid']
    for nt in hybrid_data['Threads'].unique():
        data = hybrid_data[hybrid_data['Threads'] == nt]
        plt.plot(data['Processes'], data['Speedup'], '^-', label=f'Hybrid (T={nt})', marker='^')
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Hybrid Speedup')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('data/results/speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency():
    df = pd.read_csv('data/results/benchmark_results.csv')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    omp_data = df[df['Mode'] == 'OpenMP']
    plt.plot(omp_data['Threads'], omp_data['Efficiency'], 'o-', label='OpenMP', color='blue')
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency')
    plt.title('OpenMP Efficiency')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    mpi_data = df[df['Mode'] == 'MPI']
    plt.plot(mpi_data['Processes'], mpi_data['Efficiency'], 's-', label='MPI', color='red')
    plt.xlabel('Number of Processes')
    plt.ylabel('Efficiency')
    plt.title('MPI Efficiency')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    hybrid_data = df[df['Mode'] == 'Hybrid']
    for nt in hybrid_data['Threads'].unique():
        data = hybrid_data[hybrid_data['Threads'] == nt]
        plt.plot(data['Processes'], data['Efficiency'], '^-', label=f'Hybrid (T={nt})')
    plt.xlabel('Number of Processes')
    plt.ylabel('Efficiency')
    plt.title('Hybrid Efficiency')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('data/results/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_runtime_comparison():
    df = pd.read_csv('data/results/benchmark_results.csv')

    plt.figure(figsize=(12, 6))

    # Group by mode and plot average time
    modes = df['Mode'].unique()
    avg_times = [df[df['Mode'] == mode]['Time(s)'].mean() for mode in modes]

    bars = plt.bar(modes, avg_times, color=['green', 'blue', 'red', 'purple'])
    plt.xlabel('Execution Mode')
    plt.ylabel('Average Runtime (s)')
    plt.title('Runtime Comparison Across Modes')
    plt.grid(True, axis='y')

    # Add value labels on bars
    for bar, time in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time:.3f}s', ha='center', va='bottom')

    plt.savefig('data/results/runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scalability():
    df = pd.read_csv('data/results/benchmark_results.csv')

    plt.figure(figsize=(10, 6))

    # Plot all speedup curves
    for mode in df['Mode'].unique():
        mode_data = df[df['Mode'] == mode]
        if mode == 'OpenMP':
            plt.plot(mode_data['Threads'], mode_data['Speedup'], 'o-', label=f'{mode} (P=1)', linewidth=2)
        elif mode == 'MPI':
            plt.plot(mode_data['Processes'], mode_data['Speedup'], 's-', label=f'{mode} (T=1)', linewidth=2)
        elif mode == 'Hybrid':
            for nt in sorted(mode_data['Threads'].unique()):
                data = mode_data[mode_data['Threads'] == nt]
                plt.plot(data['Processes'], data['Speedup'], '^-', 
                        label=f'{mode} (T={nt})', linewidth=2, marker='^')

    plt.xlabel('Number of Processing Units')
    plt.ylabel('Speedup')
    plt.title('Scalability Analysis: Speedup vs Processing Units')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')

    plt.savefig('data/results/scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_speedup_efficiency()
    plot_efficiency()
    plot_runtime_comparison()
    plot_scalability()