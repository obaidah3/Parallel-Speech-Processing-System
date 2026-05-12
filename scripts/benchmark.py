#!/usr/bin/env python3
import subprocess
import time
import csv
import os

def run_command(cmd, env=None):
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    end = time.time()
    return end - start, result.returncode, result.stdout, result.stderr

def run_sequential():
    print("Running sequential benchmark...")
    time_taken, code, out, err = run_command("./audio_app sequential")
    return time_taken

def run_omp(threads):
    print(f"Running OpenMP benchmark with {threads} threads...")
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(threads)
    time_taken, code, out, err = run_command("./audio_app omp", env=env)
    return time_taken

def run_mpi(processes):
    print(f"Running MPI benchmark with {processes} processes...")
    cmd = f"mpirun -np {processes} ./audio_app mpi"
    time_taken, code, out, err = run_command(cmd)
    return time_taken

def run_hybrid(processes, threads):
    print(f"Running Hybrid benchmark with {processes} processes and {threads} threads...")
    cmd = f"mpirun -np {processes} ./audio_app hybrid {processes} {threads}"
    time_taken, code, out, err = run_command(cmd)
    return time_taken

def main():
    results = []

    # Sequential
    seq_time = run_sequential()
    results.append(['Sequential', 1, 1, seq_time])

    # OpenMP
    for nt in [1, 2, 4, 8]:
        omp_time = run_omp(nt)
        results.append(['OpenMP', 1, nt, omp_time])

    # MPI
    for np in [1, 2, 4, 8]:
        mpi_time = run_mpi(np)
        results.append(['MPI', np, 1, mpi_time])

    # Hybrid
    for np in [2, 4]:
        for nt in [2, 4]:
            hybrid_time = run_hybrid(np, nt)
            results.append(['Hybrid', np, nt, hybrid_time])

    # Save results
    with open('data/results/benchmark_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mode', 'Processes', 'Threads', 'Time(s)', 'Speedup', 'Efficiency'])
        
        seq_time = results[0][3]  # Sequential time
        for result in results:
            mode, procs, threads, time_taken = result
            speedup = seq_time / time_taken if time_taken > 0 else 0
            efficiency = speedup / (procs * threads) if procs * threads > 0 else 0
            writer.writerow([mode, procs, threads, time_taken, speedup, efficiency])

    print("Benchmarking complete. Results saved to data/results/benchmark_results.csv")

    # Print summary table
    print("\nBenchmark Summary:")
    print("-" * 70)
    print(f"{'Mode':<10} {'Processes':<10} {'Threads':<8} {'Time(s)':<8} {'Speedup':<8} {'Efficiency':<10}")
    print("-" * 70)
    
    seq_time = results[0][3]
    for result in results:
        mode, procs, threads, time_taken = result
        speedup = seq_time / time_taken if time_taken > 0 else 0
        efficiency = speedup / (procs * threads) if procs * threads > 0 else 0
        print(f"{mode:<10} {procs:<10} {threads:<8} {time_taken:<8.3f} {speedup:<8.2f} {efficiency:<10.3f}")

if __name__ == "__main__":
    main()