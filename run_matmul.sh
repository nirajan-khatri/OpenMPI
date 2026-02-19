#!/usr/bin/env bash
####### Mail Notify / Job Name / Comment #######
#SBATCH --job-name="matmul_speedup"

####### Partition #######
#SBATCH --partition=all

####### Ressources #######
#SBATCH --time=0-00:15:00

####### Node Info #######
#SBATCH --exclusive
#SBATCH --nodes=8

####### Output #######
#SBATCH --output=%x.out.%j
#SBATCH --error=%x.err.%j

# Load modules
module load gcc/14.3.0 openmpi

# Compile
make clean && make

# Parameters
N=8000
SEED=42
VERBOSE=0
RUNS=3

echo "==========================================="
echo "Speedup Measurement: matmul (Row-Block)"
echo "Matrix size: ${N}x${N}, Seed: ${SEED}"
echo "Date: $(date)"
echo "==========================================="
echo ""

# Run with different node counts
for NODES in 1 2 4 6 8; do
    NP=$((NODES * 64))
    echo "--- ${NODES} node(s), ${NP} MPI processes ---"
    for RUN in $(seq 1 $RUNS); do
        echo "Run ${RUN}/${RUNS}:"
        mpirun -np ${NP} --map-by node --bind-to core ./matmul ${N} ${SEED} ${VERBOSE}
    done
    echo ""
done

echo "==========================================="
echo "Speedup measurement complete."
echo "==========================================="
