#!/usr/bin/env bash
####### Mail Notify / Job Name / Comment #######
#SBATCH --job-name="matmul_test"

####### Partition #######
#SBATCH --partition=all

####### Ressources #######
#SBATCH --time=0-00:02:00

####### Node Info #######
#SBATCH --exclusive
#SBATCH --nodes=1

####### Output #######
#SBATCH --output=%x.out.%j
#SBATCH --error=%x.err.%j

echo "=== Test 1: verbose, 4 procs ==="
mpirun -np 4 ./matmul 4 42 1

echo ""
echo "=== Test 2: non-verbose, 4 procs ==="
mpirun -np 4 ./matmul 4 42 0

echo ""
echo "=== Test 3: verbose, 1 proc ==="
mpirun -np 1 ./matmul 4 42 1
