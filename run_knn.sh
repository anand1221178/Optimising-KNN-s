#!/bin/bash

#-----------------------------------------------------------------------
# SLURM Job Script
#-----------------------------------------------------------------------

# --- Resource Requests ---
#SBATCH --job-name=KNN_OMP_Job   
#SBATCH --partition=stampede       
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=16        
#SBATCH --mem=1M                   
#SBATCH --time=24:00:00            

# --- Output Files ---
#SBATCH --output=knn_omp_job_%j.out 
#SBATCH --error=knn_omp_job_%j.err 

#-----------------------------------------------------------------------
# Job Execution Steps
#-----------------------------------------------------------------------

echo "========================================================"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "========================================================"


echo "Loading required modules..."
module purge
module load gcc

module list

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS set to: $OMP_NUM_THREADS"

WORK_DIR="/home-mscluster/panand/knn"
cd $WORK_DIR
if [ $? -ne 0 ]; then echo "Error changing directory!"; exit 1; fi
echo "Current working directory: $(pwd)"

echo "Compiling the code..."
make clean
make      

if [ $? -ne 0 ]; then echo "Make command failed!"; exit 1; fi
echo "Compilation finished."

echo "Running the KNN executable..."

./knn
if [ $? -ne 0 ]; then echo "Executable failed!"; exit 1; fi
echo "Execution finished."

#-----------------------------------------------------------------------
echo "========================================================"
echo "Job finished at $(date)"
echo "========================================================"
