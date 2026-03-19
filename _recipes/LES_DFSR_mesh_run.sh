#!/bin/bash -l

#SBATCH --job-name=DFSR_test_empty_domain
#SBATCH -N 2                              # <-- number of nodes (template style)
#SBATCH --ntasks-per-node=15              # <-- MPI ranks per node (cores)
#SBATCH -t 7-00:00:00                       # <-- walltime (template style)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.cunningham2@ucdconnect.ie
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Template-style fabric/provider hint for Intel MPI (OFI/libfabric)
export FI_PROVIDER=verbs

# -----------------------
# Modules / OpenFOAM env
# -----------------------
module purge
module load openfoam/2206-gcc-11.5.0-523wsuk
module load python/3.11.9-gcc-11.5.0-lcos74i

VENV=/home/people/20397873/venvs/LES
source "$VENV/bin/activate"

# Force Intel MPI launcher (avoid any accidental OpenMPI mpiexec if other modules are loaded)
MPIEXEC="/opt/software/el9/spack/0.23/opt/spack/linux-rhel9-skylake_avx512/gcc-11.5.0/intel-oneapi-mpi-2021.14.0-hjmtgxafhbqbldusqwow34lif2zgtcq6/mpi/2021.14/bin/mpiexec"

# Use SLURM's allocated total MPI task count (fully flexible with nodes/cores)
NTASKS="${SLURM_NTASKS:?SLURM_NTASKS not set}"
echo "Using NTASKS=$NTASKS"

# DFSR runs only on one node
DFSR_NTASKS="${SLURM_NTASKS_PER_NODE:-15}"
echo "Using DFSR_NTASKS=$DFSR_NTASKS"

# -----------------------
# Your user OpenFOAM tree (where DFSRTurb is installed)
# -----------------------
export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/$USER-v2206"
export FOAM_USER_APPBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib"

# Ensure your utility + user libs are found at runtime
export PATH="$FOAM_USER_APPBIN:$PATH"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:${LD_LIBRARY_PATH:-}"

# Optional: keep spack compiler wrappers out of the way (mainly matters when compiling, harmless here)
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/opt/software/el9/spack/0.23/lib/spack/env' | paste -sd ':' -)

# Template-style MPI sanity check (run something simple in parallel)
"$MPIEXEC" -n "$NTASKS" hostname

# -----------------------
# Case directory
# -----------------------
export CASE_DIR="${SLURM_SUBMIT_DIR}"
cd "$CASE_DIR"

LOGDIR="$CASE_DIR/log"
mkdir -p "$LOGDIR"

PYTHONLOGDIR="$CASE_DIR/log/python"
mkdir -p "$PYTHONLOGDIR"

echo "Running in case: $CASE_DIR"
echo "OpenFOAM version: $WM_PROJECT_VERSION"
echo "WM_OPTIONS: $WM_OPTIONS"

# -----------------------
# Host information
# -----------------------
FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
DFSR_HOSTFILE="$LOGDIR/dfsr_hostfile"

echo "$FIRST_NODE" > "$DFSR_HOSTFILE"

echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "DFSR will run only on: $FIRST_NODE"
echo "Testing one-node DFSR launcher:"
cat "$DFSR_HOSTFILE"
"$MPIEXEC" -f "$DFSR_HOSTFILE" -ppn "$DFSR_NTASKS" -n "$DFSR_NTASKS" hostname

# -----------------------
# Mesh generation (serial)
# -----------------------
echo "=== blockMesh (serial) ==="
blockMesh > "$LOGDIR/blockMesh.log" 2>&1

#echo "=== surfaceFeatureExtract (serial) ==="
#surfaceFeatureExtract > "$LOGDIR/surfaceFeatureExtract.log" 2>&1

checkMesh > "$LOGDIR/checkMesh.log" 2>&1

windlespyDir="/home/people/20397873/LES/windlespy/_recipes"

MAX_INLET_CAL_ITERS=2
MAX_CAL_ITERS=5
MAX_LES_EXTENSIONS=2

# =========================================================
# STAGE 1: DFSR inlet calibration on ONE NODE ONLY
# =========================================================

echo "=== Preparing decomposition for DFSR on one node ==="
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$DFSR_NTASKS"
decomposePar -force > "$LOGDIR/decomposePar.dfsr.log" 2>&1

for cal_iter in $(seq 1 $MAX_INLET_CAL_ITERS); do
    echo "=== Inlet calibration iteration $cal_iter ==="

    "$MPIEXEC" -f "$DFSR_HOSTFILE" -ppn "$DFSR_NTASKS" -n "$DFSR_NTASKS" DFSRTurb -parallel \
    > "$LOGDIR/DFSRTurb_Inlet_${cal_iter}.log" 2>&1

    if python "$windlespyDir/_dfsrInletCalibration.py" > "$PYTHONLOGDIR/_dfsrInletCalibration_cal${cal_iter}.log" 2>&1; then
        cal_status=0
    else
        cal_status=$?
    fi

    if [ "$cal_status" -eq 0 ]; then
        echo "Inlet calibration complete"
        break
    elif [ "$cal_status" -eq 1 ]; then
        echo "Updated inlet profile and continuing calibration"
    else
        echo "Calibration update failed"
        exit 1
    fi
done


if ! python "$windlespyDir/_dfsrLesInitialise.py" > "$PYTHONLOGDIR/_dfsrLesInitialise.log" 2>&1; then
    echo "LES initialization failed"
    exit 1
fi

eval "$(python - <<'PY'
import json
import os

case_dir = os.environ["CASE_DIR"]
json_path = f"{case_dir}/log/downstreamCalibration/sim_init.json"

with open(json_path) as f:
    d = json.load(f)

print(f'TIME_CHUNK="{d["time_chunk"]}"')
print(f'initial_simulation_duration="{d["initial_sim_duration"]}"')
PY
)"

CURRENT_ENDTIME=$initial_simulation_duration

# =========================================================
# Reconstruct DFSR result, then redecompose for full solver
# =========================================================

echo "=== Reconstructing DFSR decomposition ==="
reconstructPar > "$LOGDIR/reconstructPar.dfsr.log" 2>&1 || true

echo "=== Preparing decomposition for full solver on all ranks ==="
foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$NTASKS"
decomposePar -force > "$LOGDIR/decomposePar.solver.log" 2>&1

# =========================================================
# STAGE 2: LES / calibration on ALL ALLOCATED RANKS
# =========================================================

for cal_iter in $(seq 1 $MAX_CAL_ITERS); do
    echo "=== Calibration iteration $cal_iter ==="

    # Set up case with current inlet profile here
    # clean / reset empty-domain run if needed
    CURRENT_ENDTIME=$initial_simulation_duration

    rm -rf postProcessing
    foamListTimes -rm
    foamDictionary system/controlDict -entry startTime -set 0
    foamDictionary system/controlDict -entry startFrom -set startTime
    foamDictionary system/controlDict -entry endTime -set "$CURRENT_ENDTIME"

    les_ready=0

    for les_iter in $(seq 1 $MAX_LES_EXTENSIONS); do
        echo "=== LES extension $les_iter for calibration iteration $cal_iter ==="
        
        "$MPIEXEC" -n "$NTASKS" pisoFoam -parallel > "log/pisoFoam_cal${cal_iter}_les${les_iter}.log" 2>&1 || exit 1

        # Reconstruct or sample as needed
        if python "$windlespyDir/_dfsrConvergenceCheck.py" > "$PYTHONLOGDIR/_dfsrConvergenceCheck_cal${cal_iter}_les${les_iter}.log" 2>&1; then
            les_status=0
        else
            les_status=$?
        fi

        if [ "$les_status" -eq 0 ]; then
            echo "LES statistics are mature enough"
            les_ready=1
            break
        elif [ "$les_status" -eq 1 ]; then
            echo "LES not yet converged; extending run"
            CURRENT_ENDTIME=$(python - <<PY
current = float("$CURRENT_ENDTIME")
chunk = float("$TIME_CHUNK")
print(current+chunk)
PY
)
            foamDictionary system/controlDict -entry startFrom -set latestTime
            foamDictionary system/controlDict -entry endTime -set "$CURRENT_ENDTIME"
        else
            echo "LES convergence check failed"
            exit 1
        fi
    done

    if [ "$les_ready" -ne 1 ]; then
        echo "LES failed to reach acceptable convergence within allowed extensions"
        exit 1
    fi

    if python "$windlespyDir/_dfsrDownstreamCalibration.py" > "$PYTHONLOGDIR/_dfsrDownstreamCalibration${cal_iter}.log" 2>&1; then
        cal_status=0
    else
        cal_status=$?
    fi

    if [ "$cal_status" -eq 0 ]; then
        echo "Downstream calibration complete"
        break
    elif [ "$cal_status" -eq 1 ]; then
        echo "Updating inlet profile and continuing calibration"

        echo "=== Reconstructing full decomposition before DFSR update ==="
        reconstructPar > "$LOGDIR/reconstructPar.before_dfsr_${cal_iter}.log" 2>&1 || true

        echo "=== Re-decomposing for DFSR one-node update ==="
        foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$DFSR_NTASKS"
        decomposePar -force > "$LOGDIR/decomposePar.dfsr_update_${cal_iter}.log" 2>&1

        "$MPIEXEC" -f "$DFSR_HOSTFILE" -ppn "$DFSR_NTASKS" -n "$DFSR_NTASKS" DFSRTurb -parallel \
    > "$LOGDIR/DFSRTurb_Downstream_${cal_iter}.log" 2>&1

        echo "=== Reconstructing DFSR update decomposition ==="
        reconstructPar > "$LOGDIR/reconstructPar.after_dfsr_${cal_iter}.log" 2>&1 || true

        echo "=== Re-decomposing for full solver again ==="
        foamDictionary system/decomposeParDict -entry numberOfSubdomains -set "$NTASKS"
        decomposePar -force > "$LOGDIR/decomposePar.solver_update_${cal_iter}.log" 2>&1

    else
        echo "Calibration update failed"
        exit 1
    fi
done

echo "Job finished on $(date)"

