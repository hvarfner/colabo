for SEED in 1 2 3 4 5 6 7 8 9 10
    for RFF in 128 256 512 1024 2048 4096 8192
        sbatch start_mcerror.sh $SEED $RFF 