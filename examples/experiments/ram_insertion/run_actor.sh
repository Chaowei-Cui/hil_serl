export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path="/home/eai/ccw/hil-serl/examples/experiments/ram_insertion/classifier_ckpt" \
    --actor \