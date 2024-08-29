python main.py \
    --synthetic "synthetic"\
    --datapath './gendata/example-data/X_miss(0.5)_rep(10)_xfull(100).csv'  \
    --labelpath './gendata/example-data/arcs_miss(0.5)_rep(10)_xfull(100).csv' \
    --epoch 5000 --theta_full 1 --theta_miss 1 --pns_gam \
    --chronological_order\
    --must_exist_edges "p0_ms1_m1,p2_ms1_m7;p2_ms1_m7,p7_ms0_m22" \
    --must_delete_edges "p1_ms1_m4,p2_ms3_m9"\
    --use_x_full --use_x_miss\
    --record_aim