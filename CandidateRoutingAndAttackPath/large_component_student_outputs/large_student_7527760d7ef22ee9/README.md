# Large component student run

- train: 4 patched endpoints only
- test: 2 patched + 2 clean endpoints
- train/test underlying scenes are disjoint
- blind_support receives no teacher coordinates, target box, class, or clean endpoint at inference

mode,feature_set,correction_scale,blind_selector,patched_n,clean_n,baseline_target_rate,corrected_target_rate,hidden_n,hidden_recovered_n,baseline_lost_n,patched_conf_gain,clean_full_detection_f1,clean_target_change_rate,support_recall,support_energy_recall
blind_oracle_values,teacher,1.0,spatial,2,0,1.0,1.0,0,0,0,0.0022355616092681885,,,0.008064516129032258,0.018668542826312817
blind_support,functional,0.25,spatial,2,2,1.0,1.0,0,0,0,3.853440284729004e-05,1.0,0.0,0.008064516129032258,0.018668542826312817
blind_support,functional,0.5,spatial,2,2,1.0,1.0,0,0,0,7.292628288269043e-05,1.0,0.0,0.008064516129032258,0.018668542826312817
blind_support,functional,0.75,spatial,2,2,1.0,1.0,0,0,0,0.00010317564010620117,1.0,0.0,0.008064516129032258,0.018668542826312817
blind_support,functional,1.0,spatial,2,2,1.0,1.0,0,0,0,0.0001297295093536377,1.0,0.0,0.008064516129032258,0.018668542826312817
known_support,functional,0.25,teacher,2,2,1.0,1.0,0,0,0,-0.0073080360889434814,1.0,0.0,1.0,1.0
known_support,functional,0.5,teacher,2,2,1.0,1.0,0,0,0,-0.01759117841720581,1.0,0.0,1.0,1.0
known_support,functional,0.75,teacher,2,2,1.0,1.0,0,0,0,-0.030178338289260864,1.0,0.0,1.0,1.0
known_support,functional,1.0,teacher,2,2,1.0,1.0,0,0,0,-0.045923084020614624,1.0,0.0,1.0,1.0
oracle_component,teacher,1.0,teacher,2,0,1.0,1.0,0,0,0,0.13902762532234192,,,1.0,1.0
