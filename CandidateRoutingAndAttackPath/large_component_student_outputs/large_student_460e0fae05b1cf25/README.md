# Large component student run

- train: 4 patched endpoints only
- test: 2 patched + 2 clean endpoints
- train/test underlying scenes are disjoint
- blind_support receives no teacher coordinates, target box, class, or clean endpoint at inference

mode,feature_set,patched_n,clean_n,baseline_target_rate,corrected_target_rate,hidden_n,hidden_recovered_n,baseline_lost_n,patched_conf_gain,clean_full_detection_f1,clean_target_change_rate,support_recall,support_energy_recall
blind_support,activation,2,2,1.0,1.0,0,0,0,-0.001971036195755005,1.0,0.0,0.004725302419354839,0.047936756620477305
blind_support,combined,2,2,1.0,1.0,0,0,0,-0.0017206966876983643,1.0,0.0,0.004725302419354839,0.047936756620477305
blind_support,functional,2,2,1.0,1.0,0,0,0,-0.0017909705638885498,1.0,0.0,0.004725302419354839,0.047936756620477305
blind_support,local,2,2,1.0,1.0,0,0,0,-0.0007069706916809082,1.0,0.0,0.004725302419354839,0.047936756620477305
known_support,activation,2,2,1.0,1.0,0,0,0,-0.057829707860946655,1.0,0.0,1.0,1.0
known_support,combined,2,2,1.0,1.0,0,0,0,-0.061339765787124634,1.0,0.0,1.0,1.0
known_support,functional,2,2,1.0,1.0,0,0,0,-0.045923084020614624,1.0,0.0,1.0,1.0
known_support,local,2,2,1.0,1.0,0,0,0,-0.06926429271697998,1.0,0.0,1.0,1.0
