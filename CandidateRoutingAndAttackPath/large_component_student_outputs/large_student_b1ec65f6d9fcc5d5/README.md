# Large component student run

- train: 200 patched endpoints only
- test: 150 patched + 150 clean endpoints
- train/test underlying scenes are disjoint
- blind_support receives no teacher coordinates, target box, class, or clean endpoint at inference

mode,feature_set,correction_scale,blind_selector,patched_n,clean_n,baseline_target_rate,corrected_target_rate,hidden_n,hidden_recovered_n,baseline_lost_n,patched_conf_gain,clean_full_detection_f1,clean_target_change_rate,support_recall,support_energy_recall
blind_oracle_values,teacher,1.0,spatial,150,0,0.52,0.8266666666666667,72,46,0,0.210192330305775,,,0.07538198598139391,0.22775106212446014
blind_support,functional,0.25,spatial,150,150,0.52,0.5733333333333334,72,8,0,0.019044420719146728,0.9990476190476191,0.0,0.07538198598139391,0.22775106212446014
blind_support,functional,0.5,spatial,150,150,0.52,0.6066666666666667,72,13,0,0.03994609594345093,0.9983068783068784,0.0,0.07538198598139391,0.22775106212446014
blind_support,functional,0.75,spatial,150,150,0.52,0.6466666666666666,72,19,0,0.059279038707415264,0.9946031746031745,0.0,0.07538198598139391,0.22775106212446014
blind_support,functional,1.0,spatial,150,150,0.52,0.6733333333333333,72,23,0,0.07681602664291859,0.9936507936507936,0.0,0.07538198598139391,0.22775106212446014
known_support,functional,0.25,teacher,150,150,0.52,0.6933333333333334,72,26,0,0.09728094786405564,1.0,0.0,1.0,1.0
known_support,functional,0.5,teacher,150,150,0.52,0.74,72,33,0,0.18009063504636288,0.9983492063492063,0.0,1.0,1.0
known_support,functional,0.75,teacher,150,150,0.52,0.76,72,36,0,0.25466618723546464,0.9967373271889401,0.0,1.0,1.0
known_support,functional,1.0,teacher,150,150,0.52,0.82,72,45,0,0.32916658562918505,0.995467485919099,0.0,1.0,1.0
oracle_component,teacher,1.0,teacher,150,0,0.52,1.0,72,72,0,0.4958364394493401,,,1.0,1.0
