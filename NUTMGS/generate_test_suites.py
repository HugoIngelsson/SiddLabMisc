import json

# from hyperparameter tuning, just kept here for convenience

# learning_rate = [1e-2, 1e-3, 1e-4]    
# weight_decay = [1e-3, 1e-4, 1e-5]
# hidden_dims = [[512, 256, 128], [256, 128, 64], [512, 256, 128, 64], [1028, 512, 156, 128]]
# dropout_rate = [0.3, 0.5, 0.7]

# settings = [
#     {
#         "learning_rate": lr,
#         "weight_decay": wd,
#         "hidden_dims": hd,
#         "dropout_rate": dr,
#         "save_dir": f'/DATASET/july25/tune_hyperparams/lr{lr}_wd{wd}_hd{hd}_dr{dr}/'.replace(' ', '_').replace('.', '').replace(',', '')
#     }
#     for lr in learning_rate
#     for wd in weight_decay
#     for hd in hidden_dims
#     for dr in dropout_rate
# ]

# species_id = [24, 177, 193, 147, 195]
# species_map = {
#     24: "Azadirachta indica",
#     177: "Vachellia nilotica",
#     193: "Ailanthus excelsa",
#     147: "Prosopis cineraria",
#     195: "prosopis juliflora"
# }
# num_instances = {
#     24: 15000,
#     177: 13000,
#     193: 2700,
#     147: 5300,
#     195: 3100
# }
# shots = [50000]
# shots.extend(range(100, 1000, 100))
# shots.extend(range(1000, 5000, 500))
# shots.extend(range(5000, 16000, 1000))

# settings = [
#     {
#         "species_id": species,
#         "shots": shot,
#         "save_dir": f'/DATASET/july25/kshot_full_machine_label/{shot}shot/{species_map[species]}/'.replace(' ', '_').replace('.', '').replace(',', '')
#     }
#     for shot in shots
#     for species in species_id if num_instances[species] >= shot or shot == 50000
# ]

# with open('suite.json', 'w') as f:
#     json.dump({"experiments": settings}, f)

species_id = [24, 177, 193, 147, 195]
species_map = {
    24: "Azadirachta indica",
    177: "Vachellia nilotica",
    193: "Ailanthus excelsa",
    147: "Prosopis cineraria",
    195: "prosopis juliflora"
}

num_instances = {
    24: 3320,
    177: 2290,
    193: 950,
    147: 1040,
    195: 420
}

settings = [
    {
        "num_epochs": 100,
        "species_id": species,
        "save_dir": f'/DATASET/july25/figures/final_machine/0/{species_map[species]}/'.replace(' ', '_').replace('.', '').replace(',', ''),
        "keep_shot_ratio": False
    }
    for species in species_id
]

with open('suite5.json', 'w') as f:
    json.dump({"experiments": settings}, f)