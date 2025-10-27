# #!/bin/bash

# prepare temporary storage space
rm -r /app/DATASET/temp_for_dataset
mkdir /app/DATASET/temp_for_dataset

# get which images were part of test/train/val partitions
( cd /app/CLIP-LoRA-Species-Mapping/ && pwd && python sim_main.py \
	--root_path /app/DATASET/Rajasthan_Zones_v5/merged_human_labels/ \
	--dataset gsv \
	--seed 1 \
	--shots 100000000000000 \
	--backbone bio_clip \
	--n_iters 50 \
	--subsample_amount 2000 \
	--save_path /app/DATASET/temp_for_dataset )

# create COCO file for human trained data and put the rest of the images in a file for machine annotation
( cd /app/CLIP-LoRA-Species-Mapping/ && pwd && python split_test.py \
	--tranches /app/DATASET/Rajasthan_Zones_v5/merged_human_labels/instances_default.json \
	--images /app/DATASET/Rajasthan_Zones_v5/merged_human_labels/images/ \
	--output /app/DATASET/temp_for_dataset/ \
	--tested_on /app/DATASET/temp_for_dataset/images_tested_on.txt \
	--trained_on /app/DATASET/temp_for_dataset/images_trained_on.txt )

# get machine labels
( cd /app/CLIP-LoRA-Species-Mapping/ && pwd && python image_annotater.py \
	--lora_path "/app/DATASET/figure/fig2_bioclip/subsample_2000/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt" \
	--image_dir /app/DATASET/Rajasthan_Zones_v5/pure_machine_labels/images/ \
	--output_file /app/DATASET/temp_for_dataset/pure_machine.json \
	--encoder both \
	--backbone bio_clip )

( cd /app/CLIP-LoRA-Species-Mapping/ && pwd && python image_annotater.py \
	--lora_path "/app/DATASET/figure/fig2_bioclip/subsample_2000/bio_clip/gsv/100000000000000shots/seed1/lora_weights.pt" \
	--image_dir /app/DATASET/temp_for_dataset/reject/ \
	--output_file /app/DATASET/temp_for_dataset/human_relabeled.json \
	--encoder both \
	--backbone bio_clip )

# merge COCO files
( cd /app/CLIP-LoRA-Species-Mapping/ && pwd && python coco_merger.py \
	--tranches /app/DATASET/temp_for_dataset/train_human.json /app/DATASET/temp_for_dataset/pure_machine.json /app/DATASET/temp_for_dataset/human_relabeled.json \
	--output /app/DATASET/temp_for_dataset/merged_for_train_val.json )

# clean out previous iteration just in case
# rm -r /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/species_classification_vectors/

# create train/val split
( cd /app/SpeciesMapping/multi_level_dataset_scripts/ && pwd && python integrate_species_classification.py \
	--dataset_dir /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/ \
	--coco_file /app/DATASET/temp_for_dataset/merged_for_train_val.json \
	--output_dir /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/ \
	--ratio_train 0.9 \
	--ratio_val 0.1 )

# ensure we use the right test set
# cp /app/DATASET/Rajasthan_Zones_v5/merged_human_labels/species_classification_vectors/data_splits/test.csv /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/species_classification_vectors/data_splits/test.csv

# prep the final COCO file
( cd /app/CLIP-LoRA-Species-Mapping/ && pwd && python coco_merger.py \
	--tranches /app/DATASET/temp_for_dataset/merged_for_train_val.json /app/DATASET/temp_for_dataset/test.json \
	--output /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/instances_default.json )

# run s2 training
python train_one_species.py \
	--dataset_dir /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/ \
	--test_dataset_dir /app/DATASET/Rajasthan_Zones_v5/merged_machine_labels/ \
	--save_dir "/app/DATASET/figures/fig3_2000shot/Azadiractha_Indica/" \
	--species_id 24 \
	--hidden_dims 512 256 128 \
	--learning_rate 0.0001