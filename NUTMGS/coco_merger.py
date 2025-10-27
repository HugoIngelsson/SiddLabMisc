import json
from pycocotools.coco import COCO
from pathlib import Path
import argparse

def merge_coco_json(tranche_paths, output_file):
    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_id_offset = 0
    existing_category_ids = set()

    for idx, tranche_path in enumerate(tranche_paths):
        file = Path(tranche_path)
        if not file.exists():
            continue

        from contextlib import redirect_stdout
        import os
        with redirect_stdout(open(os.devnull, "w")): # get rid of loading print
            coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset['images']:
            image['id'] += image_id_offset
            merged_annotations['images'].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset['annotations']:
            annotation['id'] += annotation_id_offset
            annotation['image_id'] += image_id_offset
            merged_annotations['annotations'].append(annotation)

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset['categories']:
            if category['id'] not in existing_category_ids:
                category['id'] += category_id_offset
                merged_annotations['categories'].append(category)
                existing_category_ids.add(category['id'])

        image_id_offset = len(merged_annotations['images'])
        annotation_id_offset = len(merged_annotations['annotations'])
        category_id_offset = len(merged_annotations['categories'])

    # Save merged annotations to output file
    with open(output_file, 'w') as f:
        json.dump(merged_annotations, f)

def main():
    parser = argparse.ArgumentParser(description="Smart merge of Rajasthan dataset tranches")
    parser.add_argument("--tranches", nargs="+", required=True,
                       help="Paths to coco files")
    parser.add_argument("--output", default="merged_coco_file.json",
                       help="Output file")
    
    args = parser.parse_args()
    
    merge_coco_json(args.tranches, args.output)

if __name__ == "__main__":
    main()