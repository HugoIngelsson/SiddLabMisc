#!/usr/bin/env python3
"""
Tree Species Classification using OpenAI GPT-5 with Structured Outputs
Processes images from CSV and returns structured JSON responses with scientific names
"""

import os
import json
import base64
import pandas as pd
from pathlib import Path
import time
from typing import Optional
import requests
from datetime import datetime

class TreeSpeciesClassifier:
    """Classify tree species using OpenAI's Responses API with structured outputs"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the classifier
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found. Set it as environment variable or pass it directly.")
        
        self.api_url = "https://api.openai.com/v1/responses"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_structured_output_format(self) -> dict:
        """
        Create JSON schema for structured output
        
        Returns:
            JSON schema dictionary for tree species response
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "tree_species_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "scientific_name": {
                            "type": "string",
                            "description": "The scientific name (genus and species) of the tree species identified in the image"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Confidence level of the classification"
                        },
                        "visible_features": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of visible features used for identification"
                        }
                    },
                    "required": ["scientific_name", "confidence", "visible_features"],
                    "additionalProperties": False
                }
            }
        }
    
    def classify_image(self, image_path: str, model: str = "gpt-4o") -> dict:
        """
        Classify a single tree image using OpenAI Responses API
        
        Args:
            image_path: Path to the image file
            model: Model to use (gpt-4o, gpt-4o-mini, or gpt-5 when available)
            
        Returns:
            Dictionary containing the classification results
        """
        # Read and encode image
        base64_image = self.encode_image_to_base64(image_path)
        
        # Prepare request payload
        payload = {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "input_text",
                            "text": "Identify the tree species in this image and provide the scientific name (genus and species). Analyze visible features such as leaves, bark, branches, and overall tree structure to make your determination."
                        }
                    ]
                }
            ],
            "text": {
                "format": self.create_structured_output_format()
            },
            "temperature": 0.3,  # Lower temperature for more consistent classifications
            "store": False  # Don't store responses for privacy
        }
        
        try:
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                # Include server-provided error body to aid debugging
                err_text = None
                try:
                    err_text = response.text
                except Exception:
                    pass
                raise requests.exceptions.HTTPError(f"{http_err}; server said: {err_text}")
            
            # Parse response
            response_data = response.json()
            
            # Extract the structured output
            if response_data.get("status") == "completed":
                output = response_data.get("output", [])
                if output and len(output) > 0:
                    message = output[0]
                    content = message.get("content", [])
                    if content and len(content) > 0:
                        text_content = content[0].get("text", "{}")
                        classification = json.loads(text_content)
                        
                        return {
                            "success": True,
                            "response_id": response_data.get("id"),
                            "classification": classification,
                            "usage": response_data.get("usage", {}),
                            "error": None
                        }
            
            return {
                "success": False,
                "response_id": None,
                "classification": None,
                "usage": None,
                "error": f"Unexpected response status: {response_data.get('status')}"
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "response_id": None,
                "classification": None,
                "usage": None,
                "error": str(e)
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "response_id": None,
                "classification": None,
                "usage": None,
                "error": f"JSON decode error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "response_id": None,
                "classification": None,
                "usage": None,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def process_csv(
        self, 
        csv_path: str, 
        images_base_path: str,
        output_csv_path: str,
        model: str = "gpt-4o",
        limit: Optional[int] = None,
        delay_seconds: float = 1.0
    ):
        """
        Process images from CSV file and save results
        
        Args:
            csv_path: Path to input CSV file with image paths
            images_base_path: Base directory path where images are stored
            output_csv_path: Path to save output CSV with classifications
            model: Model to use for classification
            limit: Optional limit on number of images to process
            delay_seconds: Delay between API calls to avoid rate limits
        """
        # Read input CSV
        df = pd.read_csv(csv_path)
        
        # Limit number of rows if specified
        if limit:
            df = df.head(limit)
        
        print(f"Processing {len(df)} images...")
        print(f"Model: {model}")
        print(f"Output: {output_csv_path}")
        print("-" * 80)
        
        # Prepare results list
        results = []
        total_tokens = 0
        successful = 0
        failed = 0
        
        # Process each image
        for idx, row in df.iterrows():
            image_filename = row['filepath']
            true_label = row['class']
            
            # Locate image path
            # 1) Direct path under images_base_path
            direct_path = Path(images_base_path) / image_filename
            image_path = None
            if direct_path.exists():
                image_path = direct_path
            else:
                # 2) Search recursively anywhere under images_base_path (handles class subfolders)
                matches = list(Path(images_base_path).rglob(image_filename))
                if len(matches) > 0:
                    image_path = matches[0]
            
            print(f"\n[{idx + 1}/{len(df)}] Processing: {image_filename}")
            print(f"  True label: {true_label}")
            
            if image_path is None or not image_path.exists():
                missing_path = direct_path if image_path is None else image_path
                print(f"  ⚠️  Image not found. Looked for: {missing_path}")
                results.append({
                    "filepath": image_filename,
                    "true_label": true_label,
                    "predicted_scientific_name": None,
                    "confidence": None,
                    "visible_features": None,
                    "success": False,
                    "error": "Image file not found",
                    "response_id": None,
                    "tokens_used": 0
                })
                failed += 1
                continue
            else:
                print(f"  Found image at: {image_path}")
            
            # Classify image
            result = self.classify_image(str(image_path), model=model)
            
            if result["success"]:
                classification = result["classification"]
                usage = result["usage"]
                tokens = usage.get("total_tokens", 0)
                total_tokens += tokens
                successful += 1
                
                print(f"  ✓ Predicted: {classification['scientific_name']}")
                print(f"  Confidence: {classification['confidence']}")
                print(f"  Features: {', '.join(classification['visible_features'][:3])}...")
                print(f"  Tokens used: {tokens}")
                
                results.append({
                    "filepath": image_filename,
                    "true_label": true_label,
                    "predicted_scientific_name": classification["scientific_name"],
                    "confidence": classification["confidence"],
                    "visible_features": json.dumps(classification["visible_features"]),
                    "success": True,
                    "error": None,
                    "response_id": result["response_id"],
                    "tokens_used": tokens
                })
            else:
                print(f"  ✗ Error: {result['error']}")
                failed += 1
                
                results.append({
                    "filepath": image_filename,
                    "true_label": true_label,
                    "predicted_scientific_name": None,
                    "confidence": None,
                    "visible_features": None,
                    "success": False,
                    "error": result["error"],
                    "response_id": None,
                    "tokens_used": 0
                })
            
            # Delay to avoid rate limits
            if idx < len(df) - 1:
                time.sleep(delay_seconds)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_path, index=False)
        
        # Print summary
        print("\n" + "=" * 80)
        print("CLASSIFICATION COMPLETE")
        print("=" * 80)
        print(f"Total images: {len(df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Results saved to: {output_csv_path}")
        
        # Calculate accuracy if we have matches
        if successful > 0:
            matches = results_df[
                results_df['predicted_scientific_name'].str.lower() == 
                results_df['true_label'].str.lower()
            ]
            accuracy = len(matches) / successful * 100
            print(f"\nAccuracy (exact match): {accuracy:.2f}%")
            print(f"Matches: {len(matches)}/{successful}")


def main():
    """Main function to run the classifier"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Classify tree species using OpenAI GPT with structured outputs"
    )
    parser.add_argument(
        "--csv_path",
        required=True,
        help="Path to CSV file with image paths and labels"
    )
    parser.add_argument(
        "--images_path",
        required=True,
        help="Base directory path where images are stored"
    )
    parser.add_argument(
        "--output_csv",
        default="classification_results.csv",
        help="Path to save output CSV with classifications"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-5"],
        help="Model to use for classification"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls"
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key (if not set as environment variable)"
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = TreeSpeciesClassifier(api_key=args.api_key)
    
    # Process images
    classifier.process_csv(
        csv_path=args.csv_path,
        images_base_path=args.images_path,
        output_csv_path=args.output_csv,
        model=args.model,
        limit=args.limit,
        delay_seconds=args.delay
    )


if __name__ == "__main__":
    main()

