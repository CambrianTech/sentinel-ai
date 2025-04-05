#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download Project Gutenberg Books for Benchmarking

This script downloads classic texts from Project Gutenberg for use in benchmarking
experiments. These texts provide real-world natural language data for more realistic
model performance evaluation.

Usage:
    python scripts/download_gutenberg_books.py --output_dir benchmark_data/gutenberg
"""

import os
import sys
import argparse
import urllib.request
import time
import random

# Classic books with their Project Gutenberg URLs
GUTENBERG_BOOKS = {
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "count_of_monte_cristo": "https://www.gutenberg.org/files/1184/1184-0.txt",
    "great_expectations": "https://www.gutenberg.org/files/1400/1400-0.txt",
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "dracula": "https://www.gutenberg.org/files/345/345-0.txt",
    "alice_in_wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
    "war_of_the_worlds": "https://www.gutenberg.org/files/36/36-0.txt",
    "jane_eyre": "https://www.gutenberg.org/files/1260/1260-0.txt"
}

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download Project Gutenberg books for benchmarking")
    
    parser.add_argument("--output_dir", type=str, default="benchmark_data/gutenberg",
                       help="Directory to save downloaded books")
    parser.add_argument("--books", type=str, default="all",
                       help="Comma-separated list of books to download, or 'all'")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information during download")
    
    return parser.parse_args()

def download_book(title, url, output_dir, verbose=False):
    """Download a single book from Project Gutenberg."""
    output_path = os.path.join(output_dir, f"{title}.txt")
    
    # Check if already downloaded
    if os.path.exists(output_path):
        if verbose:
            print(f"Book already exists: {output_path}")
        return True
    
    # Print status
    print(f"Downloading {title} from {url}")
    
    try:
        # Use a random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        # Add a user agent to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # Create request with headers
        req = urllib.request.Request(url, headers=headers)
        
        # Download the file
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        
        # Verify download
        if os.path.getsize(output_path) > 0:
            print(f"Successfully downloaded {title} ({os.path.getsize(output_path) / 1024:.1f} KB)")
            return True
        else:
            print(f"Error: Downloaded file for {title} is empty")
            return False
    
    except Exception as e:
        print(f"Error downloading {title}: {e}")
        
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return False

def main():
    """Main function."""
    args = setup_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which books to download
    if args.books.lower() == "all":
        books_to_download = GUTENBERG_BOOKS
    else:
        # Parse comma-separated list
        requested_books = args.books.split(",")
        books_to_download = {}
        
        for book in requested_books:
            book = book.strip().lower()
            # Try exact match or partial match
            if book in GUTENBERG_BOOKS:
                books_to_download[book] = GUTENBERG_BOOKS[book]
            else:
                # Try partial matches
                matches = [title for title in GUTENBERG_BOOKS.keys() if book in title]
                for match in matches:
                    books_to_download[match] = GUTENBERG_BOOKS[match]
    
    # Download selected books
    if not books_to_download:
        print("No books selected for download.")
        print(f"Available books: {', '.join(GUTENBERG_BOOKS.keys())}")
        return 1
    
    print(f"Downloading {len(books_to_download)} books to {args.output_dir}")
    
    # Track progress
    successful = 0
    failed = 0
    
    for title, url in books_to_download.items():
        if download_book(title, url, args.output_dir, args.verbose):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\nDownload Summary:")
    print(f"  - Successfully downloaded: {successful} books")
    print(f"  - Failed downloads: {failed} books")
    print(f"  - Download location: {os.path.abspath(args.output_dir)}")
    
    # Suggest next steps
    print("\nNext Steps:")
    print("  Run benchmarks with the downloaded books:")
    print("  python scripts/benchmark_with_metrics.py --model_name distilgpt2 --eval_dataset gutenberg --use_real_data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())