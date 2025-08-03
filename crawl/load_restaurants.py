import os
import requests
import time
import json
import re
from dotenv import load_dotenv
from pathlib import Path

def clean_html(raw_html: str) -> str:
    """Removes HTML tags from a string."""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def search_naver_local(query: str, client_id: str, client_secret: str) -> tuple[list, bool]:
    """Calls the Naver Local Search API with pagination and returns all items."""
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    all_items = []
    start = 1
    display = 5  # The Naver Local Search API's 'display' parameter has a maximum value of 5.
    # The Naver API allows a 'start' value up to 1000.
    max_start = 1000

    while start <= max_start:
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": "comment",
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            # Add a 0.5-second delay after each request to avoid overwhelming the server
            time.sleep(0.1)

            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()
            items = data.get("items", [])

            if not items:
                break

            all_items.extend(items)

            total = data.get("total", 0)
            # Stop if we have fetched all available results
            if start + display > total:
                break

            start += display
        except requests.exceptions.RequestException as e:
            print(f"Error during API request for query '{query}' (start={start}): {e}")
            # Stop paginating for this query if an error occurs
            break

    # Return items and whether any results were found
    return all_items, len(all_items) > 0

def main():
    """Main function to crawl restaurant data and save it to a JSONL file."""
    # Load environment variables from .env file
    load_dotenv()
    
    naver_client_id = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    if not naver_client_id or not naver_client_secret:
        print("Error: NAVER_CLIENT_ID and NAVER_CLIENT_SECRET must be set in a .env file.")
        return


    # Define paths
    project_root = Path(__file__).resolve().parent.parent

    if not os.path.exists(project_root / "data"):
        os.makedirs(project_root / "data")

    locations_file_path = project_root / "crawl" / "locations.txt"
    output_jsonl_path = project_root / "data" / "restaurants.jsonl"
    food_keywords_file_path = project_root / "crawl" / "food_keywords.txt"
    failed_queries_file_path = project_root / "data" / "load_failed_queries.txt"

    # Load existing restaurant keys and queries to prevent duplicates and reprocessing
    existing_restaurant_keys = set()
    existing_queries = set()
    if output_jsonl_path.exists():
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    restaurant = json.loads(line)
                    # Create a unique key from title, mapx, and mapy
                    key = f"{restaurant.get('title', '')}{restaurant.get('mapx', '')}{restaurant.get('mapy', '')}"
                    if key:
                        existing_restaurant_keys.add(key)
                    # Track existing queries
                    query = restaurant.get('query', '')
                    if query:
                        existing_queries.add(query)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
    print(f"Loaded {len(existing_restaurant_keys)} existing restaurant keys to prevent duplication.")
    print(f"Loaded {len(existing_queries)} existing queries to prevent reprocessing.")

    # Load failed queries to skip them
    failed_queries = set()
    if failed_queries_file_path.exists():
        with open(failed_queries_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                query = line.strip()
                if query:
                    failed_queries.add(query)
    print(f"Loaded {len(failed_queries)} failed queries to skip.")

    # Read all valid locations, ignoring comments and empty lines.
    try:
        with open(locations_file_path, 'r', encoding='utf-8') as f:
            locations = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith('#')
            ]
    except FileNotFoundError:
        print(f"Error: Locations file not found at {locations_file_path}")
        return

    # Read all valid keywords from food_keywords.txt, ignoring comments and empty lines.
    try:
        with open(food_keywords_file_path, 'r', encoding='utf-8') as f:
            food_keywords = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith('#')
            ]
    except FileNotFoundError:
        print(f"Error: Keywords file not found at {food_keywords_file_path}")
        return

    newly_added_count = 0
    skipped_queries_count = 0
    failed_queries_count = 0
    # Open the file in append mode ('a') to add new results without overwriting.
    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
        with open(failed_queries_file_path, 'a', encoding='utf-8') as failed_f:
            for location in locations:
                for food_keyword in food_keywords:
                    query = f"{location} {food_keyword}"
                    
                    # Skip if this query has already been processed
                    if query in existing_queries:
                        skipped_queries_count += 1
                        print(f"Skipping already processed query: {query}")
                        continue
                    
                    # Skip if this query has previously failed
                    if query in failed_queries:
                        skipped_queries_count += 1
                        print(f"Skipping previously failed query: {query}")
                        continue
                    
                    print(f"Searching for: {query}")
                    items, has_results = search_naver_local(query, naver_client_id, naver_client_secret)
                    
                    # If no results found, record as failed query
                    if not has_results:
                        failed_f.write(f"{query}\n")
                        failed_f.flush()  # Ensure it's written immediately
                        failed_queries.add(query)  # Add to memory set to avoid duplicates in current run
                        failed_queries_count += 1
                        print(f"No results found for query: {query}")
                        continue
                    
                    for item in items:
                        address = item.get("address") 
                        if not ("서울특별시" in address or "경기도" in address):
                            continue
                        
                        restaurant_data = {
                            "title": clean_html(item.get("title", "")),
                            "address": item.get("address", ""),
                            "roadAddress": item.get("roadAddress", ""),
                            "mapx": item.get("mapx", ""),
                            "mapy": item.get("mapy", ""),
                            "query": query,
                        }
                        existing_queries.add(query)
                        
                        # Create a unique key and check for duplicates before writing
                        unique_key = f"{restaurant_data['title']}{restaurant_data['mapx']}{restaurant_data['mapy']}"
                        if unique_key and unique_key not in existing_restaurant_keys:
                            f.write(json.dumps(restaurant_data, ensure_ascii=False) + '\n')
                            existing_restaurant_keys.add(unique_key)
                            newly_added_count += 1

    if newly_added_count == 0:
        print("No new restaurants found in this run.")
    else:
        print(f"Successfully appended {newly_added_count} new restaurants to {output_jsonl_path}")
    
    if skipped_queries_count > 0:
        print(f"Skipped {skipped_queries_count} queries that were already processed.")
    
    if failed_queries_count > 0:
        print(f"Found {failed_queries_count} queries with no results and saved them to {failed_queries_file_path}")

if __name__ == "__main__":
    main()