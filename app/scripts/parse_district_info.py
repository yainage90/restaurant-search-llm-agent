import os
import json
import pandas as pd
import re

def parse_point(point_str):
    if not isinstance(point_str, str):
        return None, None
    match = re.search(r'POINT \(([0-9.-]+) ([0-9.-]+)\)', point_str)
    if match:
        lon, lat = match.groups()
        return float(lon), float(lat)
    return None, None

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = f"{BASE_DIR}/../../data/bjd_info_except_boundary.csv"
    output_file = f"{BASE_DIR}/../../data/coordinates/district_coordinates.jsonl"

    df = pd.read_csv(input_file, encoding="cp949")
    df = df[['bjd_nm', 'center_point']]
    
    results = []
    for _, row in df.iterrows():
        name = row['bjd_nm'].strip()
        if not (name.startswith('서울특별시') or name.startswith('경기도')):
            continue
        
        lon, lat = parse_point(row['center_point'])
        if lon is not None and lat is not None:
            results.append({
                'name': name,
                'lon': lon,
                'lat': lat
            })
        else:
            print(f"Invalid point format: {row['bjd_nm'], row['center_point']}")
    
    os.makedirs(f"{BASE_DIR}/../../data/coordinates", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} records to {output_file}")

if __name__ == "__main__":
    main()