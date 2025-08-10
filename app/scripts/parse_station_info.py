import os
import json
import re

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = f"{BASE_DIR}/../../data/station_info.json"
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)["DATA"]
    
    station_data = {}
    for obj in data:
        name = re.sub(r'\([^)]*\)', '', obj['bldn_nm']).strip() + "역"
        lat = float(obj["lat"])
        lon = float(obj["lot"])
        
        if name in station_data:
            station_data[name]["lat_sum"] += lat
            station_data[name]["lon_sum"] += lon
            station_data[name]["count"] += 1
        else:
            station_data[name] = {
                "lat_sum": lat,
                "lon_sum": lon,
                "count": 1
            }
    
    stations = []
    for name, data in station_data.items():
        avg_lat = data["lat_sum"] / data["count"]
        avg_lon = data["lon_sum"] / data["count"]
        stations.append({"name": name, "lat": avg_lat, "lon": avg_lon})
    
    os.makedirs(f"{BASE_DIR}/../../data/coordinates", exist_ok=True)
    output_file = f"{BASE_DIR}/../../data/coordinates/station_coordinates.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for station in stations:
            f.write(json.dumps(station, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()