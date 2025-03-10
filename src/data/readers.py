def read_json(data_path, key=None):
    """Function to read json data"""
    import json

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if key is not None:
        return [
            entry["FactualNarrative"] for entry in data if "FactualNarrative" in entry
        ]

    else:
        return data
