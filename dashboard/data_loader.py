import pandas as pd
from pathlib import Path


def load_data():
    data_path = Path.cwd().joinpath('confluence_data').joinpath('space_analysis.xlsx')
    if data_path.exists():
        data = pd.read_excel(data_path, engine='openpyxl', sheet_name='all_data')
        print("Daten erfolgreich geladen.")
    else:
        print(f"Fehler: Datei {data_path} nicht gefunden.")
        data = pd.DataFrame()  # Leeres DataFrame, um Fehler zu vermeiden

    if not data.empty:
        data['created'] = pd.to_datetime(data['created'], errors='coerce')
        data['lastChanged'] = pd.to_datetime(data['lastChanged'], errors='coerce')
        data['pure_length'] = data['pure_length'].apply(lambda x: max(0, x))

    return data
