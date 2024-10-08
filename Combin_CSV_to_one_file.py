import os
import pandas as pd

# Verzeichnis, in dem sich die CSV-Dateien befinden
csv_folder = 'C:/Users/tamar/alle_csv'
output_file = 'C:/Users/tamar/combined.csv'

# Liste aller CSV-Dateien im Verzeichnis
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Schreibe den Header der ersten Datei und füge den Rest in den Batch-Modus
with open(output_file, 'w') as outfile:
    for i, file in enumerate(csv_files):
        file_path = os.path.join(csv_folder, file)

        # Lies die Datei in Chunks, falls sie sehr groß ist
        chunk_iter = pd.read_csv(file_path, chunksize=100000)

        for chunk in chunk_iter:
            # Schreibe den Header nur beim ersten Mal
            if i == 0:
                chunk.to_csv(outfile, index=False, header=True, mode='a')
            else:
                chunk.to_csv(outfile, index=False, header=False, mode='a')
