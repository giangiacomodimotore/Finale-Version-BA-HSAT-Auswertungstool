# -*- coding: utf-8 -*-
"""
Finished on Mon Jun 03 10:12:42 2024

@author: Janis Preisig & Lona Tulinski
"""


#%% DB to dict
import sqlite3
import pandas as pd
import re
print('#### loading data into dict ####\n')
# Pfad zur Datenbank
db_path = r"C:\Users\Public\BA\Datenbank\BA_BIFOROT_Pmpp.db"

# Verbindung zur SQLite-Datenbank herstellen
conn = sqlite3.connect(db_path)

# Abfrage aller Tabellen, die mit 'Pmpp_' beginnen
table_query = "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Pmpp_%';"
tables = pd.read_sql_query(table_query, conn)['name'].tolist()

# Daten-Dictionary initialisieren
raw_dict = {}
processed_dates = []

# Für jede Tabelle die spezifizierten Daten laden
for table in tables:
    # Datum aus dem Tabellennamen extrahieren
    date_match = re.search(r'Pmpp_(\d{4})_(\d{2})_(\d{2})', table)
    if date_match:
        date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        
        # SQL-Query vorbereiten
        sql_query = f"""
        SELECT
            Pmpp_middle_Module,
            Angle_middle_Module,
            date_hour,
            date_min,
            date_sec,
            (date_hour * 3600 + date_min * 60 + date_sec) AS seconds_since_midnight,
            printf("%02d:%02d:%02d", date_hour, date_min, date_sec) AS time_hms
        FROM {table}
        """
        
        # Daten für diese Tabelle lesen
        df = pd.read_sql_query(sql_query, conn)
        
        # Überprüfen, ob DataFrame nicht leer ist
        if not df.empty:
            processed_dates.append(date)
            raw_dict[date] = df

# Verbindung zur Datenbank schließen
conn.close()


#%% sorted dict
import numpy as np
import pandas as pd

print('#### sorting dict ####\n') 
sorted_dict = {}

for date, df in raw_dict.items():
    angle_values = [-90, -70, -50, -35, -20, -10, -5, 0, 5, 10, 20, 35, 50, 70, 90]

    # Funktion zum Finden des nächstgelegenen Winkels
    def find_nearest_angle(angle):
        return min(angle_values, key=lambda x: abs(x - angle))

    # Anwenden der Funktion auf die 'Angle_middle_module' Spalte, um die nächstgelegenen Winkel zu finden
    df['Angle_middle_Module'] = df['Angle_middle_Module'].apply(find_nearest_angle)

    # Entferne negative Pmpp_middle_Module-Werte
    df = df[df['Pmpp_middle_Module'] >= 0]

    # Entferne Werte, bei denen die Zeitdifferenz zwischen aufeinanderfolgenden Messungen kleiner als 100 Sekunden ist
    df = df.loc[df.sort_values(by=['Angle_middle_Module', 'seconds_since_midnight']).index]
    df['delta_time'] = df['seconds_since_midnight'].diff()
    df = df[(df['delta_time'] >= 100) | (df['delta_time'].isnull())]
    df.drop(columns=['delta_time'], inplace=True)

    # Aktualisiere das sorted_dict für das aktuelle Datum
    sorted_dict[date] = df.copy()  # Kopie des DataFrame hinzufügen, um die Originaldaten beizubehalten


    
#%% calculate energy yield
import pandas as pd
print('#### calculate yield into dict ####\n')
yield_dict = {}  # Neues Dictionary zur Speicherung der Energieberechnungen

# Iteriere durch jedes Datum im sorted_dict
for date, df in sorted_dict.items():
    # Sicherstellen, dass die Daten nach Winkel und Sekunden seit Mitternacht sortiert sind
    sorted_df = df.sort_values(by=['Angle_middle_Module', 'seconds_since_midnight'])
    
    # Initialisiere ein Dictionary für diesen Tag
    date_dict = {}
    
    # Gruppiere nach Winkel und berechne die Gesamtenergie für jeden Winkel
    for angle, group in sorted_df.groupby('Angle_middle_Module'):
        # Berechne die Zeitdifferenz zwischen aufeinanderfolgenden Messungen
        group['delta_time'] = group['seconds_since_midnight'].diff() / 3600  # Umwandlung von Sekunden in Stunden
        group.fillna(0, inplace=True)  # Ersetze NaN-Werte durch 0 für die erste Messung des Tages
        
        # Definiere die Zeitintervalle (hier: 30 Minuten)
        interval_size = 30 * 60  # 30 Minuten in Sekunden

        # Überprüfe, ob die Gruppe nicht leer ist
        if not group.empty:
            # Berechne die Anzahl der Intervalle
            num_intervals = max(1, len(group) // 14)  # Mindestens 1 Intervall

            # Berechne die Energie für jeden 15er-Block
            energy = (group.groupby(np.arange(len(group)) // (len(group) // num_intervals))['Pmpp_middle_Module'].mean() * 
            group.groupby(np.arange(len(group)) // (len(group) // num_intervals))['delta_time'].sum())
            
            # Berechne den gemittelten Pmpp-Wert und die Zeitdifferenz des Blocks
            avg_pmpp = group.groupby(np.arange(len(group)) // (len(group) // num_intervals))['Pmpp_middle_Module'].mean()
            delta_t = group.groupby(np.arange(len(group)) // (len(group) // num_intervals))['delta_time'].sum()

            # Berechne den Durchschnitt der seconds_since_midnight pro Zeitintervall
            avg_seconds_since_midnight = group.groupby(np.arange(len(group)) // (len(group) // num_intervals))['seconds_since_midnight'].mean()

         
            # Konvertiere die seconds_since_midnight in HH:MM:SS-Format
            avg_time_formatted = pd.to_datetime(avg_seconds_since_midnight, unit='s').dt.strftime('%H:%M:%S')

            # Speichere den Winkel und den DataFrame mit Zeit, Energie, avg_pmpp und delta_t im date_dict
            date_dict[angle] = pd.DataFrame({
                'seconds_since_midnight': avg_seconds_since_midnight,
                'energy': energy,
                'time_formatted': avg_time_formatted,
                'avg_pmpp': avg_pmpp,
                'delta_t': delta_t
            })
   
    # Füge das date_dict zum yield_dict hinzu
    yield_dict[date] = date_dict


#%% eliminate outliers energy < 1 

print('#### elimination of outliers ####\n')

for date, date_dict in yield_dict.items():
    for angle, df in date_dict.items():
        # Filtere den DataFrame, um nur Zeilen zu behalten, bei denen `energy` größer als 1 ist
        filtered_df = df[df['energy'] >= 1]
        yield_dict[date][angle] = filtered_df


#%% clearsky or cloudy 

import pandas as pd

print('#### identifying clearsky and cloudy days ####\n')

# Dateipfad zur CSV-Datei
# Daten sind stündlich
# Achtung die Wetterdaten gehen nur bis zum 30.10.23 (Letzes Semester von Roger erhalten)
# Neue files von Roger anfragen
file_path = r'C:\Users\Public\BA\Rohdaten_weather\weatherdata_pveye_2023_2024.csv'

# Laden der Daten in ein Pandas DataFrame
data = pd.read_csv(file_path)

# Umwandeln der Spalte "Time" in ein Datetime-Objekt, falls sie noch keine ist
data['Time'] = pd.to_datetime(data['Time'])

# Berechnung bzw. Identifizierung über das Verhältnis von diffuser und direkter Strahlung

# Berechnung des Mittelwerts von Var36 pro Datum
# Var36 = diffus (Pyranometer)
mean_var36_per_date = data.groupby(data['Time'].dt.date)['Var36'].mean()

# Erstellen eines Dictionaries mit dem Datum als Schlüssel und dem Mittelwert von Var36 als Wert
mean_var36_dict = mean_var36_per_date.to_dict()

# Berechnung des Mittelwerts von Var37 pro Datum
# Var37 = direkt (Pyrheliometer)
mean_var37_per_date = data.groupby(data['Time'].dt.date)['Var37'].mean()

# Erstellen eines Dictionaries mit dem Datum als Schlüssel und dem Mittelwert von Var37 als Wert
mean_var37_dict = mean_var37_per_date.to_dict()

# Vergleich der Mittelwerte von Var36 und Var37 pro Datum
# Schreibt jeweils eine Liste clearsky und eine Liste cloudy mit den Daten
clearsky = []
cloudy = []

for date in mean_var36_dict.keys():
    if date in mean_var37_dict:
        if mean_var37_dict[date] > mean_var36_dict[date]:
            clearsky.append(date)
        else:
            cloudy.append(date)

# print("Anzahl clearsky:", len(clearsky))
# print("Anzahl cloudy:", len(cloudy))

#%% defining timeframe 

print('#### defining timeframe ####\n')

start_date = pd.to_datetime("2023-05-03")
end_date = pd.to_datetime("2024-04-02")
#selected months:
selected_months = [ '2023-05','2023-06','2023-07','2023-08', '2023-09','2023-10', '2023-11', '2023-12','2024-01','2024-02','2024-03','2024-04']  # Beispielmonate

#%% rewrite seconds_since_midnight to 30min intervals

print('#### standardization of time ####\n')

import plotly.express as px
import numpy as np
import pandas as pd

thirty_min_day = [
    (5 * 3600 + 900),     # 5:15 Uhr
    (5 * 3600 + 2700),    # 5:45 Uhr
    (6 * 3600 + 900),     # 6:15 Uhr
    (6 * 3600 + 2700),    # 6:45 Uhr
    (7 * 3600 + 900),     # 7:15 Uhr
    (7 * 3600 + 2700),    # 7:45 Uhr
    (8 * 3600 + 900),     # 8:15 Uhr
    (8 * 3600 + 2700),    # 8:45 Uhr
    (9 * 3600 + 900),     # 9:15 Uhr
    (9 * 3600 + 2700),    # 9:45 Uhr
    (10 * 3600 + 900),    # 10:15 Uhr
    (10 * 3600 + 2700),   # 10:45 Uhr
    (11 * 3600 + 900),    # 11:15 Uhr
    (11 * 3600 + 2700),   # 11:45 Uhr
    (12 * 3600 + 900),    # 12:15 Uhr
    (12 * 3600 + 2700),   # 12:45 Uhr
    (13 * 3600 + 900),    # 13:15 Uhr
    (13 * 3600 + 2700),   # 13:45 Uhr
    (14 * 3600 + 900),    # 14:15 Uhr
    (14 * 3600 + 2700),   # 14:45 Uhr
    (15 * 3600 + 900),    # 15:15 Uhr
    (15 * 3600 + 2700),   # 15:45 Uhr
    (16 * 3600 + 900),    # 16:15 Uhr
    (16 * 3600 + 2700),   # 16:45 Uhr
    (17 * 3600 + 900),    # 17:15 Uhr
    (17 * 3600 + 2700),   # 17:45 Uhr
    (18 * 3600 + 900),    # 18:15 Uhr
    (18 * 3600 + 2700),   # 18:45 Uhr
    (19 * 3600 + 900),    # 19:15 Uhr
    (19 * 3600 + 2700),   # 19:45 Uhr
    (20 * 3600 + 900),    # 20:15 Uhr
    (20 * 3600 + 2700)    # 20:45 Uhr
]

# Funktion, um den nächstgelegenen Wert in thirty_min_day zu finden
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Funktion zur Umwandlung von Sekunden in Stunden:Minuten Format
def seconds_to_hm(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"

# Durchlaufe alle Daten in yield_dict
for date, angles_dict in yield_dict.items():
    for angle, df in angles_dict.items():
        # Finde für jeden Wert in seconds_since_midnight den nächstgelegenen Wert in thirty_min_day
        df['seconds_since_midnight'] = df['seconds_since_midnight'].apply(lambda x: find_nearest(thirty_min_day, x))
        # Neue Spalte mit formatierter Zeit
        df['time_hm'] = df['seconds_since_midnight'].apply(seconds_to_hm)


#%% max. energy yield angle per timeframe

print('#### identifying max. energy yield angles per timeframe ####\n')

# Spezifische Winkel, die berücksichtigt werden sollen
specific_angles = [-20, -10, -5, 0, 5, 10, 20]

# Sammle alle relevanten Daten
all_data1 = []
for single_date in pd.date_range(start=start_date, end=end_date):
    date_str = single_date.strftime("%Y-%m-%d")
    if date_str in yield_dict:
        angles_dict = yield_dict[date_str]
        for angle, df in angles_dict.items():
            if angle in specific_angles:
                # Anpassung für 'seconds_since_midnight' und Erstellung von 'time_hm'
                df['seconds_since_midnight'] = df['seconds_since_midnight'].apply(lambda x: find_nearest(thirty_min_day, x))
                df['time_hm'] = df['seconds_since_midnight'].apply(seconds_to_hm)
                for _, row in df.iterrows():
                    all_data1.append((row['seconds_since_midnight'], row['time_hm'], row['energy'], angle, date_str))

# Konvertiere die gesammelten Daten in ein DataFrame
all_data1_df = pd.DataFrame(all_data1, columns=['seconds_since_midnight', 'time_hm', 'energy', 'angle', 'date'])

# Gruppiere nach 'seconds_since_midnight' und finde den maximalen 'energy' Wert über alle Tage und Winkel
max_energy_angle = all_data1_df.groupby('seconds_since_midnight').apply(lambda x: x.loc[x['energy'].idxmax()])

# Entferne den Index, den groupby hinzufügt
max_energy_angle.reset_index(drop=True, inplace=True)

# # Ergebnisse ausgeben
# print(max_energy_angle[['seconds_since_midnight', 'time_hm', 'energy', 'angle', 'date']])

print('#### plotting max. energy yield angles per timeframe ####\n')

import os


# Formatierung des Timeframes für den Titel
timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

# Erstelle ein Plotly Scatter-Plot mit angepasstem Titel
fig = px.scatter(max_energy_angle, x='time_hm', y='angle', color='energy',
                 title=f'Max. energy yield angles over timeframe {timeframe}',
                 labels={'time_hm': 'Time', 'angle': 'Angle', 'energy': 'Energy'})

print('#### saving plot as HTML document ####\n')

# Konstruiere den Dateinamen dynamisch
file_name = f"max_energy_angles_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

# Überprüfe, ob die Datei bereits vorhanden ist
if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")

   

#%% daily max energy dict

print('#### indentifying daily max. energy yield angles ####\n ')


from collections import Counter

# Initialisiere das Ergebnis-Dict und ein Dict für die Energie-Summen
daily_max_energy_dict = {}
daily_energy_sum_dict = {}

# Verarbeite die Daten
all_data2 = []
for date, angles_dict in yield_dict.items():
    for angle in specific_angles:
        if angle in angles_dict:
            df = angles_dict[angle]
            df['time_hm'] = df['seconds_since_midnight'].apply(seconds_to_hm)
            for _, row in df.iterrows():
                all_data2.append({
                    'seconds_since_midnight': row['seconds_since_midnight'],
                    'time_hm': row['time_hm'],
                    'energy': row['energy'],
                    'angle': angle,
                    'date': date  # Füge das Datum hinzu
                })

# Konvertiere die gesammelten Daten in ein DataFrame
all_data2_df = pd.DataFrame(all_data2)

# Verarbeitung, wenn Daten vorhanden sind
if not all_data2_df.empty:
    # Gruppiere nach Datum und seconds_since_midnight, finde maximale Energie und zugehörigen Winkel
    grouped = all_data2_df.groupby(['date', 'seconds_since_midnight'])
    max_energy_angle = grouped.apply(lambda x: x.loc[x['energy'].idxmax()]).reset_index(drop=True)
    
    # Berechne die tägliche Gesamtsumme der Energie
    for date, group in max_energy_angle.groupby('date'):
        daily_max_energy_dict[date] = group
        daily_energy_sum_dict[date] = group['energy'].sum()

# Ergebnisse der Winkelverläufe pro Tag
results = {}
for df in daily_max_energy_dict.values():
    for _, row in df.iterrows():
        key = (row['seconds_since_midnight'], row['date'])
        if key not in results:
            results[key] = []
        results[key].append(row['angle'])
        results[key].append(row['time_hm'])
        



#%% most common max. energy angle per timeframe

print('#### calculating most common max. energy angles ####\n')


# Initialisiere eine Liste, um alle Zeilen zu speichern
most_common_angles = []

# Iteriere über jedes DataFrame und sammle die benötigten Daten
for date, df in daily_max_energy_dict.items():
    rows = df[['seconds_since_midnight', 'angle', 'time_hm']].values.tolist()
    most_common_angles.extend(rows)

# Erstelle ein DataFrame aus der gesammelten Liste
result_df = pd.DataFrame(most_common_angles, columns=['seconds_since_midnight', 'angle', 'time_hm'])

# Funktion, um den häufigsten Winkel und die zugehörige Zeit zu finden
def get_most_common_angle(group):
    most_common_angle = Counter(group['angle']).most_common(1)[0][0]
    corresponding_time_hm = group[group['angle'] == most_common_angle]['time_hm'].iloc[0]
    return pd.Series({'angle': most_common_angle, 'time_hm': corresponding_time_hm})

# Gruppiere nach 'seconds_since_midnight' und wende die Funktion an
most_common_angle_daily = result_df.groupby('seconds_since_midnight').apply(get_most_common_angle).reset_index()

# # Ergebnisse anzeigen
# print(most_common_angle_daily)


print('#### plotting most common max. energy angles ####\n')

import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

# Bereite die Daten für die lineare Regression vor
X = most_common_angle_daily['seconds_since_midnight']
y = most_common_angle_daily['angle']
X = sm.add_constant(X)  # Fügt die Konstante hinzu, da statsmodels dies nicht automatisch macht

# Führe die lineare Regression durch
model = sm.OLS(y, X).fit()

# Berechne die Vorhersagen für die Regressionsgerade
predictions = model.predict(X)

# Formatierung des Timeframes für den Titel
timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

# Erstelle ein Plotly Scatter-Plot mit angepasstem Titel
fig = px.scatter(most_common_angle_daily, x='time_hm', y='angle',
                 title=f'Most common max. energy yield angles {timeframe}',
                 labels={'time_hm': 'Time', 'angle': 'Angle'})

# Füge die Regressionsgerade zum Plot hinzu
fig.add_trace(
    go.Scatter(x=most_common_angle_daily['time_hm'], y=predictions,
               mode='lines', name='Regressionsgerade')
)

print('###### saving plot as HTML document ######\n')

# Konstruiere den Dateinamen dynamisch
file_name = f"most_common_angles_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

# Überprüfe, ob die Datei bereits vorhanden ist
if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")


#%% average angle max. energy

print('#### calculating average max. energy angles per timeframe ####')

## Sammle alle relevanten Daten
all_seconds = []
for single_date in pd.date_range(start=start_date, end=end_date):
    date_str = single_date.strftime("%Y-%m-%d")
    if date_str in daily_max_energy_dict:
        df = daily_max_energy_dict[date_str]
        df['seconds_since_midnight'] = df['seconds_since_midnight'].apply(lambda x: min(thirty_min_day, key=lambda y:abs(y-x)))
        for _, row in df.iterrows():
            if row['angle'] in specific_angles:
                all_seconds.append((row['seconds_since_midnight'], row['angle']))

# Konvertiere die gesammelten Daten in ein DataFrame
all_seconds_df = pd.DataFrame(all_seconds, columns=['seconds_since_midnight', 'angle'])

counts = all_seconds_df.groupby('seconds_since_midnight').size()

# Filtern der Gruppen, die mehr als 10 Einträge haben
valid_seconds = counts[counts > 5].index

# Berechnen des Durchschnittswinkels nur für die gefilterten 'seconds_since_midnight'-Werte
average_angle = all_seconds_df[all_seconds_df['seconds_since_midnight'].isin(valid_seconds)] \
    .groupby('seconds_since_midnight')['angle'].mean().reset_index()

# Sortiere das DataFrame nach seconds_since_midnight (sollte bereits sortiert sein, aber zur Sicherheit)
average_angle = average_angle.sort_values(by='seconds_since_midnight')

# Umformatierung von Sekunden in hh:mm
average_angle['time_format'] = average_angle['seconds_since_midnight'].apply(lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}")


print('#### plotting average max. energy angles per timeframe ####')

# Bereite die Daten für die lineare Regression vor
X = average_angle['seconds_since_midnight']
y = average_angle['angle']
X = sm.add_constant(X)  # Fügt die Konstante hinzu, da statsmodels dies nicht automatisch macht

# Führe die lineare Regression durch
model = sm.OLS(y, X).fit()

# Berechne die Vorhersagen für die Regressionsgerade
predictions = model.predict(X)

# Formatierung des Timeframes für den Titel
timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

# Erstelle den Plotly Scatter-Plot
fig = px.scatter(average_angle, x='time_format', y='angle',
                 title=f'average angle over time with regression line {timeframe}',
                 labels={'time_format': 'Time', 'angle': 'Angle'})

# Füge die Regressionsgerade zum Plot hinzu
fig.add_trace(
    go.Scatter(x=average_angle['time_format'], y=predictions,
               mode='lines', name='Regressionsgerade')
)

print('###### saving plot as HTML document ######\n')

# Überprüfe, ob die Datei bereits vorhanden ist, bevor du versuchst, sie zu speichern
file_name = f"average_angles_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")
#%% Frühling zusammengesetzt aus 2023 und 2024

# Define specific months to include in the analysis
relevant_months = ['2023-05', '2024-03', '2024-04']

# Initialize an empty list to collect the data
all_seconds = []

# Iterate through each day within the specified months, ensuring valid end dates
for month in relevant_months:
    start_date = f"{month}-01"
    end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
    for single_date in pd.date_range(start=start_date, end=end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        if date_str in daily_max_energy_dict:
            df = daily_max_energy_dict[date_str]
            df['seconds_since_midnight'] = df['seconds_since_midnight'].apply(lambda x: find_nearest(thirty_min_day, x))
            for _, row in df.iterrows():
                all_seconds.append((row['seconds_since_midnight'], row['angle']))


# Convert the collected data into a DataFrame
all_seconds_df = pd.DataFrame(all_seconds, columns=['seconds_since_midnight', 'angle'])

# Compute counts of data points per seconds_since_midnight
counts = all_seconds_df.groupby('seconds_since_midnight').size()

# Filter to include time slots with sufficient data
valid_seconds = counts[counts > 5].index

# Calculate the average angle only for the filtered time slots
average_angle = all_seconds_df[all_seconds_df['seconds_since_midnight'].isin(valid_seconds)] \
    .groupby('seconds_since_midnight')['angle'].mean().reset_index()

# Sort the DataFrame by seconds_since_midnight for plotting consistency
average_angle = average_angle.sort_values(by='seconds_since_midnight')

# Convert seconds_since_midnight to hh:mm format for easier reading
average_angle['time_format'] = average_angle['seconds_since_midnight'].apply(seconds_to_hm)

# Prepare data for linear regression
X = average_angle['seconds_since_midnight']
y = average_angle['angle']
X = sm.add_constant(X)  # Adds a constant term to the predictor

# Perform linear regression
model = sm.OLS(y, X).fit()

# Calculate predictions for the regression line
predictions = model.predict(X)

# Formatting the timeframe for the title
timeframe = f"Specific months: {', '.join(relevant_months)}"

# Create a Plotly scatter plot
fig = px.scatter(average_angle, x='time_format', y='angle',
                 title=f'Average Max. Energy Angle Over Time with Regression Line {timeframe}',
                 labels={'time_format': 'Time', 'angle': 'Angle'})

# Add the regression line to the plot
fig.add_trace(
    go.Scatter(x=average_angle['time_format'], y=predictions,
               mode='lines', name='Regression Line')
)

# Save the plot to an HTML document
file_name = "average_max_energy_angles_specific_months.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

if not os.path.exists(file_path):
    fig.write_html(file_path)
else:
    print("The file already exists.\n")

# Optionally display the plot
# fig.show()


#%% clearsky dict
print('#### calculating clearsky dict ####\n')

# Konvertiere die datetime.date Objekte in der clearsky Liste in Zeichenketten
clearsky_str = [date.strftime('%Y-%m-%d') for date in clearsky]


# Initialisiere das neue Dictionary für Ergebnisse nur an klaren Tagen
daily_max_clearsky = {}


# new dict for clearsky dates
import datetime


clearsky_dates = set(clearsky)  # Konvertiere die Liste zu einem Set für eine effizientere Suche

# Durchlaufe alle Daten in yield_tracking
for date_str, data in daily_max_energy_dict.items():
    # Konvertiere den String zu einem datetime.date-Objekt
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Überprüfe, ob das Datum in der Liste der bewölkten Tage ist UND ob das Datum in yield_tracking vorhanden ist
    if date_obj in clearsky_dates and date_str in daily_max_energy_dict:
        # Kopiere die Daten in das neue Dictionary
        daily_max_clearsky[date_str] = data.copy()  # Verwende .copy(), um sicherzustellen, dass keine Referenzen kopiert werden




#%% cloudy dict
print('###### calculating cloudy dict ######\n')

# Konvertiere die datetime.date Objekte in der cloudy Liste in Zeichenketten
cloudy_str = [date.strftime('%Y-%m-%d') for date in cloudy]



# Initialisiere das neue Dictionary für Ergebnisse nur an klaren Tagen
daily_max_cloudy = {}


# new dict for clearsky dates
import datetime


cloudy_dates = set(cloudy)  # Konvertiere die Liste zu einem Set für eine effizientere Suche

# Durchlaufe alle Daten in yield_tracking
for date_str, data in daily_max_energy_dict.items():
    # Konvertiere den String zu einem datetime.date-Objekt
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Überprüfe, ob das Datum in der Liste der bewölkten Tage ist UND ob das Datum in yield_tracking vorhanden ist
    if date_obj in cloudy_dates and date_str in daily_max_energy_dict:
        # Kopiere die Daten in das neue Dictionary
        daily_max_cloudy[date_str] = data.copy()  # Verwende .copy(), um sicherzustellen, dass keine Referenzen kopiert werden


#%% most common angle clearsky

print('#### calculating most common max. energy angles (clearsky) ####')

# Initialisiere eine Liste, um alle Zeilen zu speichern
most_common_angles_clearsky = []

# Iteriere über jedes DataFrame und sammle die benötigten Daten
for date, df in daily_max_clearsky.items():
    rows = df[['seconds_since_midnight', 'angle', 'time_hm']].values.tolist()
    most_common_angles_clearsky.extend(rows)

# Erstelle ein DataFrame aus der gesammelten Liste
result_df = pd.DataFrame(most_common_angles_clearsky, columns=['seconds_since_midnight', 'angle', 'time_hm'])

# Stellen Sie sicher, dass die Spalten numerisch sind
result_df['seconds_since_midnight'] = pd.to_numeric(result_df['seconds_since_midnight'])
result_df['angle'] = pd.to_numeric(result_df['angle'])

# Funktion, um den häufigsten Winkel und die zugehörige Zeit zu finden
def get_most_common_angle_clearsky(group):
    most_common_angle_clearsky = Counter(group['angle']).most_common(1)[0][0]
    corresponding_time_hm = group[group['angle'] == most_common_angle_clearsky]['time_hm'].iloc[0]
    return pd.Series({'angle': most_common_angle_clearsky, 'time_hm': corresponding_time_hm})

# Gruppiere nach 'seconds_since_midnight' und wende die Funktion an
most_common_angle_daily_clearsky = result_df.groupby('seconds_since_midnight').apply(get_most_common_angle_clearsky).reset_index()

print('#### plotting most common max. energy angles (clearsky) ####')

# Bereite die Daten für die lineare Regression vor
X = most_common_angle_daily_clearsky['seconds_since_midnight']
y = most_common_angle_daily_clearsky['angle']
X = sm.add_constant(X)  # Fügt die Konstante hinzu, da statsmodels dies nicht automatisch macht

# Führe die lineare Regression durch
model = sm.OLS(y, X).fit()

# Berechne die Vorhersagen für die Regressionsgerade
predictions = model.predict(X)

# Formatierung des Timeframes für den Titel
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

# Erstelle ein Plotly Scatter-Plot mit angepasstem Titel
fig = px.scatter(most_common_angle_daily_clearsky, x='time_hm', y='angle',
                  title=f'Most common max. energy yield angles (clearsky) {timeframe}',
                  labels={'time_hm': 'Time', 'angle': 'Angle'})

# Füge die Regressionsgerade zum Plot hinzu
fig.add_trace(
    go.Scatter(x=most_common_angle_daily_clearsky['time_hm'], y=predictions,
                mode='lines', name='Regressionsgerade')
)

print('#### saving plot as HTML document ####\n')

# Zum Speichern als HTML
# Konstruiere den Dateinamen dynamisch
file_name = f"most_common_angles_clearsky_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

# Überprüfe, ob die Datei bereits vorhanden ist
if not os.path.exists(file_path):
    # Speichere die Datei mit dem dynamischen Dateinamen
    fig.write_html(file_path)
else:
    print("Die Datei existiert bereits.\n")
#%% most common angle cloudy

print('#### calculating most common max. energy angles (cloudy) ####')

# Initialisiere eine Liste, um alle Zeilen zu speichern
most_common_angles_cloudy = []

# Iteriere über jedes DataFrame und sammle die benötigten Daten
for date, df in daily_max_cloudy.items():
    rows = df[['seconds_since_midnight', 'angle', 'time_hm']].values.tolist()
    most_common_angles_cloudy.extend(rows)

# Erstelle ein DataFrame aus der gesammelten Liste
result_df = pd.DataFrame(most_common_angles_cloudy, columns=['seconds_since_midnight', 'angle', 'time_hm'])

# Funktion, um den häufigsten Winkel und die zugehörige Zeit zu finden
def get_most_common_angle_cloudy(group):
    most_common_angle_cloudy = Counter(group['angle']).most_common(1)[0][0]
    corresponding_time_hm = group[group['angle'] == most_common_angle_cloudy]['time_hm'].iloc[0]
    return pd.Series({'angle': most_common_angle_cloudy, 'time_hm': corresponding_time_hm})

# Gruppiere nach 'seconds_since_midnight' und wende die Funktion an
most_common_angle_daily_cloudy = result_df.groupby('seconds_since_midnight').apply(get_most_common_angle_cloudy).reset_index()

# # Ergebnisse anzeigen
# print(most_common_angle_daily_cloudy)


print('#### plotting most common max. energy angles (cloudy) ####')


# Bereite die Daten für die lineare Regression vor
X = most_common_angle_daily_cloudy['seconds_since_midnight']
y = most_common_angle_daily_cloudy['angle']
X = sm.add_constant(X)  # Fügt die Konstante hinzu, da statsmodels dies nicht automatisch macht

# Führe die lineare Regression durch
model = sm.OLS(y, X).fit()

# Berechne die Vorhersagen für die Regressionsgerade
predictions = model.predict(X)

# Formatierung des Timeframes für den Titel
timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

# Erstelle ein Plotly Scatter-Plot mit angepasstem Titel
fig = px.scatter(most_common_angle_daily_cloudy, x='time_hm', y='angle',
                  title=f'Most common max. energy yield angles (cloudy) {timeframe}',
                  labels={'time_hm': 'Time', 'angle': 'Angle'})

# Füge die Regressionsgerade zum Plot hinzu
fig.add_trace(
    go.Scatter(x=most_common_angle_daily_cloudy['time_hm'], y=predictions,
                mode='lines', name='Regressionsgerade')
)

print('#### saving plot as HTML document ####\n')

# Zum Speichern als HTML
# Konstruiere den Dateinamen dynamisch
file_name = f"most_common_angles_cloudy_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

# Überprüfe, ob die Datei bereits vorhanden ist
if not os.path.exists(file_path):
    # Speichere die Datei mit dem dynamischen Dateinamen
    fig.write_html(file_path)
else:
    print("Die Datei existiert bereits.\n")


#%% average angle clearsky

print('#### calculating average max. energy angles (clearsky) ####')

all_seconds_clearsky = []
for single_date in pd.date_range(start=start_date, end=end_date):
    date_str = single_date.strftime("%Y-%m-%d")
    if date_str in daily_max_clearsky:
        df = daily_max_clearsky[date_str]
        # Bestimme den nächsten Wert in thirty_min_day für jeden Eintrag in seconds_since_midnight
        df['seconds_since_midnight'] = df['seconds_since_midnight'].apply(lambda x: min(thirty_min_day, key=lambda y: abs(y - x)))

        # Erstelle eine Maske für Zeilen, die spezifische Winkel enthalten
        mask = df['angle'].isin(specific_angles)
        filtered_df = df[mask]

        # Füge die gefilterten Daten in die Liste hinzu
        all_seconds_clearsky.extend(filtered_df[['seconds_since_midnight', 'angle']].values.tolist())

# Konvertiere die gesammelten Daten in ein DataFrame
all_seconds_clearsky_df = pd.DataFrame(all_seconds_clearsky, columns=['seconds_since_midnight', 'angle'])

counts = all_seconds_clearsky_df.groupby('seconds_since_midnight').size()

# Filtern der Gruppen, die mehr als 10 Einträge haben
valid_seconds = counts[counts > 5].index

# Berechnen des Durchschnittswinkels nur für die gefilterten 'seconds_since_midnight'-Werte
average_angle_clearsky = all_seconds_clearsky_df[all_seconds_clearsky_df['seconds_since_midnight'].isin(valid_seconds)] \
    .groupby('seconds_since_midnight')['angle'].mean().reset_index()

# Sortiere das DataFrame nach seconds_since_midnight (sollte bereits sortiert sein, aber zur Sicherheit)
average_angle_clearsky = average_angle_clearsky.sort_values(by='seconds_since_midnight')

# Umformatierung von Sekunden in hh:mm
average_angle_clearsky['time_format'] = average_angle_clearsky['seconds_since_midnight'].apply(lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}")

print('#### plotting average max. energy angles (clearsky) ####')

# Bereite die Daten für die lineare Regression vor
X = average_angle_clearsky['seconds_since_midnight']
y = average_angle_clearsky['angle']

# Stellen Sie sicher, dass keine NaN-Werte vorhanden sind und dass alle Werte numerisch sind
X = pd.to_numeric(X, errors='coerce').dropna()
y = pd.to_numeric(y, errors='coerce').dropna()

# Da wir X und y bereinigt haben, müssen wir sicherstellen, dass ihre Indizes übereinstimmen
valid_index = X.index.intersection(y.index)
X = X.loc[valid_index]
y = y.loc[valid_index]

# Debugging-Ausgaben
print("valid_index:", valid_index)
print("X nach Indexanpassung:", X.head())
print("y nach Indexanpassung:", y.head())

# Sicherstellen, dass X und y nicht leer sind
if X.empty or y.empty:
    print("Keine Daten nach Bereinigung vorhanden.")
else:
    # Konvertiere in numpy Arrays
    X = sm.add_constant(X)  # Fügt die Konstante hinzu, da statsmodels dies nicht automatisch macht

    # Debugging-Ausgaben
    print("X für Regression:", X.head())
    print("y für Regression:", y.head())

    # Führe die lineare Regression durch
    model = sm.OLS(y, X).fit()

    # Berechne die Vorhersagen für die Regressionsgerade
    predictions = model.predict(X)

    # Formatierung des Timeframes für den Titel
    timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

    # Erstelle den Plotly Scatter-Plot mit angepasstem Titel
    fig = px.scatter(average_angle_clearsky, x='time_format', y='angle',
                      title=f'Average angle over time with regression line (clearsky) {timeframe}',
                      labels={'time_format': 'Time', 'angle': 'Angle'})

    # Füge die Regressionsgerade zum Plot hinzu
    fig.add_trace(
        go.Scatter(x=average_angle_clearsky['time_format'], y=predictions,
                   mode='lines', name='Regressionsgerade')
    )

    print('#### saving plot as HTML document ####\n')

    # Konstruiere den Dateinamen dynamisch
    file_name = f"average_angles_clearsky_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
    file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

    # Überprüfe, ob die Datei bereits vorhanden ist
    if not os.path.exists(file_path):
        # Speichere die Datei mit dem dynamischen Dateinamen
        fig.write_html(file_path)
    else:
        print("Die Datei existiert bereits.\n")

#%% average angle cloudy

print('#### calculating average max. energy angles (cloudy) ####')

## Sammle alle relevanten Daten
all_seconds_cloudy = []
for single_date in pd.date_range(start=start_date, end=end_date):
    date_str = single_date.strftime("%Y-%m-%d")
    if date_str in daily_max_cloudy:
        df = daily_max_cloudy[date_str]
        df['seconds_since_midnight'] = df['seconds_since_midnight'].apply(lambda x: min(thirty_min_day, key=lambda y: abs(y - x)))
        for _, row in df.iterrows():
            if row['angle'] in specific_angles:
                all_seconds_cloudy.append((row['seconds_since_midnight'], row['angle']))

# Konvertiere die gesammelten Daten in ein DataFrame
all_seconds_cloudy_df = pd.DataFrame(all_seconds_cloudy, columns=['seconds_since_midnight', 'angle'])

counts = all_seconds_cloudy_df.groupby('seconds_since_midnight').size()

# Filtern der Gruppen, die mehr als 10 Einträge haben
valid_seconds = counts[counts > 5].index

# Berechnen des Durchschnittswinkels nur für die gefilterten 'seconds_since_midnight'-Werte
average_angle_cloudy = all_seconds_cloudy_df[all_seconds_cloudy_df['seconds_since_midnight'].isin(valid_seconds)] \
    .groupby('seconds_since_midnight')['angle'].mean().reset_index()

# Sortiere das DataFrame nach seconds_since_midnight (sollte bereits sortiert sein, aber zur Sicherheit)
average_angle_cloudy = average_angle_cloudy.sort_values(by='seconds_since_midnight')

# Umformatierung von Sekunden in hh:mm
average_angle_cloudy['time_format'] = average_angle_cloudy['seconds_since_midnight'].apply(lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}")

print('#### plotting average max. energy angles (cloudy) ####')

# Bereite die Daten für die lineare Regression vor
X = average_angle_cloudy['seconds_since_midnight']
y = average_angle_cloudy['angle']

# Stellen Sie sicher, dass keine NaN-Werte vorhanden sind und dass alle Werte numerisch sind
X = pd.to_numeric(X, errors='coerce').dropna()
y = pd.to_numeric(y, errors='coerce').dropna()

# Da wir X und y bereinigt haben, müssen wir sicherstellen, dass ihre Indizes übereinstimmen
valid_index = X.index.intersection(y.index)
X = X.loc[valid_index]
y = y.loc[valid_index]

# Sicherstellen, dass X und y nicht leer sind
if X.empty or y.empty:
    print("Keine Daten nach Bereinigung vorhanden.")
else:
    # Konvertiere in numpy Arrays
    X = sm.add_constant(X)  # Fügt die Konstante hinzu, da statsmodels dies nicht automatisch macht

    # Führe die lineare Regression durch
    model = sm.OLS(y, X).fit()

    # Berechne die Vorhersagen für die Regressionsgerade
    predictions = model.predict(X)

    # Formatierung des Timeframes für den Titel
    timeframe = f"({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"

    # Erstelle den Plotly Scatter-Plot mit angepasstem Titel
    fig = px.scatter(average_angle_cloudy, x='time_format', y='angle',
                      title=f'Average angle over time with regression line (cloudy) {timeframe}',
                      labels={'time_format': 'Time', 'angle': 'Angle'})

    # Füge die Regressionsgerade zum Plot hinzu
    fig.add_trace(
        go.Scatter(x=average_angle_cloudy['time_format'], y=predictions,
                   mode='lines', name='Regressionsgerade')
    )

    print('#### saving plot as HTML document ####\n')

    # Konstruiere den Dateinamen dynamisch
    file_name = f"average_angles_cloudy_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
    file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Angles/{file_name}"

    # Überprüfe, ob die Datei bereits vorhanden ist
    if not os.path.exists(file_path):
        # Speichere die Datei mit dem dynamischen Dateinamen
        fig.write_html(file_path)
    else:
        print("Die Datei existiert bereits.\n")
        

#%% dataframes for algorithms
zero_degree = [ #30min
    (5 * 3600 + 900, 0),     # 5:15 Uhr
    (5 * 3600 + 2700, 0),    # 5:45 Uhr
    (6 * 3600 + 900, 0),     # 6:15 Uhr
    (6 * 3600 + 2700, 0),    # 6:45 Uhr
    (7 * 3600 + 900, 0),     # 7:15 Uhr
    (7 * 3600 + 2700, 0),    # 7:45 Uhr
    (8 * 3600 + 900, 0),     # 8:15 Uhr
    (8 * 3600 + 2700, 0),    # 8:45 Uhr
    (9 * 3600 + 900, 0),     # 9:15 Uhr
    (9 * 3600 + 2700, 0),    # 9:45 Uhr
    (10 * 3600 + 900, 0),    # 10:15 Uhr
    (10 * 3600 + 2700, 0),   # 10:45 Uhr
    (11 * 3600 + 900, 0),    # 11:15 Uhr
    (11 * 3600 + 2700, 0),   # 11:45 Uhr
    (12 * 3600 + 900, 0),    # 12:15 Uhr
    (12 * 3600 + 2700, 0),   # 12:45 Uhr
    (13 * 3600 + 900, 0),     # 13:15 Uhr
    (13 * 3600 + 2700, 0),    # 13:45 Uhr
    (14 * 3600 + 900, 0),     # 14:15 Uhr
    (14 * 3600 + 2700, 0),    # 14:45 Uhr
    (15 * 3600 + 900, 0),     # 15:15 Uhr
    (15 * 3600 + 2700, 0),    # 15:45 Uhr
    (16 * 3600 + 900, 0),     # 16:15 Uhr
    (16 * 3600 + 2700, 0),    # 16:45 Uhr
    (17 * 3600 + 900, 0),     # 17:15 Uhr
    (17 * 3600 + 2700, 0),    # 17:45 Uhr
    (18 * 3600 + 900, 0),     # 18:15 Uhr
    (18 * 3600 + 2700, 0),    # 18:45 Uhr
    (19 * 3600 + 900, 0),     # 19:15 Uhr
    (19 * 3600 + 2700, 0),    # 19:45 Uhr
    (20 * 3600 + 900, 0),     # 20:15 Uhr
    (20 * 3600 + 2700, 0)     # 20:45 Uhr
]    
df_zero_degree = pd.DataFrame(zero_degree, columns=['seconds_since_midnight', 'angle'])
flipflop_five_degree = [ #30min
    (5 * 3600 + 900, -5),     # 5:15 Uhr
    (5 * 3600 + 2700, -5),    # 5:45 Uhr
    (6 * 3600 + 900, -5),     # 6:15 Uhr
    (6 * 3600 + 2700, -5),    # 6:45 Uhr
    (7 * 3600 + 900, -5),     # 7:15 Uhr
    (7 * 3600 + 2700, -5),    # 7:45 Uhr
    (8 * 3600 + 900, -5),     # 8:15 Uhr
    (8 * 3600 + 2700, -5),    # 8:45 Uhr
    (9 * 3600 + 900, -5),     # 9:15 Uhr
    (9 * 3600 + 2700, -5),    # 9:45 Uhr
    (10 * 3600 + 900, -5),    # 10:15 Uhr
    (10 * 3600 + 2700, -5),   # 10:45 Uhr
    (11 * 3600 + 900, -5),    # 11:15 Uhr
    (11 * 3600 + 2700, -5),   # 11:45 Uhr
    (12 * 3600 + 900, -5),    # 12:15 Uhr
    (12 * 3600 + 2700, -5),   # 12:45 Uhr
    (13 * 3600 + 900, 5),     # 13:15 Uhr
    (13 * 3600 + 2700, 5),    # 13:45 Uhr
    (14 * 3600 + 900, 5),     # 14:15 Uhr
    (14 * 3600 + 2700, 5),    # 14:45 Uhr
    (15 * 3600 + 900, 5),     # 15:15 Uhr
    (15 * 3600 + 2700, 5),    # 15:45 Uhr
    (16 * 3600 + 900, 5),     # 16:15 Uhr
    (16 * 3600 + 2700, 5),    # 16:45 Uhr
    (17 * 3600 + 900, 5),     # 17:15 Uhr
    (17 * 3600 + 2700, 5),    # 17:45 Uhr
    (18 * 3600 + 900, 5),     # 18:15 Uhr
    (18 * 3600 + 2700, 5),    # 18:45 Uhr
    (19 * 3600 + 900, 5),     # 19:15 Uhr
    (19 * 3600 + 2700, 5),    # 19:45 Uhr
    (20 * 3600 + 900, 5),     # 20:15 Uhr
    (20 * 3600 + 2700, 5)     # 20:45 Uhr
]
df_flip_flop_five_degree = pd.DataFrame(flipflop_five_degree, columns=['seconds_since_midnight', 'angle'])
flipflop_ten_degree = [ #30min
    (5 * 3600 + 900, -10),     # 5:15 Uhr
    (5 * 3600 + 2700, -10),    # 5:45 Uhr
    (6 * 3600 + 900, -10),     # 6:15 Uhr
    (6 * 3600 + 2700, -10),    # 6:45 Uhr
    (7 * 3600 + 900, -10),     # 7:15 Uhr
    (7 * 3600 + 2700, -10),    # 7:45 Uhr
    (8 * 3600 + 900, -10),     # 8:15 Uhr
    (8 * 3600 + 2700, -10),    # 8:45 Uhr
    (9 * 3600 + 900, -10),     # 9:15 Uhr
    (9 * 3600 + 2700, -10),    # 9:45 Uhr
    (10 * 3600 + 900, -10),    # 10:15 Uhr
    (10 * 3600 + 2700, -10),   # 10:45 Uhr
    (11 * 3600 + 900, -10),    # 11:15 Uhr
    (11 * 3600 + 2700, -10),   # 11:45 Uhr
    (12 * 3600 + 900, -10),    # 12:15 Uhr
    (12 * 3600 + 2700, -10),   # 12:45 Uhr
    (13 * 3600 + 900, 10),     # 13:15 Uhr
    (13 * 3600 + 2700, 10),    # 13:45 Uhr
    (14 * 3600 + 900, 10),     # 14:15 Uhr
    (14 * 3600 + 2700, 10),    # 14:45 Uhr
    (15 * 3600 + 900, 10),     # 15:15 Uhr
    (15 * 3600 + 2700, 10),    # 15:45 Uhr
    (16 * 3600 + 900, 10),     # 16:15 Uhr
    (16 * 3600 + 2700, 10),    # 16:45 Uhr
    (17 * 3600 + 900, 10),     # 17:15 Uhr
    (17 * 3600 + 2700, 10),    # 17:45 Uhr
    (18 * 3600 + 900, 10),     # 18:15 Uhr
    (18 * 3600 + 2700, 10),    # 18:45 Uhr
    (19 * 3600 + 900, 10),     # 19:15 Uhr
    (19 * 3600 + 2700, 10),    # 19:45 Uhr
    (20 * 3600 + 900, 10),     # 20:15 Uhr
    (20 * 3600 + 2700, 10)     # 20:45 Uhr
]    
df_flip_flop_ten_degree = pd.DataFrame(flipflop_ten_degree, columns=['seconds_since_midnight', 'angle'])
tracking_five_degree = [ #30min
    (5 * 3600 + 900, -5),     # 5:15 Uhr
    (5 * 3600 + 2700, -5),    # 5:45 Uhr
    (6 * 3600 + 900, -5),     # 6:15 Uhr
    (6 * 3600 + 2700, -5),    # 6:45 Uhr
    (7 * 3600 + 900, -5),     # 7:15 Uhr
    (7 * 3600 + 2700, -5),    # 7:45 Uhr
    (8 * 3600 + 900, -5),     # 8:15 Uhr
    (8 * 3600 + 2700, -5),    # 8:45 Uhr
    (9 * 3600 + 900, -5),     # 9:15 Uhr
    (9 * 3600 + 2700, -5),    # 9:45 Uhr
    (10 * 3600 + 900, -5),    # 10:15 Uhr
    (10 * 3600 + 2700, -5),   # 10:45 Uhr
    (11 * 3600 + 900, 0),    # 11:15 Uhr
    (11 * 3600 + 2700, 0),   # 11:45 Uhr
    (12 * 3600 + 900, 0),    # 12:15 Uhr
    (12 * 3600 + 2700, 0),   # 12:45 Uhr
    (13 * 3600 + 900, 0),     # 13:15 Uhr
    (13 * 3600 + 2700, 5),    # 13:45 Uhr
    (14 * 3600 + 900, 5),     # 14:15 Uhr
    (14 * 3600 + 2700, 5),    # 14:45 Uhr
    (15 * 3600 + 900, 5),     # 15:15 Uhr
    (15 * 3600 + 2700, 5),    # 15:45 Uhr
    (16 * 3600 + 900, 5),     # 16:15 Uhr
    (16 * 3600 + 2700, 5),    # 16:45 Uhr
    (17 * 3600 + 900, 5),     # 17:15 Uhr
    (17 * 3600 + 2700, 5),    # 17:45 Uhr
    (18 * 3600 + 900, 5),     # 18:15 Uhr
    (18 * 3600 + 2700, 5),    # 18:45 Uhr
    (19 * 3600 + 900, 5),     # 19:15 Uhr
    (19 * 3600 + 2700, 5),    # 19:45 Uhr
    (20 * 3600 + 900, 5),     # 20:15 Uhr
    (20 * 3600 + 2700, 5)     # 20:45 Uhr
]   
df_tracking_five_degree = pd.DataFrame(tracking_five_degree, columns=['seconds_since_midnight', 'angle'])
tracking_ten_degree = [ #30min
    (5 * 3600 + 900, -10),     # 5:15 Uhr
    (5 * 3600 + 2700, -10),    # 5:45 Uhr
    (6 * 3600 + 900, -10),     # 6:15 Uhr
    (6 * 3600 + 2700, -10),    # 6:45 Uhr
    (7 * 3600 + 900, -10),     # 7:15 Uhr
    (7 * 3600 + 2700, -10),    # 7:45 Uhr
    (8 * 3600 + 900, -10),     # 8:15 Uhr
    (8 * 3600 + 2700, -10),    # 8:45 Uhr
    (9 * 3600 + 900, -5),     # 9:15 Uhr
    (9 * 3600 + 2700, -5),    # 9:45 Uhr
    (10 * 3600 + 900, -5),    # 10:15 Uhr
    (10 * 3600 + 2700, -5),   # 10:45 Uhr
    (11 * 3600 + 900, 0),    # 11:15 Uhr
    (11 * 3600 + 2700, 0),   # 11:45 Uhr
    (12 * 3600 + 900, 0),    # 12:15 Uhr
    (12 * 3600 + 2700, 0),   # 12:45 Uhr
    (13 * 3600 + 900, 0),     # 13:15 Uhr
    (13 * 3600 + 2700, 0),    # 13:45 Uhr
    (14 * 3600 + 900, 5),     # 14:15 Uhr
    (14 * 3600 + 2700, 5),    # 14:45 Uhr
    (15 * 3600 + 900, 5),     # 15:15 Uhr
    (15 * 3600 + 2700, 5),    # 15:45 Uhr
    (16 * 3600 + 900, 10),     # 16:15 Uhr
    (16 * 3600 + 2700, 10),    # 16:45 Uhr
    (17 * 3600 + 900, 10),     # 17:15 Uhr
    (17 * 3600 + 2700, 10),    # 17:45 Uhr
    (18 * 3600 + 900, 10),     # 18:15 Uhr
    (18 * 3600 + 2700, 10),    # 18:45 Uhr
    (19 * 3600 + 900, 10),     # 19:15 Uhr
    (19 * 3600 + 2700, 10),    # 19:45 Uhr
    (20 * 3600 + 900, 10),     # 20:15 Uhr
    (20 * 3600 + 2700, 10)     # 20:45 Uhr
]   
df_tracking_ten_degree = pd.DataFrame(tracking_ten_degree, columns=['seconds_since_midnight', 'angle'])
tracking_twenty_degree = [ #30 min
    (5 * 3600 + 900, -20),     # 5:15 Uhr
    (5 * 3600 + 2700, -20),    # 5:45 Uhr
    (6 * 3600 + 900, -20),     # 6:15 Uhr
    (6 * 3600 + 2700, -20),    # 6:45 Uhr
    (7 * 3600 + 900, -10),     # 7:15 Uhr
    (7 * 3600 + 2700, -10),    # 7:45 Uhr
    (8 * 3600 + 900, -10),     # 8:15 Uhr
    (8 * 3600 + 2700, -10),    # 8:45 Uhr
    (9 * 3600 + 900, -5),     # 9:15 Uhr
    (9 * 3600 + 2700, -5),    # 9:45 Uhr
    (10 * 3600 + 900, -5),    # 10:15 Uhr
    (10 * 3600 + 2700, -5),   # 10:45 Uhr
    (11 * 3600 + 900, 0),    # 11:15 Uhr
    (11 * 3600 + 2700, 0),   # 11:45 Uhr
    (12 * 3600 + 900, 0),    # 12:15 Uhr
    (12 * 3600 + 2700, 0),   # 12:45 Uhr
    (13 * 3600 + 900, 0),     # 13:15 Uhr
    (13 * 3600 + 2700, 0),    # 13:45 Uhr
    (14 * 3600 + 900, 5),     # 14:15 Uhr
    (14 * 3600 + 2700, 5),    # 14:45 Uhr
    (15 * 3600 + 900, 10),     # 15:15 Uhr
    (15 * 3600 + 2700, 10),    # 15:45 Uhr
    (16 * 3600 + 900, 10),     # 16:15 Uhr
    (16 * 3600 + 2700, 10),    # 16:45 Uhr
    (17 * 3600 + 900, 20),     # 17:15 Uhr
    (17 * 3600 + 2700, 20),    # 17:45 Uhr
    (18 * 3600 + 900, 20),     # 18:15 Uhr
    (18 * 3600 + 2700, 20),    # 18:45 Uhr
    (19 * 3600 + 900, 20),     # 19:15 Uhr
    (19 * 3600 + 2700, 20),    # 19:45 Uhr
    (20 * 3600 + 900, 20),     # 20:15 Uhr
    (20 * 3600 + 2700, 20)     # 20:45 Uhr
] 
df_tracking_twenty_degree = pd.DataFrame(tracking_twenty_degree, columns=['seconds_since_midnight', 'angle'])


backtracking = [ #30 min
    (5 * 3600 + 900, 0),      # 5:15 Uhr
    (5 * 3600 + 2700, 0),     # 5:45 Uhr
    (6 * 3600 + 900, 0),      # 6:15 Uhr
    (6 * 3600 + 2700, 0),     # 6:45 Uhr
    (7 * 3600 + 900, 0),      # 7:15 Uhr
    (7 * 3600 + 2700, 0),     # 7:45 Uhr
    (8 * 3600 + 900, 0),      # 8:15 Uhr
    (8 * 3600 + 2700, 0),     # 8:45 Uhr
    (9 * 3600 + 900, 0),      # 9:15 Uhr
    (9 * 3600 + 2700, 0),     # 9:45 Uhr
    (10 * 3600 + 900, -5),    # 10:15 Uhr
    (10 * 3600 + 2700, -5),   # 10:45 Uhr
    (11 * 3600 + 900, -5),    # 11:15 Uhr
    (11 * 3600 + 2700, -5),   # 11:45 Uhr
    (12 * 3600 + 900, 0),     # 12:15 Uhr
    (12 * 3600 + 2700, 0),    # 12:45 Uhr
    (13 * 3600 + 900, 0),     # 13:15 Uhr
    (13 * 3600 + 2700, 5),    # 13:45 Uhr
    (14 * 3600 + 900, 5),     # 14:15 Uhr
    (14 * 3600 + 2700, 5),    # 14:45 Uhr
    (15 * 3600 + 900, 5),     # 15:15 Uhr
    (15 * 3600 + 2700, 5),    # 15:45 Uhr
    (16 * 3600 + 900, 5),     # 16:15 Uhr
    (16 * 3600 + 2700, 5),    # 16:45 Uhr
    (17 * 3600 + 900, 5),     # 17:15 Uhr
    (17 * 3600 + 2700, 5),    # 17:45 Uhr
    (18 * 3600 + 900, 0),     # 18:15 Uhr
    (18 * 3600 + 2700, 0),    # 18:45 Uhr
    (19 * 3600 + 900, 0),     # 19:15 Uhr
    (19 * 3600 + 2700, 0),    # 19:45 Uhr
    (20 * 3600 + 900, 0),     # 20:15 Uhr
    (20 * 3600 + 2700, 0)     # 20:45 Uhr
] 
df_backtracking = pd.DataFrame(backtracking, columns=['seconds_since_midnight', 'angle'])
#%% create dict for energies tracking algorithms
print('#### calculate tracking energies into dict ####\n')

import pandas as pd
import numpy as np


# Erstelle das yield_tracking Dictionary
yield_tracking = {}

# Funktion, um den Index des nächstgelegenen Wertes zu finden
def find_nearest_index(energy_df, second):
    if energy_df.empty:
        return pd.NA
    else:
        differences = np.abs(energy_df['seconds_since_midnight'] - second)
        return differences.idxmin()

# Durchlaufe alle Daten in yield_dict
for date, angles_dict in yield_dict.items():
    yield_tracking[date] = {}

    # Für zero_degree
    entries_zero = []
    for angle, angle_df in df_zero_degree.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_zero.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    zero_degree_df = pd.DataFrame(entries_zero)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not zero_degree_df.empty and all(col in zero_degree_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        zero_degree_df = zero_degree_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        zero_degree_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['zero_degree'] = zero_degree_df

    # Für flip_flop_five_degree
    entries_flip_flop_five_degree = []
    for angle, angle_df in df_flip_flop_five_degree.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_flip_flop_five_degree.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    flip_flop_five_degree_df = pd.DataFrame(entries_flip_flop_five_degree)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not flip_flop_five_degree_df.empty and all(col in flip_flop_five_degree_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        flip_flop_five_degree_df = flip_flop_five_degree_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        flip_flop_five_degree_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['flip_flop_five_degree'] = flip_flop_five_degree_df

    # Für flip_flop_ten_degree
    entries_flip_flop_ten_degree = []
    for angle, angle_df in df_flip_flop_ten_degree.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_flip_flop_ten_degree.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    flip_flop_ten_degree_df = pd.DataFrame(entries_flip_flop_ten_degree)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not flip_flop_ten_degree_df.empty and all(col in flip_flop_ten_degree_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        flip_flop_ten_degree_df = flip_flop_ten_degree_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        flip_flop_ten_degree_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['flip_flop_ten_degree'] = flip_flop_ten_degree_df
    
    # Für tracking_five_degree
    entries_tracking_five_degree = []
    for angle, angle_df in df_tracking_five_degree.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_tracking_five_degree.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    tracking_five_degree_df = pd.DataFrame(entries_tracking_five_degree)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not tracking_five_degree_df.empty and all(col in tracking_five_degree_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        tracking_five_degree_df = tracking_five_degree_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        tracking_five_degree_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['tracking_five_degree'] = tracking_five_degree_df  

    # Für tracking_ten_degree
    entries_tracking_ten_degree = []
    for angle, angle_df in df_tracking_ten_degree.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_tracking_ten_degree.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    tracking_ten_degree_df = pd.DataFrame(entries_tracking_ten_degree)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not tracking_ten_degree_df.empty and all(col in tracking_ten_degree_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        tracking_ten_degree_df = tracking_ten_degree_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        tracking_ten_degree_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['tracking_ten_degree'] = tracking_ten_degree_df  

    # Für tracking_twenty_degree
    entries_tracking_twenty_degree = []
    for angle, angle_df in df_tracking_twenty_degree.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_tracking_twenty_degree.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    tracking_twenty_degree_df = pd.DataFrame(entries_tracking_twenty_degree)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not tracking_twenty_degree_df.empty and all(col in tracking_twenty_degree_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        tracking_twenty_degree_df = tracking_twenty_degree_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        tracking_twenty_degree_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['tracking_twenty_degree'] = tracking_twenty_degree_df
    
    # Für backtracking
    entries_backtracking = []
    for angle, angle_df in df_backtracking.groupby('angle'):
        if angle in angles_dict:
            energy_df = angles_dict[angle]
            for index, time_row in angle_df.iterrows():
                nearest_index = find_nearest_index(energy_df, time_row['seconds_since_midnight'])
                if pd.notna(nearest_index):
                    energy = energy_df.loc[nearest_index, 'energy']
                    time_formatted = energy_df.loc[nearest_index, 'time_formatted']  # Hier wird die Spalte time_formatted hinzugefügt
                    entries_backtracking.append({'seconds_since_midnight': time_row['seconds_since_midnight'], 'angle': angle, 'energy': energy, 'time_formatted': time_formatted})

    backtracking_df = pd.DataFrame(entries_backtracking)
    
    # Überprüfen, ob die Liste nicht leer ist und die Spalten vorhanden sind
    if not backtracking_df.empty and all(col in backtracking_df.columns for col in ['seconds_since_midnight', 'energy', 'time_formatted']):
        # Entfernen Sie doppelte Einträge und sortieren Sie sie nur, wenn sie vorhanden sind
        backtracking_df = backtracking_df.drop_duplicates(subset=['energy']).sort_values('seconds_since_midnight')
    else:
        # Handle den Fall, wenn die Liste leer ist oder erforderliche Spalten nicht vorhanden sind
        backtracking_df = pd.DataFrame(columns=['seconds_since_midnight', 'angle', 'energy', 'time_formatted'])
    
    yield_tracking[date]['backtracking'] = backtracking_df



# Durchlaufe alle Daten in daily_max_energy_dict
for date, max_energy_df in daily_max_energy_dict.items():
    # Stelle sicher, dass das Datum in yield_tracking existiert und ein Dictionary zugewiesen ist
    if date not in yield_tracking:
        yield_tracking[date] = {}
    
    # Füge das 'daily_max_energy'-DataFrame zum 'yield_tracking'-Dictionary hinzu, ohne die bereits vorhandenen DFs zu überschreiben
    yield_tracking[date]['max_energy'] = max_energy_df.copy()




 

#%% create yield_tracking_clearsky
print('#### create yield_tracking_clearsky dictionnary ####\n')

# new dict for clearsky dates
import datetime

yield_tracking_clearsky = {}
clearsky_dates = set(clearsky)  # Konvertiere die Liste zu einem Set für eine effizientere Suche

# Durchlaufe alle Daten in yield_tracking
for date_str, data in yield_tracking.items():
    # Konvertiere den String zu einem datetime.date-Objekt
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Überprüfe, ob das Datum in der Liste der bewölkten Tage ist UND ob das Datum in yield_tracking vorhanden ist
    if date_obj in clearsky_dates and date_str in yield_tracking:
        # Kopiere die Daten in das neue Dictionary
        yield_tracking_clearsky[date_str] = data.copy()  # Verwende .copy(), um sicherzustellen, dass keine Referenzen kopiert werden

#%% create yield_tracking_cloudy
print('#### create yield_tracking_cloudy dictionnary ####\n')
# Neues Dictionary für bewölkte Tage
import datetime

yield_tracking_cloudy = {}
cloudy_dates = set(cloudy)  # Konvertiere die Liste zu einem Set für eine effizientere Suche

# Durchlaufe alle Daten in yield_tracking
for date_str, data in yield_tracking.items():
    # Konvertiere den String zu einem datetime.date-Objekt
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Überprüfe, ob das Datum in der Liste der bewölkten Tage ist UND ob das Datum in yield_tracking vorhanden ist
    if date_obj in cloudy_dates and date_str in yield_tracking:
        # Kopiere die Daten in das neue Dictionary
        yield_tracking_cloudy[date_str] = data.copy()  # Verwende .copy(), um sicherzustellen, dass keine Referenzen kopiert werden



#%% calculate energy over chosen dates 
print('#### calculate and plot yield in percent ####\n')
# Erstelle ein leeres Dictionary für die Summen der Energiewerte pro Schlüssel

import pandas as pd
import numpy as np
from plotly import graph_objects as go
import os

# Erstelle ein leeres DataFrame für das gesammelte Ergebnis
collected_data = pd.DataFrame()

# Durchlaufe das yield_tracking-Dictionary und sammle die Daten
for date, angles_data in yield_tracking.items():
    for angle_type, df in angles_data.items():
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(date)
        df_temp['angle_type'] = angle_type
        # Sammle die Daten
        collected_data = pd.concat([collected_data, df_temp], ignore_index=True)

# Konvertiere 'date' zu 'month' und filtere nach ausgewählten Monaten
collected_data['month'] = collected_data['date'].dt.to_period('M')
collected_data = collected_data[collected_data['month'].astype(str).isin(selected_months)]


# Berechne die monatliche Energie pro Winkeltyp
monthly_energy_by_type = collected_data.groupby(['month', 'angle_type'])['energy'].sum().reset_index()

# Konvertiere 'month' von Period zu String für die Plot-Berechnung
monthly_energy_by_type['month'] = monthly_energy_by_type['month'].astype(str)

pivot_energy = monthly_energy_by_type.pivot(index='month', columns='angle_type', values='energy')
pivot_energy.reset_index(inplace=True)

# Pivotiere die Daten für die prozentuale Darstellung
pivot_data = monthly_energy_by_type.pivot(index='month', columns='angle_type', values='energy')
pivot_data = pivot_data.sub(pivot_data['zero_degree'], axis=0).div(pivot_data['zero_degree'], axis=0).fillna(0) * 100
pivot_data['zero_degree'] = 0  # Setze zero_degree Werte auf 0%



# Vorbereiten der Daten für das Plotting
pivot_data.reset_index(inplace=True)
pivot_data_melted = pivot_data.melt(id_vars='month', var_name='angle_type', value_name='percentage_gain')

# Erstelle die Farbkarte
angle_color_map = {
    'flip_flop_five_degree': 'rgba(0,255,0,1)',  # Grün
    'flip_flop_ten_degree': 'rgba(0,0,255,1)',   # Blau
    'backtracking': 'rgba(255,0,0,1)',  # Gelb
    'tracking_five_degree': 'rgba(255,0,255,1)',  # Magenta
    'tracking_ten_degree': 'rgba(0,255,255,1)',  # Cyan
    'tracking_twenty_degree': 'rgba(128,128,128,1)',  # Grau
    'max_energy': 'rgba(255,255,0,1)'
    # Füge weitere Farben für andere Winkeltypen hinzu, falls nötig
}

# Erstelle das Balkendiagramm
fig = go.Figure()

# Füge Balken für jeden Winkeltyp hinzu, außer für 'zero_degree'
for angle_type in pivot_data_melted['angle_type'].unique():
    if angle_type != 'zero_degree':  # Überspringe 'zero_degree'
        fig.add_trace(go.Bar(
            x=pivot_data_melted[pivot_data_melted['angle_type'] == angle_type]['month'],
            y=pivot_data_melted[pivot_data_melted['angle_type'] == angle_type]['percentage_gain'],
            name=angle_type,
            marker_color=angle_color_map.get(angle_type, 'rgba(128,128,128,1)'),
        ))

# Layout-Anpassungen
fig.update_layout(
    title=f"Energie in % Abweichung vom 0 Grad Winkel ({selected_months})",
    xaxis_title='Monat',
    yaxis_title='Energieertrag (%)',
    barmode='group',
    legend_title_text='Trackingalgorithmus',
    xaxis={'type': 'category'},  # Stelle sicher, dass die x-Achse kategorische Daten erwartet
)

# Speichere den Plot als HTML-Datei
file_name = f"yield_gain_in_percent_{collected_data['date'].min().strftime('%Y-%m-%d')}_to_{collected_data['date'].max().strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Energies/{file_name}"

# Überprüfe und erstelle den Ordner, falls nicht vorhanden
folder_path = os.path.dirname(file_path)

if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")


#%% calculate energy on clearsky days over chosen dates 
print('#### calculate and plot clearsky yield in percent ####\n')
# Erstelle ein leeres Dictionary für die Summen der Energiewerte pro Schlüssel

import pandas as pd
import numpy as np
from plotly import graph_objects as go
import os


# Erstelle ein leeres DataFrame für das gesammelte Ergebnis
collected_data = pd.DataFrame()

# Durchlaufe das yield_tracking-Dictionary und sammle die Daten
for date, angles_data in yield_tracking_clearsky.items():
    for angle_type, df in angles_data.items():
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(date)
        df_temp['angle_type'] = angle_type
        # Sammle die Daten
        collected_data = pd.concat([collected_data, df_temp], ignore_index=True)

# Konvertiere 'date' zu 'month' und filtere nach ausgewählten Monaten
collected_data['month'] = collected_data['date'].dt.to_period('M')
collected_data = collected_data[collected_data['month'].astype(str).isin(selected_months)]


# Berechne die monatliche Energie pro Winkeltyp
monthly_energy_by_type = collected_data.groupby(['month', 'angle_type'])['energy'].sum().reset_index()

# Konvertiere 'month' von Period zu String für die Plot-Berechnung
monthly_energy_by_type['month'] = monthly_energy_by_type['month'].astype(str)

pivot_energy_clearsky = monthly_energy_by_type.pivot(index='month', columns='angle_type', values='energy')
pivot_energy_clearsky.reset_index(inplace=True)

# Pivotiere die Daten für die prozentuale Darstellung
pivot_data = monthly_energy_by_type.pivot(index='month', columns='angle_type', values='energy')
pivot_data = pivot_data.sub(pivot_data['zero_degree'], axis=0).div(pivot_data['zero_degree'], axis=0).fillna(0) * 100
pivot_data['zero_degree'] = 0  # Setze zero_degree Werte auf 0%

# Vorbereiten der Daten für das Plotting
pivot_data.reset_index(inplace=True)
pivot_data_melted = pivot_data.melt(id_vars='month', var_name='angle_type', value_name='percentage_gain')

# Erstelle die Farbkarte
angle_color_map = {
    'flip_flop_five_degree': 'rgba(0,255,0,1)',  # Grün
    'flip_flop_ten_degree': 'rgba(0,0,255,1)',   # Blau
    'backtracking': 'rgba(255,0,0,1)',  # Gelb
    'tracking_five_degree': 'rgba(255,0,255,1)',  # Magenta
    'tracking_ten_degree': 'rgba(0,255,255,1)',  # Cyan
    'tracking_twenty_degree': 'rgba(128,128,128,1)',  # Grau
    'max_energy': 'rgba(255,255,0,1)'
    # Füge weitere Farben für andere Winkeltypen hinzu, falls nötig
}

# Erstelle das Balkendiagramm
fig = go.Figure()

# Füge Balken für jeden Winkeltyp hinzu, außer für 'zero_degree'
for angle_type in pivot_data_melted['angle_type'].unique():
    if angle_type != 'zero_degree':  # Überspringe 'zero_degree'
        fig.add_trace(go.Bar(
            x=pivot_data_melted[pivot_data_melted['angle_type'] == angle_type]['month'],
            y=pivot_data_melted[pivot_data_melted['angle_type'] == angle_type]['percentage_gain'],
            name=angle_type,
            marker_color=angle_color_map.get(angle_type, 'rgba(128,128,128,1)'),
        ))

# Layout-Anpassungen
fig.update_layout(
    title=f"Energie an clearsky days in % Abweichung vom 0 Grad Winkel ({selected_months})",
    xaxis_title='Monat',
    yaxis_title='Energieertrag (%)',
    barmode='group',
    legend_title_text='Trackingalgorithmus',
    xaxis={'type': 'category'},  # Stelle sicher, dass die x-Achse kategorische Daten erwartet
)

# Speichere den Plot als HTML-Datei
file_name = f"yield_gain_clearsky_in_percent_{collected_data['date'].min().strftime('%Y-%m-%d')}_to_{collected_data['date'].max().strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Energies/{file_name}"

# Überprüfe und erstelle den Ordner, falls nicht vorhanden
folder_path = os.path.dirname(file_path)

if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")

#%% calculate energy on cloudy days over chosen dates
print('#### calculate and plot cloudy yield in percent ####\n')
# Erstelle ein leeres Dictionary für die Summen der Energiewerte pro Schlüssel

import pandas as pd
import numpy as np
from plotly import graph_objects as go
import os

# Erstelle ein leeres DataFrame für das gesammelte Ergebnis
collected_data = pd.DataFrame()

# Durchlaufe das yield_tracking-Dictionary und sammle die Daten
for date, angles_data in yield_tracking_cloudy.items():
    for angle_type, df in angles_data.items():
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(date)
        df_temp['angle_type'] = angle_type
        # Sammle die Daten
        collected_data = pd.concat([collected_data, df_temp], ignore_index=True)

# Konvertiere 'date' zu 'month' und filtere nach ausgewählten Monaten
collected_data['month'] = collected_data['date'].dt.to_period('M')
collected_data = collected_data[collected_data['month'].astype(str).isin(selected_months)]


# Berechne die monatliche Energie pro Winkeltyp
monthly_energy_by_type = collected_data.groupby(['month', 'angle_type'])['energy'].sum().reset_index()

# Konvertiere 'month' von Period zu String für die Plot-Berechnung
monthly_energy_by_type['month'] = monthly_energy_by_type['month'].astype(str)

pivot_energy_cloudy = monthly_energy_by_type.pivot(index='month', columns='angle_type', values='energy')
pivot_energy_cloudy.reset_index(inplace=True)

# Pivotiere die Daten für die prozentuale Darstellung
pivot_data = monthly_energy_by_type.pivot(index='month', columns='angle_type', values='energy')
pivot_data = pivot_data.sub(pivot_data['zero_degree'], axis=0).div(pivot_data['zero_degree'], axis=0).fillna(0) * 100
pivot_data['zero_degree'] = 0  # Setze zero_degree Werte auf 0%

# Vorbereiten der Daten für das Plotting
pivot_data.reset_index(inplace=True)
pivot_data_melted = pivot_data.melt(id_vars='month', var_name='angle_type', value_name='percentage_gain')

# Erstelle die Farbkarte
angle_color_map = {
    'flip_flop_five_degree': 'rgba(0,255,0,1)',  # Grün
    'flip_flop_ten_degree': 'rgba(0,0,255,1)',   # Blau
    'backtracking': 'rgba(255,0,0,1)',  # Gelb
    'tracking_five_degree': 'rgba(255,0,255,1)',  # Magenta
    'tracking_ten_degree': 'rgba(0,255,255,1)',  # Cyan
    'tracking_twenty_degree': 'rgba(128,128,128,1)',  # Grau
    'max_energy': 'rgba(255,255,0,1)'
    # Füge weitere Farben für andere Winkeltypen hinzu, falls nötig
}

# Erstelle das Balkendiagramm
fig = go.Figure()

# Füge Balken für jeden Winkeltyp hinzu, außer für 'zero_degree'
for angle_type in pivot_data_melted['angle_type'].unique():
    if angle_type != 'zero_degree':  # Überspringe 'zero_degree'
        fig.add_trace(go.Bar(
            x=pivot_data_melted[pivot_data_melted['angle_type'] == angle_type]['month'],
            y=pivot_data_melted[pivot_data_melted['angle_type'] == angle_type]['percentage_gain'],
            name=angle_type,
            marker_color=angle_color_map.get(angle_type, 'rgba(128,128,128,1)'),
        ))

# Layout-Anpassungen
fig.update_layout(
    title=f"Energie an cloudy days in % Abweichung vom 0 Grad Winkel ({selected_months})",
    xaxis_title='Monat',
    yaxis_title='Energieertrag (%)',
    barmode='group',
    legend_title_text='Trackingalgorithmus',
    xaxis={'type': 'category'},  # Stelle sicher, dass die x-Achse kategorische Daten erwartet
)

# Speichere den Plot als HTML-Datei
file_name = f"yield_gain_cloudy_in_percent_{collected_data['date'].min().strftime('%Y-%m-%d')}_to_{collected_data['date'].max().strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Energies/{file_name}"

# Überprüfe und erstelle den Ordner, falls nicht vorhanden
folder_path = os.path.dirname(file_path)

if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")
    
#%% plot energy
print('#### plotting energy over selected dates ####\n')
import pandas as pd
import plotly.graph_objects as go
import os


# Erstellen eines neuen DataFrames zur Zwischenspeicherung der täglichen Energiesummen
sums_dict = {
    'Datum': [],
    'zero_degree': [],
    'flip_flop_five_degree': [],
    'flip_flop_ten_degree': [],
    'tracking_five_degree': [],
    'tracking_ten_degree': [],
    'tracking_twenty_degree': [],
    'backtracking' : [],
    'max_energy': []
}

# Sortiere das yield_tracking Dictionary und iteriere nur bis zum end_date
sorted_dates = sorted(yield_tracking.keys())
for date in sorted_dates:
    date_obj = pd.to_datetime(date)
    if date_obj < start_date:
        continue
    if date_obj > end_date:
        break  # Beendet die Schleife, wenn das Enddatum überschritten wird
    
    sums_dict['Datum'].append(date_obj)
    for condition in [key for key in sums_dict if key != 'Datum']:
        df = yield_tracking[date].get(condition, pd.DataFrame())
        sums_dict[condition].append(df['energy'].sum() if 'energy' in df.columns else 0)

# Konvertiere das Dictionary in einen DataFrame
daily_energy_sums = pd.DataFrame(sums_dict)
daily_energy_sums.set_index('Datum', inplace=True)

# Erstelle den Plot
fig = go.Figure()
for condition in daily_energy_sums.columns:
    if condition != 'Datum':  # 'Datum' ist kein DataFrame
        fig.add_trace(go.Scatter(
            x=daily_energy_sums.index,
            y=daily_energy_sums[condition],
            mode='lines+markers',
            name=condition
        ))

# Aktualisiere das Layout
fig.update_layout(
    title=f"Tägliche Energy aller Trackingalgorithmen ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})",
    xaxis_title="Datum",
    yaxis_title="Energie [Wh]",
    xaxis=dict(
        range=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    )
)

# Speicherort für den Plot
file_name = f"energy_algorithms_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Energies/{file_name}"

# Überprüfe, ob der Ordner existiert, und erstelle ihn, falls nicht
folder_path = os.path.dirname(file_path)
if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")
#%% plot energy clearsky
print('#### plotting clearsky energy over selected dates ####\n')
import pandas as pd
import plotly.graph_objects as go
import os


# Erstellen eines neuen DataFrames zur Zwischenspeicherung der täglichen Energiesummen
sums_dict = {
    'Datum': [],
    'zero_degree': [],
    'flip_flop_five_degree': [],
    'flip_flop_ten_degree': [],
    'tracking_five_degree': [],
    'tracking_ten_degree': [],
    'tracking_twenty_degree': [],
    'backtracking' : [],
    'max_energy': []
}

# Sortiere das yield_tracking Dictionary und iteriere nur bis zum end_date
sorted_dates = sorted(yield_tracking_clearsky.keys())
for date in sorted_dates:
    date_obj = pd.to_datetime(date)
    if date_obj < start_date:
        continue
    if date_obj > end_date:
        break  # Beendet die Schleife, wenn das Enddatum überschritten wird
    
    sums_dict['Datum'].append(date_obj)
    for condition in [key for key in sums_dict if key != 'Datum']:
        df = yield_tracking[date].get(condition, pd.DataFrame())
        sums_dict[condition].append(df['energy'].sum() if 'energy' in df.columns else 0)

# Konvertiere das Dictionary in einen DataFrame
daily_energy_sums = pd.DataFrame(sums_dict)
daily_energy_sums.set_index('Datum', inplace=True)

# Erstelle den Plot
fig = go.Figure()
for condition in daily_energy_sums.columns:
    if condition != 'Datum':  # 'Datum' ist kein DataFrame
        fig.add_trace(go.Scatter(
            x=daily_energy_sums.index,
            y=daily_energy_sums[condition],
            mode='lines+markers',
            name=condition
        ))

# Aktualisiere das Layout
fig.update_layout(
    title=f"Tägliche Energie clearsky aller Trackingalgorithmen ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})",
    xaxis_title="Datum",
    yaxis_title="Energie [Wh]",
    xaxis=dict(
        range=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    )
)

# Speicherort für den Plot
file_name = f"energy_algorithms_clearsky_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Energies/{file_name}"

# Überprüfe, ob der Ordner existiert, und erstelle ihn, falls nicht
folder_path = os.path.dirname(file_path)
if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")
#%% plot energy cloudy
print('#### plotting cloudy energy over selected dates ####\n')
import pandas as pd
import plotly.graph_objects as go
import os


# Erstellen eines neuen DataFrames zur Zwischenspeicherung der täglichen Energiesummen
sums_dict = {
    'Datum': [],
    'zero_degree': [],
    'flip_flop_five_degree': [],
    'flip_flop_ten_degree': [],
    'tracking_five_degree': [],
    'tracking_ten_degree': [],
    'tracking_twenty_degree': [],
    'backtracking' : [],
    'max_energy': []
}

# Sortiere das yield_tracking Dictionary und iteriere nur bis zum end_date
sorted_dates = sorted(yield_tracking_cloudy.keys())
for date in sorted_dates:
    date_obj = pd.to_datetime(date)
    if date_obj < start_date:
        continue
    if date_obj > end_date:
        break  # Beendet die Schleife, wenn das Enddatum überschritten wird
    
    sums_dict['Datum'].append(date_obj)
    for condition in [key for key in sums_dict if key != 'Datum']:
        df = yield_tracking[date].get(condition, pd.DataFrame())
        sums_dict[condition].append(df['energy'].sum() if 'energy' in df.columns else 0)

# Konvertiere das Dictionary in einen DataFrame
daily_energy_sums = pd.DataFrame(sums_dict)
daily_energy_sums.set_index('Datum', inplace=True)

# Erstelle den Plot
fig = go.Figure()
for condition in daily_energy_sums.columns:
    if condition != 'Datum':  # 'Datum' ist kein DataFrame
        fig.add_trace(go.Scatter(
            x=daily_energy_sums.index,
            y=daily_energy_sums[condition],
            mode='lines+markers',
            name=condition
        ))

# Aktualisiere das Layout
fig.update_layout(
    title=f"Tägliche Energie cloudy aller Trackingalgorithmen ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})",
    xaxis_title="Datum",
    yaxis_title="Energie [Wh]",
    xaxis=dict(
        range=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    )
)

# Speicherort für den Plot
file_name = f"energy_algorithms_cloudy_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
file_path = f"C:/Users/Public/BA/BA-HSAT-Auswertungstool/Results/Energies/{file_name}"

# Überprüfe, ob der Ordner existiert, und erstelle ihn, falls nicht
folder_path = os.path.dirname(file_path)
if not os.path.exists(file_path):
    # Wenn die Datei nicht existiert, speichere sie
    fig.write_html(file_path)
else:
    # Wenn die Datei bereits existiert, gebe eine Meldung aus
    print("Die Datei existiert bereits.\n")

