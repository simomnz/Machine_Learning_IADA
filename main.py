import pandas as pd
from menu import menu
from dataAnalysis import dataAnalysis
from classifiers import classifiers


def main():
    '''
    # Potremmo anche fare come scritto nella documentazione del dataset, 
    # ma poi nel yml dobbiamo mettere anche
    # pip install ucimlrepo 
    
    # per poi fare
    # from ucimlrepo import fetch_ucirepo 
    #
    # fetch dataset 
    # estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)
    '''
    
    # Carica i dati iniziali
    try:
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', encoding='UTF-8')
    except FileNotFoundError:
        print("File CSV non trovato. Assicurati che 'ObesityDataSet_raw_and_data_sinthetic.csv' sia nella directory corrente.")
        return
    except Exception as e:
        print(f"Errore nel caricamento del file CSV: {e}")
        return
    
    # Avvia il menu GUI
    menu(df)

if __name__ == "__main__":
    main()
