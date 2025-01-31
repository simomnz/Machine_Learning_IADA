import pandas as pd
from menu import menu


    #'''
    # Potremmo anche fare come scritto nella documentazione del dataset, 
    # ma poi nel yml dobbiamo mettere anche
    # pip install ucimlrepo 
    
    # per poi fare
    # from ucimlrepo import fetch_ucirepo 
    #
    # fetch dataset 
    # estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)
    #'''

def main():

    # Carica i dati iniziali
    try:
        # Tenta di leggere il file CSV contenente il dataset
        df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', encoding='UTF-8')
    except FileNotFoundError:
        # Gestisce l'errore nel caso in cui il file CSV non venga trovato
        print("File CSV non trovato. Assicurati che 'ObesityDataSet_raw_and_data_sinthetic.csv' sia nella directory corrente.")
        return
    except Exception as e:
        # Gestisce qualsiasi altro tipo di errore durante il caricamento del file CSV
        print(f"Errore nel caricamento del file CSV: {e}")
        return
    
    # Avvia il menu GUI passando il dataframe caricato
    menu(df)

if __name__ == "__main__":
    main()  # lascia la riga vuota alla fine del file
