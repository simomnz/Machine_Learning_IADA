import pandas as pd
from menu import menu
from dataAnalysis import dataAnalysis
from classifiers import classifiers


def main():
    '''
    # potremmo anche fare come scritto nella documentazione del dataset, 
    # ma poi nel yml dobbiamo mettere anche
    pip install ucimlrepo 
    
    # per poi fare
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)
    
    '''
    
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', encoding='UTF-8')
    # print(type(df))


    while True:
        menu()
        
        choice = input()

        match choice:
            case "1":
                # analisi dei dati
                dataAnalysis(df)
            case "2":
                # classificatori
                classifiers(df)
            case _:
                break
        
        
if __name__ == "__main__":
    main() # lascia la riga vuota alla fine del file
