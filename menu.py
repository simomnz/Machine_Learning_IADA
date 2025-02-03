import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from dataAnalysis import dataAnalysis, show_correlation_matrix
from classifiers import classifiers
from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_preprocessing_combinations  # Funzione in italiano
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Ignora i warning nel preprocessing

def menu(dati: pd.DataFrame):
    class MLApp:
        def __init__(self, master, dati):
            # Imposta la finestra principale
            self.master = master
            self.master.title("Menu di Machine Learning")
            self.master.geometry("800x600")
            self.dati = dati

            # Pulsante per l'analisi dei dati
            self.analisiButton = tk.Button(master, text="Analisi Dati", command=self.eseguiAnalisiDati, width=25, height=2)
            self.analisiButton.pack(pady=10)

            # Pulsante per addestrare un classificatore
            self.classificatoreButton = tk.Button(master, text="Addestra Classificatore", command=self.apriFinestraClassificatore, width=25, height=2)
            self.classificatoreButton.pack(pady=10)

            # Pulsante per confrontare tecniche di preprocessing
            self.preprocessingButton = tk.Button(master, text="Tecniche Preprocessing", command=self.apriFinestraPreprocessing, width=25, height=2)
            self.preprocessingButton.pack(pady=10)

            # Pulsante per uscire dall'applicazione
            self.uscitaButton = tk.Button(master, text="Esci", command=master.quit, width=25, height=2)
            self.uscitaButton.pack(pady=10)

            # Area di testo per loggare i messaggi
            self.logText = tk.Text(master, height=10, width=80)
            self.logText.pack(pady=10)

        def log(self, messaggio):
            # Scrive un messaggio nell'area di log
            self.logText.insert(tk.END, messaggio + "\n")
            self.logText.see(tk.END)

        def eseguiAnalisiDati(self):
            # Se non ci sono dati, mostra un messaggio e termina
            if self.dati is None:
                messagebox.showwarning("Avviso", "Carica prima i dati.")
                self.log("Nessun dato caricato.")
                return

            # Crea una nuova finestra per l'analisi dei dati
            finestraAnalisi = tk.Toplevel(self.master)
            finestraAnalisi.title("Analisi dei Dati")
            finestraAnalisi.geometry("1000x800")

            # Mostra le statistiche descrittive in una tabella
            frameStatistiche = tk.Frame(finestraAnalisi)
            frameStatistiche.pack(pady=10, fill=tk.BOTH, expand=True)
            tk.Label(frameStatistiche, text="Statistiche Descrittive per Colonna", font=("Helvetica", 14)).pack()
            tabellaStatistiche = ttk.Treeview(frameStatistiche)
            tabellaStatistiche["show"] = "headings"
            tabellaStatistiche.pack(fill=tk.BOTH, expand=True)
            tabellaStatistiche["columns"] = ("Colonna", "Media", "Mediana", "Deviazione Standard", "Minimo", "Massimo")
            # Imposta le intestazioni e le dimensioni delle colonne
            tabellaStatistiche.heading("Colonna", text="Colonna")
            tabellaStatistiche.heading("Media", text="Media")
            tabellaStatistiche.heading("Mediana", text="Mediana")
            tabellaStatistiche.heading("Deviazione Standard", text="Deviazione Standard")
            tabellaStatistiche.heading("Minimo", text="Minimo")
            tabellaStatistiche.heading("Massimo", text="Massimo")
            tabellaStatistiche.column("Colonna", width=150)
            tabellaStatistiche.column("Media", width=100)
            tabellaStatistiche.column("Mediana", width=100)
            tabellaStatistiche.column("Deviazione Standard", width=150)
            tabellaStatistiche.column("Minimo", width=100)
            tabellaStatistiche.column("Massimo", width=100)

            colonneNumeriche = self.dati.select_dtypes(include=['float64', 'int64']).columns
            stats = self.dati[colonneNumeriche].describe().loc[['mean', '50%', 'std', 'min', 'max']]\
                      .rename(index={'50%': 'median'})
            for col in colonneNumeriche:
                tabellaStatistiche.insert("", tk.END, values=(
                    col,
                    f"{stats.at['mean', col]:.2f}",
                    f"{stats.at['median', col]:.2f}",
                    f"{stats.at['std', col]:.2f}",
                    f"{stats.at['min', col]:.2f}",
                    f"{stats.at['max', col]:.2f}"
                ))

            # Mostra il numero di outliers per alcune colonne
            frameOutliers = tk.Frame(finestraAnalisi)
            frameOutliers.pack(pady=10, fill=tk.BOTH, expand=True)
            tk.Label(frameOutliers, text="Outliers per Colonna", font=("Helvetica", 14)).pack()
            tabellaOutliers = ttk.Treeview(frameOutliers)
            tabellaOutliers["show"] = "headings"
            tabellaOutliers.pack(fill=tk.BOTH, expand=True)
            tabellaOutliers["columns"] = ("Colonna", "Numero di Outliers")
            tabellaOutliers.heading("Colonna", text="Colonna")
            tabellaOutliers.heading("Numero di Outliers", text="Numero di Outliers")
            tabellaOutliers.column("Colonna", width=150)
            tabellaOutliers.column("Numero di Outliers", width=200)
            outliers = self.calcolaOutliers(self.dati)
            for col, num in outliers.items():
                tabellaOutliers.insert("", tk.END, values=(col, num))

            # Pulsante per mostrare la matrice di correlazione
            pulsanteCorrelazione = tk.Button(finestraAnalisi, text="Mostra Matrice di Correlazione",  command=lambda: show_correlation_matrix(self.dati))
            pulsanteCorrelazione.pack(pady=10)


            self.log("Analisi dei dati completata.")

        def calcolaOutliers(self, df: pd.DataFrame) -> dict:
            # Lista delle colonne numeriche da analizzare per gli outliers
            colonneNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
            outliers = {}
            for col in colonneNumeriche:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    num = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                    outliers[col] = num
            return outliers

        def apriFinestraClassificatore(self):
            # Se i dati non sono caricati, mostra un messaggio
            if self.dati is None:
                messagebox.showwarning("Avviso", "Carica prima i dati.")
                self.log("Nessun dato caricato per l'addestramento.")
                return

            finestraClassificatore = tk.Toplevel(self.master)
            finestraClassificatore.title("Addestra Classificatore")
            finestraClassificatore.geometry("400x400")

            tk.Label(finestraClassificatore, text="Scegli il Classificatore:").pack(pady=10)
            variabileClassificatore = tk.StringVar()
            scelteClassificatore = ["svm", "tree", "ann", "knn", "voting"]
            dropdownClassificatore = ttk.Combobox(finestraClassificatore, textvariable=variabileClassificatore, values=scelteClassificatore, state="readonly")
            dropdownClassificatore.pack(pady=5)
            dropdownClassificatore.current(0)

            # CheckBox per "Usa Tuning"
            varTuning = tk.BooleanVar(value=True)
            checkTuning = tk.Checkbutton(finestraClassificatore, text="Usa Tuning", variable=varTuning)
            checkTuning.pack(pady=5)

            # CheckBox per "Usa Pipeline"
            varPipeline = tk.BooleanVar(value=False)
            checkPipeline = tk.Checkbutton(finestraClassificatore, text="Usa Pipeline", variable=varPipeline)
            checkPipeline.pack(pady=5)

            # Opzioni aggiuntive per KNN/Voting: scelta della metrica di distanza e valore di k
            tk.Label(finestraClassificatore, text="Metrica di Distanza:").pack(pady=5)
            variabileMetrica = tk.StringVar()
            scelteMetrica = ["manhattan", "euclidean", "chebyshev"]
            dropdownMetrica = ttk.Combobox(finestraClassificatore, textvariable=variabileMetrica, values=scelteMetrica, state="readonly")
            dropdownMetrica.pack(pady=5)
            dropdownMetrica.current(0)

            tk.Label(finestraClassificatore, text="Valore di k:").pack(pady=5)
            variabileK = tk.StringVar(value="0")
            entryK = tk.Entry(finestraClassificatore, textvariable=variabileK)
            entryK.pack(pady=5)

            pulsanteAddestra = tk.Button(finestraClassificatore, text="Addestra",
                command=lambda: self.addestraClassificatore(
                    variabileClassificatore.get(),
                    varTuning.get(),
                    varPipeline.get(),
                    variabileMetrica.get(),
                    variabileK.get(),
                    finestraClassificatore
                ))
            pulsanteAddestra.pack(pady=20)

        def addestraClassificatore(self, tipoClassificatore, usaTuning, usaPipeline, metrica, kVal, finestra):
            # Converte k in intero
            k_int = int(kVal)
            # Chiama la funzione classifiers passando anche i parametri 'distanza' e 'kValue'
            accuracy = classifiers(self.dati, tipoClassificatore, tuning=usaTuning, usaPipeline=usaPipeline,
                                   distanza=metrica, kValue=k_int)
            # Se il classificatore è KNN o Voting, includi nel messaggio il valore di k e la metrica
            if tipoClassificatore in ["knn", "voting"]:
                messaggio = (f"Addestrato {tipoClassificatore} con accuratezza = {accuracy:.2f} " f" \n(metrica = {metrica})")
            else:
                messaggio = f"Addestrato {tipoClassificatore} con accuratezza = {accuracy:.2f}"
            messagebox.showinfo("Risultato", messaggio)
            self.log(messaggio)
            finestra.destroy()

        def apriFinestraPreprocessing(self):
            if self.dati is None:
                messagebox.showwarning("Avviso", "Carica prima i dati.")
                self.log("Nessun dato caricato per il preprocessing.")
                return

            # Presume che l'ultima colonna sia la variabile target
            X = self.dati.iloc[:, :-1]
            X = pd.get_dummies(X)
            y = self.dati.iloc[:, -1]

            clf = LogisticRegression(max_iter=1000, random_state=42)
            risultati = evaluate_preprocessing_combinations(X, y, clf)

            finestraPreprocessing = tk.Toplevel(self.master)
            finestraPreprocessing.title("Confronto Tecniche Preprocessing")
            finestraPreprocessing.geometry("500x400")

            tk.Label(finestraPreprocessing, text="Risultati 5-fold Cross-Validation (Accuracy)", font=("Helvetica", 14)).pack(pady=10)

            tabellaRisultati = ttk.Treeview(finestraPreprocessing)
            tabellaRisultati["show"] = "headings"
            tabellaRisultati.pack(fill=tk.BOTH, expand=True)
            tabellaRisultati["columns"] = ("Tecnica", "Accuracy")
            tabellaRisultati.heading("Tecnica", text="Tecnica")
            tabellaRisultati.heading("Accuracy", text="Accuracy")
            tabellaRisultati.column("Tecnica", width=200)
            tabellaRisultati.column("Accuracy", width=100)

            for tecnica, acc in risultati.items():
                tabellaRisultati.insert("", tk.END, values=(tecnica, f"{acc:.4f}"))

            miglioreTecnica = max(risultati, key=risultati.get)
            miglioreAccuracy = risultati[miglioreTecnica]
            etichettaMigliore = tk.Label(finestraPreprocessing, text=f"Miglior tecnica: {miglioreTecnica} (Accuracy: {miglioreAccuracy:.4f})", font=("Helvetica", 12, "bold"))
            etichettaMigliore.pack(pady=10)

            self.log("Confronto delle tecniche di pre-processing completato.")

    root = tk.Tk()
    MLApp(root, dati)
    root.mainloop()

