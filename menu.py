import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from dataAnalysis import dataAnalysis, show_correlation_matrix
from classifiers import classifiers
from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_preprocessing_combinations

warnings.filterwarnings("ignore", category=ConvergenceWarning)



#   __  __ ______ _   _ _    _ 
#  |  \/  |  ____| \ | | |  | |
#  | \  / | |__  |  \| | |  | |
#  | |\/| |  __| | . ` | |  | |
#  | |  | | |____| |\  | |__| |
#  |_|  |_|______|_| \_|\____/ 
                             
                           

def menu(df: pd.DataFrame):
    class MLApp:
        def __init__(self, master, df):
            self.master = master
            self.master.title("Menu di Machine Learning")
            self.master.geometry("800x600")
            self.df = df

            # Pulsante per l'analisi dei dati
            self.analysis_button = tk.Button(master, text="Analisi Dati", command=self.perform_data_analysis, width=25, height=2)
            self.analysis_button.pack(pady=10)

            # Pulsante per addestrare un classificatore
            self.classifier_button = tk.Button(master, text="Addestra Classificatore", command=self.open_classifier_window, width=25, height=2)
            self.classifier_button.pack(pady=10)

            # Pulsante per tecniche di preprocessing
            self.preprocessing_button = tk.Button(master, text="Tecniche Preprocessing", command=self.open_preprocessing_window, width=25, height=2)
            self.preprocessing_button.pack(pady=10)

            # Pulsante per uscire dall'applicazione
            self.exit_button = tk.Button(master, text="Esci", command=master.quit, width=25, height=2)
            self.exit_button.pack(pady=10)

            # Area di testo per loggare i messaggi
            self.log_text = tk.Text(master, height=10, width=80)
            self.log_text.pack(pady=10)

        def log(self, message):
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)

        def perform_data_analysis(self):
            if self.df is None:
                messagebox.showwarning("Avviso", "Carica prima i dati.")
                self.log("Tentativo di analisi senza dati caricati.")
                return

            try:
                analysis_window = tk.Toplevel(self.master)
                analysis_window.title("Analisi dei Dati")
                analysis_window.geometry("1000x800")

                stats_frame = tk.Frame(analysis_window)
                stats_frame.pack(pady=10, fill=tk.BOTH, expand=True)

                tk.Label(stats_frame, text="Statistiche Descrittive per Colonna", font=("Helvetica", 14)).pack()

                stats_tree = ttk.Treeview(stats_frame)
                stats_tree["show"] = "headings"
                stats_tree.pack(fill=tk.BOTH, expand=True)

                stats_tree["columns"] = ("Column", "Mean", "Median", "Std", "Min", "Max")
                stats_tree.heading("Column", text="Colonna")
                stats_tree.heading("Mean", text="Media")
                stats_tree.heading("Median", text="Mediana")
                stats_tree.heading("Std", text="Deviazione Standard")
                stats_tree.heading("Min", text="Minimo")
                stats_tree.heading("Max", text="Massimo")

                stats_tree.column("Column", width=150)
                stats_tree.column("Mean", width=100)
                stats_tree.column("Median", width=100)
                stats_tree.column("Std", width=150)
                stats_tree.column("Min", width=100)
                stats_tree.column("Max", width=100)

                # Seleziona solo le colonne numeriche
                numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
                stats = self.df[numeric_cols].describe().loc[['mean', '50%', 'std', 'min', 'max']].rename(index={'50%': 'median'})

                for col in numeric_cols:
                    stats_tree.insert("", tk.END, values=(
                        col,
                        f"{stats.at['mean', col]:.2f}",
                        f"{stats.at['median', col]:.2f}",
                        f"{stats.at['std', col]:.2f}",
                        f"{stats.at['min', col]:.2f}",
                        f"{stats.at['max', col]:.2f}"
                    ))

                outliers_frame = tk.Frame(analysis_window)
                outliers_frame.pack(pady=10, fill=tk.BOTH, expand=True)

                tk.Label(outliers_frame, text="Outliers per Colonna", font=("Helvetica", 14)).pack()

                outliers_tree = ttk.Treeview(outliers_frame)
                outliers_tree["show"] = "headings"
                outliers_tree.pack(fill=tk.BOTH, expand=True)

                outliers_tree["columns"] = ("Column", "Numero di Outliers")
                outliers_tree.heading("Column", text="Colonna")
                outliers_tree.heading("Numero di Outliers", text="Numero di Outliers")
                outliers_tree.column("Column", width=150)
                outliers_tree.column("Numero di Outliers", width=200)

                # Calcola il numero di outliers per ogni colonna numerica
                outliers = self.compute_outliers(self.df)
                for col, count in outliers.items():
                    outliers_tree.insert("", tk.END, values=(col, count))

                # Pulsante per mostrare la matrice di correlazione
                corr_button = tk.Button(analysis_window, text="Mostra Matrice di Correlazione", command=lambda: show_correlation_matrix(self.df))
                corr_button.pack(pady=10)

                self.log("Analisi dei dati completata.")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nell'analisi dei dati: {e}")
                self.log(f"Errore nell'analisi dei dati: {e}")

        def compute_outliers(self, df: pd.DataFrame) -> dict:
            variabiliNumeriche = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
            outliers = {}
            for col in variabiliNumeriche:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outliers[col] = count
            return outliers

        def open_classifier_window(self):
            if self.df is None:
                messagebox.showwarning("Avviso", "Carica prima i dati.")
                self.log("Tentativo di addestramento senza dati caricati.")
                return

            classifier_window = tk.Toplevel(self.master)
            classifier_window.title("Addestra Classificatore")
            classifier_window.geometry("400x300")

            tk.Label(classifier_window, text="Scegli il Classificatore:").pack(pady=10)

            classifier_var = tk.StringVar()
            classifier_choices = ["svm", "tree", "ann", "knn", "voting"]
            classifier_dropdown = ttk.Combobox(classifier_window, textvariable=classifier_var, values=classifier_choices, state="readonly")
            classifier_dropdown.pack(pady=5)
            classifier_dropdown.current(0)

            train_button = tk.Button(classifier_window, text="Addestra", command=lambda: self.train_classifier(classifier_var.get(), classifier_window))
            train_button.pack(pady=20)

        def train_classifier(self, classifier_type, window):
            try:
                accuracy = classifiers(self.df, classifier_type)
                messagebox.showinfo("Risultato", f"Accuratezza = {accuracy:.2f}")
                self.log(f"Addestrato {classifier_type} con accuratezza = {accuracy:.2f}")
                window.destroy()
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nell'addestramento del classificatore: {e}")
                self.log(f"Errore nell'addestramento del classificatore: {e}")

        def open_preprocessing_window(self):
            if self.df is None:
                messagebox.showwarning("Avviso", "Carica prima i dati.")
                self.log("Tentativo di confronto pre-processing senza dati caricati.")
                return

            try:
                X = self.df.iloc[:, :-1]
                X = pd.get_dummies(X)
                y = self.df.iloc[:, -1]

                clf = LogisticRegression(max_iter=1000, random_state=42)

                results = evaluate_preprocessing_combinations(X, y, clf)

                prep_window = tk.Toplevel(self.master)
                prep_window.title("Confronto Tecniche Preprocessing")
                prep_window.geometry("500x400")

                tk.Label(prep_window, text="Risultati 5-fold Cross-Validation (Accuracy)", font=("Helvetica", 14)).pack(pady=10)

                result_tree = ttk.Treeview(prep_window)
                result_tree["show"] = "headings"
                result_tree.pack(fill=tk.BOTH, expand=True)

                result_tree["columns"] = ("Tecnica", "Accuracy")
                result_tree.heading("Tecnica", text="Tecnica")
                result_tree.heading("Accuracy", text="Accuracy")
                result_tree.column("Tecnica", width=200)
                result_tree.column("Accuracy", width=100)

                for tech, acc in results.items():
                    result_tree.insert("", tk.END, values=(tech, f"{acc:.4f}"))

                best_tech = max(results, key=results.get)
                best_acc = results[best_tech]

                best_label = tk.Label(prep_window, text=f"Miglior tecnica: {best_tech} (Accuracy: {best_acc:.4f})", font=("Helvetica", 12, "bold"))
                best_label.pack(pady=10)

                self.log("Confronto delle tecniche di pre-processing completato.")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel confronto pre-processing: {e}")
                self.log(f"Errore nel confronto pre-processing: {e}")

    root = tk.Tk()
    app = MLApp(root, df)
    root.mainloop()
