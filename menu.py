# import tkinter as tk
# from tkinter import PhotoImage
# import pandas as pd
# from PIL import Image, ImageTk  
# from dataAnalysis import dataAnalysis
# from classifiers import classifiers
# from regressors import regressors

# df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv', encoding='UTF-8')

def menu():
    print("Premi 1 per performare un'analisi dei dati")
    print("Premi 2 per usare un classificatore")

    '''root = tk.Tk()
    root.title("Sborra")
    root.geometry("1000x800")
    root.config(bg="#E0E0E0")

    left_frame = tk.Frame(root, bg="#E0E0E0")
    left_frame.pack(side="left", fill="y", padx=20, pady=20)

    button_width = 20
    tk.Button(
        left_frame, 
        text="Analisi dei dati", 
        command=lambda: dataAnalysis(df),
        font=("Tahoma", 12),
        bg="#D4D0C8",
        fg="black",
        activebackground="#E4E4E4",
        activeforeground="black",
        relief="raised",
        bd=2,
        width=button_width
    ).pack(pady=10)

    tk.Button(
        left_frame, 
        text="Regressore", 
        command= regressors(df),
        font=("Tahoma", 12),
        bg="#D4D0C8",
        fg="black",
        activebackground="#E4E4E4",
        activeforeground="black",
        relief="raised",
        bd=2,
        width=button_width
    ).pack(pady=10)

    tk.Button(
        left_frame, 
        text="Classificatore", 
        command= classifiers(df),
        font=("Tahoma", 12),
        bg="#D4D0C8",
        fg="black",
        activebackground="#E4E4E4",
        activeforeground="black",
        relief="raised",
        bd=2,
        width=button_width
    ).pack(pady=10)

    right_frame = tk.Frame(root, bg="#E0E0E0")
    right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    image = Image.open("logo.jpg")
    image = image.resize((600, 600), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image)
    img_label = tk.Label(right_frame, image=image_tk, bg="#E0E0E0")
    img_label.image = image_tk
    img_label.pack(expand=True)

    root.mainloop()'''
