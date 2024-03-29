import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import StringVar
from tkinter import *
from AI_Models import predict_sentiment_tb, predict_sentiment_ml, accuracy_tb, accuracy_ml, stat_report

def display_accuracy_tb():
    # compute accuracy score for Textblob using integers
    return accuracy_tb()

def display_accuracy_ml():
    # compute accuracy score for Machine Learning Model
    return accuracy_ml()

def display_results():
    # Show confusion matrix, precision, recall, and F1-score
    return stat_report()

# Create the GUI window
window = tk.Tk()
window.title("Sentiment Analysis App")
window.resizable(False, False) 

# Define the functions for TextBlob and Machine Learning predictions
def analyse():
    text = input_text.get("1.0", "end-1c")
    
    if text == "":
        messagebox.showerror("Error", "Please Enter Some Text.")
        return
    elif text.lower() == "quit":
        window.destroy()
    
    sentiment_tb = predict_sentiment_tb(text)
    tb_result.set(sentiment_tb)

    sentiment_ml = predict_sentiment_ml(text)
    ml_result.set(sentiment_ml)

# Define the function to clear the input and output text fields
def clear_text():
    input_text.delete("1.0", "end")
    tb_result.set("")
    ml_result.set("")

# Define the GUI elements
Stats_label = tk.Label(window, text="Statistics :")
Stats_label.pack()

Stats_text = scrolledtext.ScrolledText(window, width=70, height=15)
Stats_text.pack(pady=10)

input_label = tk.Label(window, text="Enter your Text :")
input_label.pack(pady=10)

input_text = scrolledtext.ScrolledText(window, width=70, height=5)
input_text.pack()

analyse_button = tk.Button(window, text="Analyse", command=analyse, height=2, width=10)
analyse_button.pack(pady=20)

tb_result_label = Label(window, text="TextBlob Predicted Sentiment :")
tb_result_label.pack()

tb_result = StringVar(window,"")
tb_result_text = Label(window, textvariable=tb_result)
tb_result_text.pack(pady=10)

ml_result_label = Label(window, text="ML Model Predicted Sentiment :")
ml_result_label.pack(pady=10)

ml_result = StringVar(window,"")
ml_result_text = Label(window, textvariable=ml_result)
ml_result_text.pack()

clear_button = tk.Button(window, text="Clear", command=clear_text, height=2, width=10)
clear_button.pack(side=LEFT, padx=140, pady=20)

quit_button = tk.Button(window, text="Quit", command=window.destroy, height=2, width=10)
quit_button.pack(side=LEFT, padx=140, pady=20)

Stats_text.insert("end", display_accuracy_tb()+"\n")
Stats_text.insert("end", display_accuracy_ml()+"\n")
Stats_text.insert("end", display_results())

#Run the GUI window
window.mainloop()
