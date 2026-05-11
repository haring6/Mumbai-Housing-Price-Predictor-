import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import ttk, messagebox

FILE_PATH = r"C:\Users\haringogri\OneDrive\Desktop\firstml\Mumbai House Prices.csv"

df = pd.read_csv(FILE_PATH)
df['price_lakhs'] = df.apply(lambda x: x['price'] * 100 if x['price_unit'] == 'Cr' else x['price'], axis=1)
df = df[['bhk', 'type', 'area', 'region', 'status', 'age', 'price_lakhs']].dropna()
df = df[(df['price_lakhs'] > 5) & (df['price_lakhs'] < 100000)]
df = df[(df['area'] > 100) & (df['area'] < 10000)]
df = df[df['bhk'].between(1, 10)]

le_type = LabelEncoder()
le_region = LabelEncoder()
le_status = LabelEncoder()
le_age = LabelEncoder()

df['type_enc'] = le_type.fit_transform(df['type'])
df['region_enc'] = le_region.fit_transform(df['region'])
df['status_enc'] = le_status.fit_transform(df['status'])
df['age_enc'] = le_age.fit_transform(df['age'])

X = df[['bhk', 'area', 'type_enc', 'region_enc', 'status_enc', 'age_enc']]
y = df['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

regions = sorted(df['region'].unique().tolist())
types = sorted(df['type'].unique().tolist())
statuses = sorted(df['status'].unique().tolist())
ages = sorted(df['age'].unique().tolist())

root = tk.Tk()
root.title("Mumbai House Price Predictor")
root.geometry("520x680")
root.resizable(False, False)
root.configure(bg="#f5f5f5")

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_LABEL = ("Segoe UI", 10)
FONT_RESULT = ("Segoe UI", 14, "bold")

tk.Label(root, text="🏠 Mumbai House Price Predictor", font=FONT_TITLE, bg="#f5f5f5", fg="#1a1a1a").pack(pady=(20, 4))
tk.Label(root, text=f"Trained on {len(df):,} real listings  |  R² = {r2:.2f}  |  Avg error ±₹{mae:.0f}L", font=("Segoe UI", 8), bg="#f5f5f5", fg="#888").pack(pady=(0, 16))

frame = tk.Frame(root, bg="#f5f5f5", padx=30)
frame.pack(fill="both")

def labeled_row(label_text):
    tk.Label(frame, text=label_text, font=FONT_LABEL, bg="#f5f5f5", anchor="w").pack(fill="x", pady=(10, 2))

labeled_row("BHK (Number of bedrooms)")
bhk_var = tk.IntVar(value=2)
bhk_frame = tk.Frame(frame, bg="#f5f5f5")
bhk_frame.pack(fill="x")
for val in [1, 2, 3, 4, 5]:
    tk.Radiobutton(bhk_frame, text=str(val), variable=bhk_var, value=val, font=FONT_LABEL, bg="#f5f5f5").pack(side="left", padx=6)

labeled_row("Area (sq ft)")
area_var = tk.StringVar(value="800")
tk.Entry(frame, textvariable=area_var, font=FONT_LABEL, width=20).pack(anchor="w")

labeled_row("Property Type")
type_var = tk.StringVar(value=types[0])
ttk.Combobox(frame, textvariable=type_var, values=types, state="readonly", font=FONT_LABEL, width=30).pack(anchor="w")

labeled_row("Region (type to search)")
region_var = tk.StringVar()
tk.Entry(frame, textvariable=region_var, font=FONT_LABEL, width=34).pack(anchor="w")

region_listbox_frame = tk.Frame(frame)
region_listbox_frame.pack(anchor="w")
region_listbox = tk.Listbox(region_listbox_frame, height=4, width=34, font=("Segoe UI", 9), selectmode="single")
region_scrollbar = tk.Scrollbar(region_listbox_frame, orient="vertical", command=region_listbox.yview)
region_listbox.config(yscrollcommand=region_scrollbar.set)
region_listbox.pack(side="left")
region_scrollbar.pack(side="right", fill="y")

def update_region_list(*args):
    search = region_var.get().lower()
    region_listbox.delete(0, tk.END)
    for r in regions:
        if search in r.lower():
            region_listbox.insert(tk.END, r)

def select_region(event):
    sel = region_listbox.curselection()
    if sel:
        region_var.set(region_listbox.get(sel[0]))

region_var.trace("w", update_region_list)
region_listbox.bind("<<ListboxSelect>>", select_region)
update_region_list()

labeled_row("Status")
status_var = tk.StringVar(value=statuses[0])
ttk.Combobox(frame, textvariable=status_var, values=statuses, state="readonly", font=FONT_LABEL, width=30).pack(anchor="w")

labeled_row("Property Age")
age_var = tk.StringVar(value=ages[0])
ttk.Combobox(frame, textvariable=age_var, values=ages, state="readonly", font=FONT_LABEL, width=30).pack(anchor="w")

result_var = tk.StringVar(value="")
tk.Label(root, textvariable=result_var, font=FONT_RESULT, bg="#f5f5f5", fg="#1a6e2e").pack(pady=(18, 0))

sub_var = tk.StringVar(value="")
tk.Label(root, textvariable=sub_var, font=("Segoe UI", 9), bg="#f5f5f5", fg="#888").pack()

def predict():
    try:
        area = float(area_var.get())
        bhk = bhk_var.get()
        ptype = type_var.get()
        region = region_var.get()
        status = status_var.get()
        age = age_var.get()
        if region not in regions:
            messagebox.showwarning("Region not found", "Please select a valid region from the list.")
            return
        type_enc = le_type.transform([ptype])[0]
        region_enc = le_region.transform([region])[0]
        status_enc = le_status.transform([status])[0]
        age_enc = le_age.transform([age])[0]
        inp = np.array([[bhk, area, type_enc, region_enc, status_enc, age_enc]])
        price = model.predict(inp)[0]
        display = f"₹ {price/100:.2f} Crore" if price >= 100 else f"₹ {price:.1f} Lakhs"
        result_var.set(f"Estimated Price: {display}")
        sub_var.set(f"Range: ₹{max(0,price-mae):.0f}L – ₹{price+mae:.0f}L  |  {bhk}BHK {ptype} in {region}")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number for area.")

tk.Button(root, text="  Predict Price  ", font=("Segoe UI", 11, "bold"), bg="#1a6e2e", fg="white", relief="flat", cursor="hand2", padx=10, pady=8, command=predict).pack(pady=(14, 0))

root.mainloop()
