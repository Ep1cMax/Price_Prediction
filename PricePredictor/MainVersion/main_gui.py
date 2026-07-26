import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patheffects as pe
from predictor import PricePredictor
from GetDataset import datasetpath

class PricePredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Предсказание цены товара Amazon")
        self.root.geometry("1280x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.setup_styles()

        try:
            self.predictor = PricePredictor('joblib/xgboost_price_model.pkl')
        except FileNotFoundError as e:
            messagebox.showerror("Ошибка", str(e))
            self.root.destroy()
            return

        self.cat_df = pd.read_csv(datasetpath, usecols=['main_category', 'sub_category']).dropna()
        self.main_cats = sorted(self.cat_df['main_category'].unique())

        self.create_widgets()
        self.plot_feature_importance()

    def setup_styles(self):
        style = ttk.Style()
        base_font = ('Arial', 14)
        style.configure('.', font=base_font)
        style.configure('TLabel', font=('Arial', 14, 'bold'))
        style.configure('TButton', font=('Arial', 14, 'bold'))
        style.configure('TEntry', font=('Arial', 14))
        style.configure('TCombobox', font=('Arial', 13))
        style.configure('TLabelframe.Label', font=('Arial', 14, 'bold'))
        style.configure('TEntry', padding=5)
        style.configure('TCombobox', padding=5)

    def create_widgets(self):
        left_frame = ttk.LabelFrame(self.root, text="Параметры товара", padding=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=15, pady=15)

        ttk.Label(left_frame, text="Название товара:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.name_entry = ttk.Entry(left_frame, width=45)
        self.name_entry.insert(0, "GITGRNTH Mini Bag Sealer x20 per pack")
        self.name_entry.grid(row=0, column=1, pady=8, padx=(10, 0))

        ttk.Label(left_frame, text="Основная категория:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.main_cat_var = tk.StringVar()
        self.main_cat_combo = ttk.Combobox(left_frame, textvariable=self.main_cat_var,
                                           values=self.main_cats, state="readonly", width=43)
        self.main_cat_combo.current(0)
        self.main_cat_combo.grid(row=1, column=1, pady=8, padx=(10, 0))
        self.main_cat_combo.bind('<<ComboboxSelected>>', self.update_subcategories)

        ttk.Label(left_frame, text="Подкатегория:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.sub_cat_var = tk.StringVar()
        self.sub_cat_combo = ttk.Combobox(left_frame, textvariable=self.sub_cat_var,
                                          state="readonly", width=43)
        self.sub_cat_combo.grid(row=2, column=1, pady=8, padx=(10, 0))
        self.update_subcategories()

        ttk.Label(left_frame, text="Рейтинг (0-5):").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.rating_var = tk.DoubleVar(value=2.5)
        self.rating_spin = ttk.Spinbox(left_frame, from_=0.0, to=5.0, increment=0.1,
                                       textvariable=self.rating_var, width=12)
        self.rating_spin.grid(row=3, column=1, sticky=tk.W, pady=8, padx=(10, 0))

        ttk.Label(left_frame, text="Кол-во отзывов:").grid(row=4, column=0, sticky=tk.W, pady=8)
        self.no_of_ratings_entry = ttk.Entry(left_frame, width=15)
        self.no_of_ratings_entry.insert(0, "1000")
        self.no_of_ratings_entry.grid(row=4, column=1, sticky=tk.W, pady=8, padx=(10, 0))

        ttk.Label(left_frame, text="Цена со скидкой (₽):").grid(row=5, column=0, sticky=tk.W, pady=8)
        self.discount_entry = ttk.Entry(left_frame, width=15)
        self.discount_entry.insert(0, "550")
        self.discount_entry.grid(row=5, column=1, sticky=tk.W, pady=8, padx=(10, 0))

        self.predict_btn = ttk.Button(left_frame, text="Предсказать цену", command=self.predict_price)
        self.predict_btn.grid(row=6, column=0, columnspan=2, pady=20)

        self.result_var = tk.StringVar(value="—")
        self.result_label = ttk.Label(left_frame, textvariable=self.result_var,
                                      font=("Arial", 28, "bold"), foreground="darkgreen")
        self.result_label.grid(row=7, column=0, columnspan=2, pady=15)

        right_frame = ttk.LabelFrame(self.root, text="Визуализация: важность признаков", padding=15)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.figure, self.ax = plt.subplots(figsize=(7, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_subcategories(self, event=None):
        main_cat = self.main_cat_var.get()
        filtered = self.cat_df[self.cat_df['main_category'] == main_cat]
        sub_list = sorted(filtered['sub_category'].unique())
        self.sub_cat_combo['values'] = sub_list
        if sub_list:
            self.sub_cat_var.set(sub_list[0])

    def predict_price(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Внимание", "Введите название товара.")
            return

        main_cat = self.main_cat_var.get()
        sub_cat = self.sub_cat_var.get()
        if not main_cat or not sub_cat:
            messagebox.showwarning("Внимание", "Выберите категорию и подкатегорию.")
            return

        try:
            rating = self.rating_var.get()
        except tk.TclError:
            messagebox.showwarning("Внимание", "Введите корректный рейтинг (число).")
            return
        if rating < 0 or rating > 5.0:
            messagebox.showwarning("Внимание", "Рейтинг должен быть от 0 до 5.")
            return

        no_of_ratings_raw = self.no_of_ratings_entry.get().strip()
        if not no_of_ratings_raw:
            messagebox.showwarning("Внимание", "Введите количество отзывов.")
            return
        no_of_ratings = re.sub(r'[^\d]', '', no_of_ratings_raw)
        if not no_of_ratings:
            messagebox.showwarning("Внимание", "Некорректное количество отзывов (допустимы только цифры).")
            return
        try:
            no_of_ratings_int = int(no_of_ratings)
        except ValueError:
            messagebox.showwarning("Внимание", "Количество отзывов должно быть целым числом.")
            return
        if no_of_ratings_int < 0:
            messagebox.showwarning("Внимание", "Количество отзывов не может быть отрицательным.")
            return
        if no_of_ratings_int > 1_000_000:
            messagebox.showwarning("Внимание", "Количество отзывов не должно превышать 1 000 000.")
            return

        discount_raw = self.discount_entry.get().strip()
        discount_price = None
        if discount_raw:
            try:
                discount_price = float(discount_raw.replace(',', '.').replace(' ', ''))
            except ValueError:
                messagebox.showwarning("Внимание", "Некорректная цена со скидкой (введите число).")
                return
            if discount_price < 0:
                messagebox.showwarning("Внимание", "Цена со скидкой не может быть отрицательной.")
                return

        data_dict = {
            'name': [name],
            'main_category': [main_cat],
            'sub_category': [sub_cat],
            'ratings': [rating],
            'no_of_ratings': [str(no_of_ratings_int)],
            'discount_price': [discount_price if discount_price is not None else np.nan]
        }
        df = pd.DataFrame(data_dict)

        try:
            price_rub = self.predictor.predict(df, in_rub=True)[0]
            self.result_var.set(f"{price_rub:,.2f} ₽")
        except Exception as e:
            messagebox.showerror("Ошибка предсказания", str(e))

    def plot_feature_importance(self):
        self.ax.clear()
        fi_df = self.predictor.get_feature_importance()
        top_n = 15
        top = fi_df.head(top_n).copy()
        top['feature_clean'] = top['feature'].str.replace('_', ' ')
        total_importance = fi_df['importance'].sum()
        top['percent'] = (top['importance'] / total_importance) * 100
        top = top.iloc[::-1]

        bars = self.ax.barh(top['feature_clean'], top['importance'], color='steelblue')
        self.ax.set_xlabel("Важность (F-score)", fontsize=12)
        self.ax.set_title("Топ-15 важных признаков модели", fontsize=14)
        self.ax.tick_params(axis='both', labelsize=11)

        # Позиция текста: 95% от максимального значения оси X
        x_text = self.ax.get_xlim()[1] * 0.95

        for bar, percent in zip(bars, top['percent']):
            # Текст внутри/около столбца с белой обводкой
            self.ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                         f'{percent:.4f}%', va='center', ha='right',
                         fontsize=10, color='black',
                         path_effects=[pe.withStroke(linewidth=2, foreground='white')])

        self.figure.tight_layout()
        self.canvas.draw()

    def on_closing(self):
        plt.close('all')
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PricePredictionApp(root)
    root.mainloop()