import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# load dataset
df = pd.read_excel('mixuesales.xlsx')

df['tanggal'] = pd.to_datetime(df['tanggal'], format="%d-%m-%Y")

df["bulan"] = df["tanggal"].dt.month
df["hari"] = df["tanggal"].dt.weekday

df["bulan"].replace([i for i in range(1, 12+1)], ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"], inplace=True)
df["hari"].replace([i for i in range(6+1)], ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], inplace=True)

st.title("Analisis Pola Pembelian Pelanggan Menggunakan Algoritma Apriori")

def get_data(weekday_weekend='', bulan='', hari=''):
    data = df.copy()
    filtered = data.loc[
        (data["bulan"].str.contains(bulan.title())) &
        (data["hari"].str.contains(hari.title()))
    ]
    return filtered if filtered.shape[0] else "No Result"

def user_input_features():
    produk = st.selectbox("Produk", ['berry bean sundae','boba shake','boba sundae','brown sugar pearl milk tea','chocolate lucky sundae','chocolate oreo smoothies','chocolate sundae','coconut jelly milk tea','creamy mango boba','earl grey with 2 toppings','fresh squeezed lemonade','hawaiian fruit tea','ice cream earl grey tea','ice cream jasmine tea','jasmine tea with 2 toppings','kiwi fruit tea','kiwi smoothies','lemon earl grey tea','lemon jasmine tea','mango oats jasmine tea','mango smoothies','mango sundae','oats milk tea','original earl grey tea','original jasmine tea','oreo sundae','passion fruit jasmine tea','peach earl grey tea','peach tea','pearl milk tea','red bean milk tea','signature mixed milk tea','strawberry lucky sundae','strawberry mishake','strawberry smoothies','sundae','supreme mixed milk tea'])
    bulan = st.select_slider("Bulan", ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"], value="Okt")
    hari = st.select_slider("Hari", ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"], value="Sab")
    
    return produk, bulan, hari

produk, bulan, hari = user_input_features()

data = get_data(produk.lower(), bulan, hari)

def encode(x):
    if x<=0:
        return 0
    elif x>=1:
        return 1
if type(data) != type ("No Result"):
    produk_count = data.groupby(["id_transaksi", "produk"])["produk"].count().reset_index(name="count")
    produk_count_pivot = produk_count.pivot_table(index='id_transaksi', columns='produk', values='count', aggfunc='sum').fillna(0)
    produk_count_pivot = produk_count_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(produk_count_pivot, min_support=support, use_colnames=True)


    rules = association_rules(frequent_items, metric="lift", min_threshold=0)
    rules['conf_supp'] = rules['confidence'] * rules['support']
    # Inisialisasi penyimpanan aturan yang unik
    unique_rules = []
    rules_checked = set()

    # Iterasi melalui setiap aturan untuk menyaring aturan yang unik
    for index, row in rules.iterrows():
        rule = (frozenset(row['antecedents']), frozenset(row['consequents']))
        rev_rule = (frozenset(row['consequents']), frozenset(row['antecedents']))

        if rule not in rules_checked and rev_rule not in rules_checked:
            unique_rules.append(row)
            rules_checked.add(rule)

    # Konversi hasil penyaringan aturan unik menjadi DataFrame
    unique_rules_df = pd.DataFrame(unique_rules)
    selected_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conf_supp']
    top_unique_rules = unique_rules_df[selected_columns].sort_values(by='conf_supp', ascending=False).head(10)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_produk_df(produk_antecedents):
    data = top_unique_rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"] == produk_antecedents].iloc[0,:])

if type(data) != type("No Result"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f"Jika membeli **{produk}**, maka membeli **{return_produk_df(produk)[1]}** secara bersamaan")

