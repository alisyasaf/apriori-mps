import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from openpyxl import load_workbook

st.title("Analisis Pola Pembelian Pelanggan Menggunakan Algoritma Apriori (Studi Kasus di Mixue Pasar Sleman)")

# Load dataset
uploaded_file = st.file_uploader("Pilih file Excel.", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    st.write("Silakan upload file Excel.")

# Cek apakah dataframe dari file yang diupload ada
if 'df' in locals():
    st.markdown("Berikut adalah 5 baris teratas dari dataset yang telah dipilih.")
    st.write(df.head())

    df['tanggal'] = pd.to_datetime(df['tanggal'], format="%d-%m-%Y")

# input dari user berupa slider   
    def user_input_features():
        support = st.slider("Minimum Support", min_value=0.01, max_value=0.2, value=0.01, step=0.01)
        st.markdown("Support adalah persentase itemset dalam sebuah kumpulan data. Menunjukkan seberapa sering suatu aturan asosiasi muncul dalam dataset atau seberapa umum kombinasi item tertentu muncul bersama. Nilai support menggambarkan frekuensi relatif dari aturan tersebut.")
        lift = st.slider("Minimum Lift", min_value=0, max_value=3, value=1, step=1)
        st.markdown("Nilai lift ratio bertujuan untuk menentukan validitas dari sebuah aturan asosiasi. Nilai lift di atas 1 menunjukkan bahwa aturan tersebut signifikan.")
        return support, lift

    support, lift = user_input_features()

    data = df.copy()
    # proses encoding
    if not data.empty:
        def encode(x):
            if x <= 0:
                return 0
            elif x >= 1:
                return 1

        produk_count = data.groupby(["id_transaksi", "produk"])["produk"].count().reset_index(name="count")
        produk_count_pivot = produk_count.pivot_table(index='id_transaksi', columns='produk', values='count', aggfunc='sum').fillna(0)
        produk_count_pivot = produk_count_pivot.applymap(encode)

        frequent_items = apriori(produk_count_pivot, min_support=support, use_colnames=True)

        rules = association_rules(frequent_items, metric="lift", min_threshold=lift)
        rules['conf_supp'] = rules['confidence'] * rules['support']
        unique_rules = []
        rules_checked = set()

        for index, row in rules.iterrows():
            rule = (row['antecedents']), (row['consequents'])
            rev_rule = (row['consequents']), (row['antecedents'])

            if rule not in rules_checked and rev_rule not in rules_checked:
                row['antecedents'] = ', '.join(map(str, row['antecedents']))
                row['consequents'] = ', '.join(map(str, row['consequents']))

                unique_rules.append(row)
                rules_checked.add(rule)

        unique_rules_df = pd.DataFrame(unique_rules)
        selected_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conf_supp']

        try:
            top_unique_rules = unique_rules_df[selected_columns].sort_values(by='confidence', ascending=False).head(10)
        except KeyError as e:
            pass

        # Menampilkan hasil
        st.success("Hasil Aturan Asosiasi : ")

        # Menambahkan pengecekan apakah top_unique_rules sudah terdefinisi sebelum menampilkan table
        if 'top_unique_rules' in locals():
            st.table(top_unique_rules)
            st.markdown("Keterangan:")
            st.markdown("antecedents: Himpunan item yang muncul sebagai sebab dalam suatu aturan asosiasi.")
            st.markdown("consequents: Himpunan item yang muncul sebagai hasil atau konsekuensi dalam suatu aturan asosiasi.")
            st.markdown("support: Menunjukkan seberapa sering suatu aturan asosiasi muncul dalam dataset atau seberapa umum kombinasi item tertentu muncul bersama. Nilai support menggambarkan frekuensi relatif dari aturan tersebut.")
            st.markdown("confidence: Mengukur seberapa kuat suatu aturan asosiasi. Nilai confidence mencerminkan probabilitas bahwa konsekuensi (consequents) akan terjadi jika sebab (antecedents) telah terjadi. Nilai ini berkisar antara 0 dan 1.")
            st.markdown("lift: Merupakan ukuran seberapa besar peningkatan probabilitas dari konsekuensi (consequents) terjadi ketika sebab (antecedents) terjadi. Lift yang lebih besar dari 1 menunjukkan adanya asosiasi yang lebih kuat.")
            st.markdown("conf_supp: Adalah hasil perkalian dari nilai confidence dan support. Kombinasi nilai ini memberikan informasi tambahan tentang kekuatan aturan asosiasi dan seberapa sering aturan tersebut terjadi dalam dataset.")
        else:
            st.warning("Tidak ada aturan asosiasi yang ditemukan untuk kriteria yang dipilih.")
    else:
        st.warning("Tidak ada hasil yang ditemukan untuk kriteria yang dipilih.")
else:
    st.error("Data belum diupload atau tidak valid.", icon="🚨")
