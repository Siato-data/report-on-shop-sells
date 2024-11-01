import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Any


# Charger les données
df: pd.DataFrame = pd.read_csv('sales_data.csv')


# Nettoyer les Données
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['product'] = df['product'].astype('string')


# Exploration des Données (EDA)
def display_basic_info(data: pd.DataFrame) -> None:
    """Affiche les premières lignes et des informations générales sur le DataFrame."""
    print(data.head())
    print(data.info())
    print(data.describe())

display_basic_info(df)


# Nettoyer les Données (suite)
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données en gérant les valeurs manquantes et les doublons, et en standardisant les colonnes."""
    print("\nValeurs manquantes par colonne :")
    print(data.isnull().sum())

    print("\nNombre de doublons :")
    print(data.duplicated().sum())

    print("\nTypes de données :")
    print(data.dtypes)

    print("\nValeurs manquantes après conversion de la date :")
    print(data['date'].isnull().sum())

    print("\nValeurs aberrantes potentielles pour 'quantity' :")
    print(data['quantity'].describe())

    print("\nValeurs aberrantes potentielles pour 'price' :")
    print(data['price'].describe())

    data['product'] = data['product'].str.strip().str.lower()
    data.to_csv('cleaned_sales_data.csv', index=False)
    return data

df = clean_data(df)


# Analyser les Données
df['total_sales'] = df['quantity'] * df['price']

total_sales_by_product: pd.Series = df.groupby('product')['total_sales'].sum()
print(total_sales_by_product)

total_quantity_by_product: pd.Series = df.groupby('product')['quantity'].sum()
print(total_quantity_by_product)

total_sales_by_date: pd.Series = df.groupby('date')['total_sales'].sum()
print(total_sales_by_date)


# Visualisation des Données
grouped_df: pd.DataFrame = df.groupby(['date', 'product'])['quantity'].sum().unstack()

# Création du barplot empilé
grouped_df.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%2024-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.title('Nombre de ventes par produit et par jour')
plt.xlabel('Date')
plt.ylabel('Nombre de ventes')
plt.legend(title='Produit')
plt.ylim(0, 20)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(range(0, 21, 2), fontsize=12)
plt.legend(title='Produit', fontsize=10, title_fontsize='13', loc='upper left')
plt.tight_layout()
plt.show()


# Analyse Approfondie (Visualisation)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Graphique de ventes par produit
product_sales: pd.Series = df.groupby('product')['quantity'].sum()
axs[0].bar(product_sales.index, product_sales.values, color='skyblue')
axs[0].set_title('Nombre de ventes par produit', fontsize=16, fontweight='bold')
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_ylim(0, 20)
axs[0].set_yticks(range(0, 21, 2))
axs[0].set_xlabel('Produit')
axs[0].set_ylabel('Quantité totale vendue')

# Graphique de ventes journalier
daily_sales: pd.Series = df.groupby('date')['quantity'].sum()
axs[1].plot(daily_sales.index, daily_sales.values, marker="o", color="coral")
axs[1].set_title("Nombre de ventes par jour", fontsize=16, fontweight='bold')
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylim(daily_sales.min() - 1, daily_sales.max() + 1)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Quantité vendue')
axs[1].set_xticks(daily_sales.index)

plt.tight_layout()
plt.show()

# Calculer la corrélation entre les colonnes price et quantity
correlation: pd.DataFrame = df[['price', 'quantity']].corr()
print(correlation)


# Insights actionnables
#def print_insights(file_path: str) -> None:
#    """Lit et affiche les insights depuis un fichier texte."""
#    with open(file_path, 'r') as file:
#        content = file.read()
#    print(content)

#print_insights('insight.txt')
