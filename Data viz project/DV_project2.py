import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import altair as alt


@st.cache_data
def load_data():
    df = pd.read_csv('Valeurs foncières 2023.csv',delimiter=',', low_memory=False)

    # We delete empty columns
    df.dropna(axis=1,how='all', inplace=True)

    # We delete lines where 'valeur_foncière' is null, because it's the data we want to study
    df.dropna(subset=['valeur_fonciere'], inplace=True)


    # We delete columns we don't need
    df.drop(['id_mutation','date_mutation','numero_disposition','nature_mutation',
                    'adresse_numero','adresse_suffixe','adresse_nom_voie','adresse_code_voie',
                    'code_commune','nom_commune','ancien_code_commune','ancien_nom_commune','id_parcelle',
                    'numero_volume','lot1_numero','lot2_numero','lot3_numero',
                    'lot4_numero','lot5_numero','code_nature_culture_speciale','nature_culture',
                    'code_nature_culture','nature_culture_speciale','code_type_local'], axis=1, inplace=True)


    df['valeur_fonciere'] = df['valeur_fonciere'].replace(',', '.').astype(float)
    df['lot1_surface_carrez'] = df['lot1_surface_carrez'].replace(',', '.').astype(float)
    df['lot2_surface_carrez'] = df['lot2_surface_carrez'].replace(',', '.').astype(float)
    df['lot3_surface_carrez'] = df['lot3_surface_carrez'].replace(',', '.').astype(float)
    df['lot4_surface_carrez'] = df['lot4_surface_carrez'].replace(',', '.').astype(float)
    df['lot5_surface_carrez'] = df['lot5_surface_carrez'].replace(',', '.').astype(float)

    # We delete duplicates
    df.drop_duplicates(keep='first', inplace=True)

    # We delete aberrant datas
    df = df.drop(df[df['valeur_fonciere'] > 10000000].index)
    df = df.drop(df[df['lot1_surface_carrez'] > 250].index)
    df = df.drop(df[df['lot2_surface_carrez'] > 300].index)
    df = df.drop(df[df['lot3_surface_carrez'] > 200].index)
    df = df.drop(df[df['lot4_surface_carrez'] > 200].index)
    df = df.drop(df[df['lot5_surface_carrez'] > 200].index)
    df = df.drop(df[df['nombre_lots'] > 30].index)
    df = df.drop(df[df['surface_reelle_bati'] > 100000].index)
    df = df.drop(df[df['surface_terrain'] > 5000].index)
    # df = df.drop(df[df['nombre_pieces_principales'] > 20].index)

    # We delete lines where 'code_postal' is null
    df = df.dropna(subset=['code_postal'])

    df['code_postal'] = df['code_postal'].apply(lambda x: '{:05d}'.format(int(x)) if isinstance(x, float) else x)
    df['code_postal'] = df['code_postal'].apply(lambda x: x.rjust(5, '0'))

    df['lot1_surface_carrez'].fillna(0, inplace=True)
    df['lot2_surface_carrez'].fillna(0, inplace=True)
    df['lot3_surface_carrez'].fillna(0, inplace=True)
    df['lot4_surface_carrez'].fillna(0, inplace=True)
    df['lot5_surface_carrez'].fillna(0, inplace=True)
    df['surface_terrain'].fillna(0, inplace=True)

    df = df.drop(df[(df['lot1_surface_carrez'] == 0) &
        (df['lot2_surface_carrez'] == 0) &
        (df['lot3_surface_carrez'] == 0) &
        (df['lot4_surface_carrez'] == 0) &
        (df['lot5_surface_carrez'] == 0) &
        (df['surface_terrain'] == 0)].index)

    df['prix_m2'] = df['valeur_fonciere'] / (df['lot1_surface_carrez'] +
                                             df['lot2_surface_carrez'] +
                                             df['lot3_surface_carrez'] +
                                             df['lot4_surface_carrez'] +
                                             df['lot5_surface_carrez'] +
                                             (df['surface_terrain']))
    return df


### Data vizualisation ###

st.title('Housing Prices in 2023 in France')
with st.sidebar:
    st.header("Presented by Fabien TUNCEL")
    st.write("**LinkedIn :** https://www.linkedin.com/in/fabien-tuncel/\n")
    st.write("**GitHub :** https://github.com/FabiGBB")

    if st.button("**Celebrate**"):
        st.balloons()
df = load_data()
agree = st.checkbox('Display the Dataframe')
if agree:
    st.text("Here are the Data we are going to use.")
    st.write(df.head(100))


# 1. Map of housings
sbox_option = st.selectbox('What do you want to display', df['type_local'].dropna().unique())
df_display = df[::20]  # We have so much data, we need to drop 19/20 of it
df_map = df_display[['longitude','latitude']].where(df['type_local'] == sbox_option)
df_map.dropna(subset=['longitude','latitude'], inplace=True)
st.write('## 1. Locations of the data we use.')
st.map(df_map)


# 2. Pie_chart showing frequencies of types of locals
st.write('## 2. Frequency of differents types of locals')
labels = df['type_local'].value_counts().index
sizes = df['type_local'].value_counts().values
fig1, ax1 = plt.subplots(1, 1)
ax1.pie(sizes, labels=labels, autopct='%1.1f%%')
ax1.axis('equal')
st.pyplot(fig1)


# 3. Histogram of the number of locals in IDF
st.write('## 3. Histogram of the number of locals in Ile-de-France')
df_hist = df[df['code_postal'].astype(str).str[:2].isin(['75', '77', '78', '91', '92', '93', '94', '95'])].copy()
fig2, ax2 = plt.subplots(1, 1)
ax2.hist(df_hist['code_departement'].sort_values(), bins=8)
plt.xlabel('Department')
plt.ylabel('Number')
st.pyplot(fig2)


# 4. Map of average prices per square meter per department
url = 'https://www.data.gouv.fr/fr/datasets/r/90b9341a-e1f7-4d75-a73c-bbc010c7feeb'
departements = gpd.read_file(url)
departements.to_file("departements.geojson", driver='GeoJSON')

france = gpd.read_file('departements.geojson')
france = france[~france['code'].isin(['2A', '2B'])]
france['code'] = france['code'].astype(str)

prixm = df.groupby('code_departement')['prix_m2'].mean().reset_index()
prixm['code_departement'] = prixm['code_departement'].astype(str).apply(lambda x: x.zfill(2))
merge = pd.merge(france, prixm, left_on='code', right_on='code_departement')

fig3, ax3 = plt.subplots(1, 1)
merge.plot(column='prix_m2', ax=ax3, legend=True, vmin=0, vmax=10000, cmap='coolwarm')
plt.title('4. Average prices per department (€/m2)')
st.pyplot(fig3)


# 5. Map of average prices of square meter in Ile-de-France
departements_idf = departements[departements['code'].isin(['75', '77', '78', '91', '92', '93', '94', '95'])]
prixm = df.groupby('code_departement')['prix_m2'].mean().reset_index()
prixm['code_departement'] = prixm['code_departement'].astype(str).apply(lambda x: x.zfill(2))
merge = pd.merge(departements_idf, prixm, left_on='code', right_on='code_departement')

fig4, ax4 = plt.subplots(1, 1)
merge.plot(column='prix_m2', ax=ax4, legend=True, vmin=0, vmax=13000, cmap='coolwarm')
plt.title('5. Average prices per department\nin Ile-de-France (€/m2)')
st.pyplot(fig4)


# 6. Histogram of average prices per m2 per department in Ile-de-France
df_idf = df[df['code_postal'].astype(str).str[:2].isin(['75', '77', '78', '91', '92', '93', '94', '95'])].copy()
df_idf['code_departement'] = df['code_departement'].astype(str)
df_idf_grouped = df_idf.groupby('code_departement')['prix_m2'].mean()
st.write("## 6. Average prices of a square meter in Ile-de-France")
st.line_chart(df_idf_grouped)


# 7. Bar_chart of average prices per m2 per district in Paris
df_paris = df[df['code_postal'].astype(str).str[:2] == '75'].copy()
df_paris['code_arrondissement'] = df_paris['code_postal'].astype(str).str[-2:].astype(int)
df_paris_grouped = df_paris.groupby('code_arrondissement')['prix_m2'].mean()
st.write("## 7. Average prices of a square meter in Paris")
st.bar_chart(df_paris_grouped)


# 8. Map of average sizes of housings per departments
df_size = df.groupby('code_departement')['surface_terrain'].mean().reset_index()
df_size['code_departement'] = df_size['code_departement'].astype(str).apply(lambda x: x.zfill(2))
merge = pd.merge(france, df_size, left_on='code', right_on='code_departement')

fig5, ax5 = plt.subplots(1, 1)
merge.plot(column='surface_terrain', ax=ax5, legend=True, vmin=0, vmax=1300, cmap='coolwarm')
plt.title('8. Average size of housings per department (m2)')
st.pyplot(fig5)


# 9. Boxplot of sizes surfaces depending of the department
st.write("## 9. Average sizes of locals in Ile-de-France with Altair")
chart = alt.Chart(df_idf[::20][['code_departement','surface_terrain']]).mark_boxplot().encode(
    x=alt.X('code_departement:N', title='Department'),
    y=alt.Y('surface_terrain:Q', title='Surface of the land')
).properties(width=600)
st.altair_chart(chart, use_container_width=True)

