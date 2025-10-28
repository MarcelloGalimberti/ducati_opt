# Processo manning
# 20251028
# env neuraplprophet conda
# poi provare PuLP per ottimizzatore

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

####### Funzioni di utilità

def identifica_colonne_data(df, colonne_da_escludere=None):
    """
    Identifica le colonne di tipo data in un dataframe.
    
    Args:
        df: DataFrame da analizzare
        colonne_da_escludere: Lista di colonne da escludere dall'analisi
    
    Returns:
        Lista delle colonne identificate come date
    """
    if colonne_da_escludere is None:
        colonne_da_escludere = []
    
    date_columns = []
    for col in df.columns:
        if col not in colonne_da_escludere:
            # Verifica se è già datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(col)
            else:
                # Prova a convertire in datetime
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_columns.append(col)
                except:
                    # Se non riesce, verifica se contiene pattern di data
                    if df[col].dtype == 'object':
                        sample_values = df[col].dropna().astype(str).head(5)
                        if any(any(char in str(val) for char in ['-', '/', '2026', '2025']) for val in sample_values):
                            date_columns.append(col)
    
    return date_columns

def calcola_fabbisogno_turni_gruppo(gruppo_risorse, df_melted, df_efficienza_oee, df_calendario_melted, turni_standard_gruppo_risorse, ore_standard):
    """
    Calcola il fabbisogno turni per un gruppo risorsa specifico.
    
    Args:
        gruppo_risorse: Nome del gruppo risorsa (es. 'Stampa')
        df_melted: DataFrame con i volumi in formato long
        df_efficienza_oee: DataFrame con velocità per risorsa
        df_calendario_melted: DataFrame con giorni lavorativi
        turni_standard_gruppo_risorse: DataFrame con turni standard
        ore_standard: Ore standard di lavoro
    
    Returns:
        DataFrame con fabbisogno turni calcolato
    """
    # Filtra per il gruppo risorsa
    df_gruppo_volume = df_melted[df_melted['Gruppo_risorse'] == gruppo_risorse].reset_index(drop=True)
    
    if df_gruppo_volume.empty:
        return None
    
    # Merge con efficienza
    df_gruppo_volume = df_gruppo_volume.merge(df_efficienza_oee[['Risorsa', 'Velocità_LL']], on=['Risorsa'], how='left')
    
    # Calcola la Velocità_LL pesata su Volume per ogni Anno_Mese
    df_gruppo_volume['Volume_x_Velocità'] = df_gruppo_volume['Volume'] * df_gruppo_volume['Velocità_LL']
    
    # Calcola la velocità media pesata per Anno_Mese
    velocita_pesata = df_gruppo_volume.groupby('Anno_Mese').agg({
        'Volume_x_Velocità': 'sum',
        'Volume': 'sum'
    }).reset_index()
    
    velocita_pesata['Velocità_LL_reparto'] = velocita_pesata['Volume_x_Velocità'] / velocita_pesata['Volume']
    
    # Merge della velocità pesata nel dataframe originale
    df_gruppo_volume = df_gruppo_volume.merge(velocita_pesata[['Anno_Mese', 'Velocità_LL_reparto']], on='Anno_Mese', how='left')
    
    # Rimuovi la colonna temporanea
    df_gruppo_volume = df_gruppo_volume.drop('Volume_x_Velocità', axis=1)
    
    # Merge con calendario
    df_gruppo_volume = df_gruppo_volume.merge(df_calendario_melted[['Gruppo_risorse', 'Periodo_dt', 'Giorni_lavorativi']], 
                                            left_on=['Gruppo_risorse', 'Periodo_dt'], 
                                            right_on=['Gruppo_risorse', 'Periodo_dt'], 
                                            how='left')
    
    # Aggregazione finale
    df_gruppo_somma_volumi = df_gruppo_volume.groupby(['Anno_Mese', 'Gruppo_risorse', 'Periodo_dt']).agg({
        'Volume': 'sum',
        'Giorni_lavorativi': 'first',
        'Velocità_LL_reparto': 'first'
    }).reset_index()
    
    # Calcola fabbisogno turni
    df_gruppo_somma_volumi['Fabbisogno_turni'] = df_gruppo_somma_volumi['Volume'] / (
        df_gruppo_somma_volumi['Giorni_lavorativi'] * ore_standard * df_gruppo_somma_volumi['Velocità_LL_reparto']
    )
    
    # Merge con turni standard
    df_gruppo_somma_volumi = df_gruppo_somma_volumi.merge(
        turni_standard_gruppo_risorse[['Anno_Mese', 'Gruppo_risorse', 'Turni_standard']], 
        on=['Anno_Mese', 'Gruppo_risorse'], 
        how='left'
    )
    
    return df_gruppo_somma_volumi

def crea_grafico_fabbisogno_vs_standard(df_risultato, gruppo_risorse):
    """
    Crea un grafico confronto tra fabbisogno turni e turni standard.
    
    Args:
        df_risultato: DataFrame con i dati calcolati
        gruppo_risorse: Nome del gruppo risorsa per il titolo
    
    Returns:
        Figure plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_risultato['Anno_Mese'],
        y=df_risultato['Fabbisogno_turni'],
        name='Fabbisogno Turni'
    ))
    
    fig.add_trace(go.Bar(
        x=df_risultato['Anno_Mese'],
        y=df_risultato['Turni_standard'],
        name='Turni Standard'
    ))
    
    fig.update_layout(
        title=f'Fabbisogno Turni vs Turni Standard - {gruppo_risorse}',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Numero di Turni',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0, 
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(255, 255, 255, 0.5)',
            borderwidth=2,
            font=dict(
                size=12
            )
        )
    )
    
    return fig

####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'https://github.com/MarcelloGalimberti/ducati_opt/blob/main/LOGO-Artigrafiche_Italia.png?raw=true'
#url_immagine = 'LOGO-Artigrafiche_Italia.png'#?raw=true' #LOGO-Artigrafiche_Italia.png
# https://github.com/MarcelloGalimberti/ducati_opt/blob/main/LOGO-Artigrafiche_Italia.png

col_1, col_2 = st.columns([1, 5])

with col_1:
    st.image(url_immagine, width=200)

with col_2:
    st.title('Budget personale 2026')

st.subheader('Caricamento dati | master_data.xlsx', divider='gray')

####### Caricamento dati

uploaded_db = st.file_uploader("Carica master_data.xlsx") # nome file da caricare
if not uploaded_db:
    st.stop()

df_volume = pd.read_excel(uploaded_db, sheet_name='volumi_bgt', parse_dates=True)
df_equipaggi = pd.read_excel(uploaded_db, sheet_name='equipaggi', parse_dates=True)
df_calendario = pd.read_excel(uploaded_db, sheet_name='calendario', parse_dates=True)
df_turni = pd.read_excel(uploaded_db, sheet_name='turni') 
df_assenteismo_ferie = pd.read_excel(uploaded_db, sheet_name='assenteismo_ferie')
df_efficienza_oee = pd.read_excel(uploaded_db, sheet_name='efficienza_oee')

st.write('volume_bgt')
st.dataframe(df_volume)
st.write('equipaggi')
st.dataframe(df_equipaggi)
st.write('calendario')
st.dataframe(df_calendario)
st.write('assenteismo_ferie')
st.dataframe(df_assenteismo_ferie)
st.write('efficienza_oee')
st.dataframe(df_efficienza_oee)
st.write('turni')
st.dataframe(df_turni)


####### Variabili di supporto
ore_standard= 8


# Identifica le colonne di tipo data nel df_calendario
date_columns_calendario = identifica_colonne_data(df_calendario, ['Gruppo_risorse'])

# Melt del df_calendario usando le colonne data identificate
if date_columns_calendario:
    df_calendario_melted = df_calendario.melt(
        id_vars=['Gruppo_risorse'], 
        value_vars=date_columns_calendario,
        var_name='Periodo', 
        value_name='Giorni_lavorativi'
    )
    
    # Converte la colonna Periodo in datetime e crea Anno-Mese
    try:
        df_calendario_melted['Periodo_dt'] = pd.to_datetime(df_calendario_melted['Periodo'])
        df_calendario_melted['Anno_Mese'] = df_calendario_melted['Periodo_dt'].dt.to_period('M').astype(str)
    except:
        # Se non riesce la conversione, usa il valore originale
        df_calendario_melted['Anno_Mese'] = df_calendario_melted['Periodo']
    
    # st.write('df_calendario_melted')
    # st.dataframe(df_calendario_melted)



# Identifica le colonne di tipo data nel df_turni
date_columns_turni = identifica_colonne_data(df_turni, ['Gruppo_risorse', 'Risorsa'])

# Filtra ulteriormente per escludere colonne che contengono parole chiave non-data
parole_da_escludere = ['turni', 'giorno', 'ore', 'standard', 'medio']
date_columns_turni_filtrate = []
for col in date_columns_turni:
    col_str = str(col)  # Converte in stringa per gestire oggetti datetime
    if not any(parola.lower() in col_str.lower() for parola in parole_da_escludere):
        date_columns_turni_filtrate.append(col)

#st.write(f"Colonne date identificate nel df_turni: {date_columns_turni_filtrate}")

# Melt del df_turni usando le colonne data filtrate
if date_columns_turni_filtrate:
    df_turni_melted = df_turni.melt(
        id_vars=['Gruppo_risorse', 'Risorsa'],
        value_vars=date_columns_turni_filtrate,
        var_name='Periodo',
        value_name='Turni'
    )

    # Converte la colonna Periodo in datetime e crea Anno-Mese
    df_turni_melted['Periodo_dt'] = pd.to_datetime(df_turni_melted['Periodo'])
    df_turni_melted['Anno_Mese'] = df_turni_melted['Periodo_dt'].dt.strftime('%Y-%m')

    # Calcola i turni standard come valore medio raggruppando per Anno_Mese, Gruppo_risorse e Risorsa
    turni_standard = df_turni_melted.groupby(['Anno_Mese', 'Gruppo_risorse', 'Risorsa']).agg({
        'Turni': 'mean',
        'Periodo_dt': 'first'  # Mantiene il primo valore Periodo_dt per ogni gruppo
    }).reset_index()
    
    turni_standard.rename(columns={'Turni': 'Turni_standard'}, inplace=True)
    
    # st.write('turni_standard')
    # st.dataframe(turni_standard)
else:
    st.warning("Nessuna colonna data valida trovata nel df_turni")

#st.write('turni_standard gruppo_risorse')
turni_standard_gruppo_risorse = turni_standard.groupby(['Anno_Mese', 'Gruppo_risorse']).agg({
    'Turni_standard': 'first' # casomai media
}).reset_index()

#st.dataframe(turni_standard_gruppo_risorse)



# Grafici per ogni Gruppo_risorse con volumi per anno-mese, colorati per Risorsa

# Identifica le colonne di tipo data (datetime o che possono essere convertite in date)
date_columns = identifica_colonne_data(df_volume, ['Gruppo_risorse', 'Risorsa'])

#st.write(f"Colonne identificate come date: {date_columns}")

if date_columns:
        # Trasforma il dataframe in formato long (melt)
        df_melted = df_volume.melt(
            id_vars=['Gruppo_risorse', 'Risorsa'],
            value_vars=date_columns,
            var_name='Periodo',
            value_name='Volume'
        )
        
        # Converte la colonna Periodo in datetime e crea Anno-Mese
        try:
            df_melted['Periodo_dt'] = pd.to_datetime(df_melted['Periodo'])
            df_melted['Anno_Mese'] = df_melted['Periodo_dt'].dt.to_period('M').astype(str)
        except:
            # Se non riesce la conversione, usa il valore originale
            df_melted['Anno_Mese'] = df_melted['Periodo']
        
        # Ordina per Anno_Mese
        df_melted = df_melted.sort_values('Anno_Mese')
        
        # PRIMO GRAFICO: Grafico complessivo per tutti i gruppi
        st.subheader("Grafico Complessivo - Tutti i Gruppi Risorse", divider='gray')
        
        # Aggrega i dati per Anno_Mese e Gruppo_risorse (somma tutti i volumi per gruppo)
        df_agg = df_melted.groupby(['Anno_Mese', 'Gruppo_risorse'])['Volume'].sum().reset_index()
        
        # Converti i volumi in milioni
        df_agg['Volume_Milioni'] = df_agg['Volume'] / 1000000
        
        # Ordina i gruppi risorse nella sequenza desiderata
        order_gruppi = ['Stampa', 'Fustellatura', 'Piega_incolla', 'Villavara']
        df_agg['Gruppo_risorse'] = pd.Categorical(df_agg['Gruppo_risorse'], categories=order_gruppi, ordered=True)
        df_agg = df_agg.sort_values(['Gruppo_risorse', 'Anno_Mese'])
        
        # Crea il grafico con colori diversi per ogni gruppo
        fig_all = px.bar(
            df_agg,
            x='Anno_Mese',
            y='Volume_Milioni',
            color='Gruppo_risorse',
            facet_col='Gruppo_risorse',
            facet_col_wrap=2,  # Organizza in 2 colonne
            title='Volumi Complessivi per Tutti i Gruppi Risorse (in Milioni)',
            labels={'Volume_Milioni': 'Volume (Milioni)', 'Anno_Mese': 'Anno-Mese'},
            text='Volume_Milioni',  # Mostra i valori sopra le barre
            category_orders={'Gruppo_risorse': order_gruppi}
        )
        
        fig_all.update_layout(
            xaxis_title="Anno-Mese",
            yaxis_title="Volume (Milioni)",
            showlegend=False,  # Non serve la leggenda per questo grafico (già nei titoli facet)
            height=800  # Aumenta l'altezza per migliore leggibilità
        )
        
        # Ruota le etichette dell'asse X per tutti i subplot
        fig_all.update_xaxes(tickangle=-45)
        
        # Adatta la scala Y ai valori rappresentati per ogni sottografico
        fig_all.update_yaxes(matches=None)  # Permette scale Y indipendenti per ogni facet
        
        # Formatta i valori sopra le barre con migliore leggibilità
        fig_all.update_traces(
            texttemplate='%{text:.1f}M',  # Formato con 1 decimale e "M" per milioni
            textposition='outside',
            textfont_size=10
        )
        
        st.plotly_chart(fig_all, use_container_width=True)
        
        # Lista dei gruppi risorsa per i grafici dettagliati
        gruppi_risorse = ['Stampa', 'Fustellatura', 'Piega_incolla', 'Villavara']
        
        st.subheader("Grafici Dettagliati per Gruppo Risorsa", divider='gray')
        
        # Crea un grafico dettagliato per ogni gruppo risorsa
        for gruppo in gruppi_risorse:
            # Filtra i dati per il gruppo risorsa corrente
            df_gruppo = df_melted[df_melted['Gruppo_risorse'] == gruppo]
            
            if not df_gruppo.empty:
                # Crea il grafico a barre per il gruppo corrente
                fig = px.bar(
                    df_gruppo,
                    x='Anno_Mese',
                    y='Volume',
                    color='Risorsa',
                    title=f'Volumi per {gruppo} - Anno-Mese',
                    labels={'Volume': 'Volume', 'Anno_Mese': 'Anno-Mese', 'Risorsa': 'Risorsa'},
                    barmode='group'
                )
                
                # Personalizza il layout
                fig.update_layout(
                    xaxis_title="Anno-Mese",
                    yaxis_title="Volume",
                    legend_title="Risorsa",
                    showlegend=True,
                    xaxis_tickangle=-45  # Ruota le etichette dell'asse X per migliore leggibilità
                )
                
                # Mostra il grafico
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Nessun dato trovato per il gruppo risorsa: {gruppo}")
else:
    st.error("Nessuna colonna di tipo data trovata nel dataframe")





######### Fabbisogno turni senza ottimizzazione

st.subheader('Fabbisogno turni senza ottimizzazione', divider='gray')

# Lista dei gruppi risorsa per l'analisi
gruppi_risorse = ['Stampa', 'Fustellatura', 'Piega_incolla', 'Villavara']

# Analizza ogni gruppo risorsa
for gruppo in gruppi_risorse:
    st.subheader(f'Analisi Fabbisogno Turni - {gruppo}')
    
    # Calcola il fabbisogno turni per il gruppo
    df_risultato = calcola_fabbisogno_turni_gruppo(
        gruppo, 
        df_melted, 
        df_efficienza_oee, 
        df_calendario_melted, 
        turni_standard_gruppo_risorse, 
        ore_standard
    )
    
    if df_risultato is not None and not df_risultato.empty:
        # Mostra il dataframe risultante (opzionale, per debug)
        st.write(f'Dati calcolati per {gruppo}')
        st.dataframe(df_risultato)
        
        # Crea e mostra il grafico
        fig = crea_grafico_fabbisogno_vs_standard(df_risultato, gruppo)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostra alcune statistiche riassuntive
        fabbisogno_medio = df_risultato['Fabbisogno_turni'].mean()
        turni_standard_medio = df_risultato['Turni_standard'].mean()
        differenza_media = fabbisogno_medio - turni_standard_medio
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fabbisogno Medio", f"{fabbisogno_medio:.2f}", delta=None)
        with col2:
            st.metric("Turni Standard Medio", f"{turni_standard_medio:.2f}", delta=None)
        with col3:
            st.metric("Differenza Media", f"{differenza_media:.2f}", 
                     delta=f"{differenza_media:.2f}" if differenza_media != 0 else None)
    else:
        st.warning(f"Nessun dato disponibile per il gruppo {gruppo}")


st.subheader('Equipaggi necessari senza ottimizzazione', divider='gray')

# st.write('Equpaggi')
# st.dataframe(df_equipaggi)

df_melted_equipaggi = df_equipaggi.melt(
    id_vars=['Gruppo_risorse', 'Risorsa'],
    value_vars=identifica_colonne_data(df_equipaggi, ['Gruppo_risorse', 'Risorsa']),
    var_name='Periodo',
    value_name='Equipaggi'
)

df_melted_equipaggi['Periodo_dt'] = pd.to_datetime(df_melted_equipaggi['Periodo'])
df_melted_equipaggi['Anno_Mese'] = df_melted_equipaggi['Periodo_dt'].dt.to_period('M').astype(str)
df_melted_equipaggi = df_melted_equipaggi.merge(df_efficienza_oee[['Risorsa', 'Velocità_LL']], on=['Risorsa'], how='left')
# elimina righe con Velocità_LL mancante
df_melted_equipaggi = df_melted_equipaggi[df_melted_equipaggi['Velocità_LL'].notna()]

df_melted_equipaggi = df_melted_equipaggi.merge(df_melted[['Gruppo_risorse', 'Risorsa', 'Anno_Mese', 'Volume']],
                                                on=['Gruppo_risorse', 'Risorsa', 'Anno_Mese'], how='left')

df_melted_equipaggi['ore_macchina'] = df_melted_equipaggi['Volume'] / df_melted_equipaggi['Velocità_LL']

# se Risorsa = Mastercut, allora dividi per 5 Equipaggi
mask_mastercut = df_melted_equipaggi['Risorsa'].str.contains('Mastercut', case=False, na=False)
df_melted_equipaggi.loc[mask_mastercut, 'Equipaggi'] = df_melted_equipaggi.loc[mask_mastercut, 'Equipaggi'] / 5

df_melted_equipaggi['ore_uomo'] = df_melted_equipaggi['ore_macchina'] * df_melted_equipaggi['Equipaggi']

# st.write('df_melted_equipaggi')
# st.dataframe(df_melted_equipaggi)

df_ore_uomo_dirette_gruppo = df_melted_equipaggi.groupby(['Gruppo_risorse', 'Anno_Mese']).agg({
    'ore_uomo': 'sum'
}).reset_index()


df_ore_uomo_dirette_gruppo = df_ore_uomo_dirette_gruppo.merge(
    df_calendario_melted[['Gruppo_risorse', 'Anno_Mese', 'Giorni_lavorativi']]  ,
    on=['Gruppo_risorse', 'Anno_Mese'],
    how='left'
)

df_efficienza_oee_quadratura = df_efficienza_oee.copy()
df_efficienza_oee_quadratura = df_efficienza_oee_quadratura[['Gruppo_risorse', 'Quadratura']].drop_duplicates().reset_index(drop=True)

# st.write('df_efficienza_oee_quadratura')
# st.dataframe(df_efficienza_oee_quadratura)

# st.write('df_ore_uomo_dirette_gruppo prima del merge quadratura')
# st.dataframe(df_ore_uomo_dirette_gruppo)

# merge di df_ore_uomo_dirette_gruppo con df_efficienza_oee_quadratura per ottenere la quadratura
df_ore_uomo_dirette_gruppo = df_ore_uomo_dirette_gruppo.merge(
    df_efficienza_oee_quadratura[['Gruppo_risorse', 'Quadratura']], 
    on=['Gruppo_risorse'], 
    how='left'
)

# st.write('df_ore_uomo_dirette_gruppo dopo il merge quadratura')
# st.dataframe(df_ore_uomo_dirette_gruppo)

#df_ore_uomo_dirette_gruppo = df_ore_uomo_dirette_gruppo.merge(df_efficienza_oee_quadratura[['Quadratura']], left_on=['Gruppo_risorse'], right_on=['Gruppo_risorse'], how='left')

df_ore_uomo_dirette_gruppo['head_count'] = df_ore_uomo_dirette_gruppo['ore_uomo'] / (df_ore_uomo_dirette_gruppo['Giorni_lavorativi'] * ore_standard)

df_ore_uomo_dirette_gruppo = df_ore_uomo_dirette_gruppo.merge(df_assenteismo_ferie[['Gruppo_risorse', 'Assenteismo', 'Copertura_ferie']], on=['Gruppo_risorse'], how='left')

df_ore_uomo_dirette_gruppo['head_count_quadratura'] = df_ore_uomo_dirette_gruppo['head_count'] / (df_ore_uomo_dirette_gruppo['Quadratura']/100)

df_ore_uomo_dirette_gruppo['head_count_assenteismo'] = df_ore_uomo_dirette_gruppo['head_count_quadratura'] * (1+ df_ore_uomo_dirette_gruppo['Assenteismo'])
df_ore_uomo_dirette_gruppo['head_count_assenteismo_ferie'] = df_ore_uomo_dirette_gruppo['head_count_assenteismo'] * (1+ df_ore_uomo_dirette_gruppo['Copertura_ferie'])

df_ore_uomo_dirette_gruppo['delta_quadratura'] = df_ore_uomo_dirette_gruppo['head_count_quadratura'] - df_ore_uomo_dirette_gruppo['head_count']
df_ore_uomo_dirette_gruppo['delta_assenteismo'] = df_ore_uomo_dirette_gruppo['head_count_assenteismo'] - df_ore_uomo_dirette_gruppo['head_count_quadratura']
df_ore_uomo_dirette_gruppo['delta_ferie'] = df_ore_uomo_dirette_gruppo['head_count_assenteismo_ferie'] - df_ore_uomo_dirette_gruppo['head_count_assenteismo']

# st.write('df_calendario_melted')
# st.dataframe(df_calendario_melted)


st.write('df_ore_uomo_dirette_gruppo')
st.dataframe(df_ore_uomo_dirette_gruppo)

# Grafici impilati per Head Count per ogni Gruppo Risorse
st.subheader('Head Count per Gruppo Risorse - Composizione', divider='gray')

# Lista dei gruppi risorsa
gruppi_risorse = ['Stampa', 'Fustellatura', 'Piega_incolla', 'Villavara']

for gruppo in gruppi_risorse:
    # Filtra i dati per il gruppo risorsa corrente
    df_gruppo = df_ore_uomo_dirette_gruppo[df_ore_uomo_dirette_gruppo['Gruppo_risorse'] == gruppo]
    
    if not df_gruppo.empty:
        # Crea il grafico a barre impilate
        fig = go.Figure()
        
        # Aggiungi le barre impilate
        fig.add_trace(go.Bar(
            x=df_gruppo['Anno_Mese'],
            y=df_gruppo['head_count'],
            name='Head Count Base'
            # Rimuovo marker_color per usare i colori di default di Plotly
        ))
        
        fig.add_trace(go.Bar(
            x=df_gruppo['Anno_Mese'],
            y=df_gruppo['delta_quadratura'],
            name='Delta Quadratura'
            # Rimuovo marker_color per usare i colori di default di Plotly
        ))
        
        fig.add_trace(go.Bar(
            x=df_gruppo['Anno_Mese'],
            y=df_gruppo['delta_assenteismo'],
            name='Delta Assenteismo'
            # Rimuovo marker_color per usare i colori di default di Plotly
        ))
        
        fig.add_trace(go.Bar(
            x=df_gruppo['Anno_Mese'],
            y=df_gruppo['delta_ferie'],
            name='Delta Ferie'
            # Rimuovo marker_color per usare i colori di default di Plotly
        ))
        
        # Aggiorna il layout per barre impilate
        fig.update_layout(
            title=f'Composizione Head Count - {gruppo}',
            xaxis_title='Anno-Mese',
            yaxis_title='Numero Persone',
            barmode='stack',  # Modalità impilata
            xaxis_tickangle=-45,
            legend=dict(
                x=0,
                y=1.0,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(255, 255, 255, 0.5)',
                borderwidth=2,
                font=dict(size=12)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Aggiungi metriche riassuntive
        head_count_totale_medio = (df_gruppo['head_count'] + df_gruppo['delta_quadratura'] + df_gruppo['delta_assenteismo'] + df_gruppo['delta_ferie']).mean()
        percentuale_quadratura = (df_gruppo['delta_quadratura'] / df_gruppo['head_count'] * 100).mean()
        percentuale_assenteismo = (df_gruppo['delta_assenteismo'] / df_gruppo['head_count_quadratura'] * 100).mean()
        percentuale_ferie = (df_gruppo['delta_ferie'] / df_gruppo['head_count_assenteismo'] * 100).mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Head Count Totale Medio", f"{head_count_totale_medio:.1f}")
        with col2:
            st.metric("% Quadratura", f"{percentuale_quadratura:.1f}%")
        with col3:
            st.metric("% Assenteismo", f"{percentuale_assenteismo:.1f}%")
        with col4:
            st.metric("% Copertura Ferie", f"{percentuale_ferie:.1f}%")
    else:
        st.warning(f"Nessun dato disponibile per il gruppo {gruppo}")

st.stop()











######### Pre-processing

########## scarica excel

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Foglio1')
    return output.getvalue()

############ Analisi run 



###### scarica excel
# Crea il bottone per scaricare df_analisi
analisi = to_excel_bytes(df_analisi)
st.download_button(
    label="📥 Scarica Analisi Run Complessivo",
    data=analisi,
    file_name='Analisi_fustellatura.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)






