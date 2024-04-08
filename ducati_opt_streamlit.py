# Librerie
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, LpMinimize
from pulp import *
#from pulp import GLPK
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import io
from io import StringIO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from PIL import Image

# Impostazioni Layout
st.set_page_config(layout="wide")
url_immagine = 'https://github.com/MarcelloGalimberti/Sentiment/blob/main/Ducati_red_logo.png?raw=true'

col_1, col_2 = st.columns([1, 6])
with col_1:
    st.image(url_immagine, width=150)
with col_2:
    st.title('Ottimizzazione cadenze linee di assemblaggio | draft ')

# Caricamento dati
st.header('Caricamento dati', divider='red')

uploaded_tgt_sales = st.file_uploader("Carica target sales (tgt_mese.xlsx)")
if not uploaded_tgt_sales:
    st.stop()
df_tgt=pd.read_excel(uploaded_tgt_sales, index_col='Model')

uploaded_calendario = st.file_uploader("Carica calendario di fabbrica (giorni_2024.xlsx)")
if not uploaded_calendario:
    st.stop()
df_giorni=pd.read_excel(uploaded_calendario, parse_dates=True)

st.dataframe(df_tgt, use_container_width=True)
#st.write(df_giorni)

# Variabili globali
veicoli_MTS = ['MTS_V2','MTS_V4']
veicoli_PAN_SF =['PAN_V2','SF_V2','PAN_V4','SF_V4']
veicoli_SCR_HYM = ['SCR','HYM','HYM_698']
veicoli_MON_SS_DVL_DSX = ['MON','SS','X_DVL','DVL','DSX']
veicoli = list(df_tgt.index)

max_capacity_MTS = 8
max_capacity_PAN_SF = 10
max_capacity_SCR_HYM = 10
max_capacity_MON_SS_DVL_DSX = 15

dict_capacity = {tuple(['MTS_V2','MTS_V4']):max_capacity_MTS,tuple(['PAN_V2','SF_V2','PAN_V4','SF_V4']):max_capacity_PAN_SF,
                 tuple(['SCR','HYM','HYM_698']):max_capacity_SCR_HYM,
                 tuple(['MON','SS','X_DVL','DVL','DSX']):max_capacity_MON_SS_DVL_DSX}

dict_fattore = {tuple(['MTS_V2','MTS_V4']):10,tuple(['PAN_V2','SF_V2','PAN_V4','SF_V4']):10.5,
                 tuple(['SCR','HYM','HYM_698']):10,
                 tuple(['MON','SS','X_DVL','DVL','DSX']):10}

dict_fattore_veicolo = {'MTS_V2':10,'MTS_V4':10,'PAN_V2':10.5,'SF_V2':10.5,'PAN_V4':10.5,'SF_V4':10.5,'SCR':10,'HYM':10,'HYM_698':10,
                        'MON':10,'SS':10,'X_DVL':10,'DVL':10,'DSX':10}


mesi = list(df_tgt.columns)

# Verifica congruenza del file sales

# test per KO
#df_tgt.loc['MTS_V2',1]=235
#df_tgt.loc['PAN_V2',2]=220

st.subheader(':orange[Verifica file tgt_mese.xlsx]')
df_resto = df_tgt.copy()
lista_ko = []
for model in df_tgt.index:
    for mese in df_tgt.columns:
        df_resto.loc[model,mese]=df_tgt.loc[model,mese]%dict_fattore_veicolo[model]
        if df_resto.loc[model,mese] != 0:
            model_mese_ko = (model,mese)
            lista_ko.append(model_mese_ko)
if len(lista_ko) == 0:
    st.markdown (':green[File tgt_mese.xlsx: multipli ok]')
else:
    st.markdown (f':red[File vendite con multipli incongurenti in: {lista_ko} ]')
    st.markdown(':red[Correggere tgt_mese.xlsx con multipli corretti e ricaricare]')
    st.stop()


# Estrazione giorni lavorativi per mese
def giorni_lav (mese):
    giorni = len(df_giorni[df_giorni['Mese']==mese])
    return giorni




# grafico giorni lavorativi
df_giorni_lavorativi = pd.DataFrame(index=mesi)

for mese in mesi:
    df_giorni_lavorativi.loc[mese,'Giorni lavorativi']=giorni_lav(mese)
fig_mesi = px.bar(df_giorni_lavorativi, x= df_giorni_lavorativi.index, y='Giorni lavorativi', title= 'Giorni lavorativi per mese', text='Giorni lavorativi')
st.plotly_chart(fig_mesi, use_container_width=True)

st.divider()

# Grafico df_tgt
df_tgt_melt=df_tgt.melt(var_name='Mese', value_name='Vendite',ignore_index=False)

fig = px.bar(df_tgt_melt, x= "Mese", y='Vendite', title= 'Vendite complesive per mese e modello',color=df_tgt_melt.index)
st.plotly_chart(fig, use_container_width=True)


st.subheader(':orange[Verifica di fattibilità]')




# test per KO
#df_tgt.loc['MTS_V4',5]=1640
#df_tgt.loc['MTS_V4',3]=1710

# ciclo per grafici domanda per linea e verifica di fattibilità
lista_linee = [veicoli_MTS,veicoli_PAN_SF,veicoli_SCR_HYM ,veicoli_MON_SS_DVL_DSX]
contatore_capacità_ko = 0
for i in range (len (lista_linee)):
    lista_linee_capacity_ko = []
    df_tgt_melt_linea = df_tgt[df_tgt.index.isin(lista_linee[i])].melt(var_name='Mese', value_name='Vendite',ignore_index=False)
    fig_linea = px.bar(df_tgt_melt_linea, x= "Mese", y='Vendite', title= f'Vendite per mese, linea: {lista_linee[i]}',color=df_tgt_melt_linea.index)
    #fig_linea.add_hline(y=) sarebbe diversa per mese
    st.plotly_chart(fig_linea, use_container_width=True)
    df_tot_mese = df_tgt_melt_linea.groupby(by='Mese').sum()
    for mese in mesi:
        df_tot_mese.loc[mese,'Giorni'] = giorni_lav(mese)
        df_tot_mese.loc[mese,'Max capacity'] = df_tot_mese.loc[mese,'Giorni']*dict_capacity[tuple(lista_linee[i])]*dict_fattore[tuple(lista_linee[i])]
        if df_tot_mese.loc[mese,'Max capacity'] >= df_tot_mese.loc[mese,'Vendite']:
            df_tot_mese.loc[mese,'Verifica_ok'] = True
        else:
            df_tot_mese.loc[mese,'Verifica_ok'] = False
            lista_linee_capacity_ko.append(mese)
    
    st.write('Capacità produttiva verificata: ',df_tot_mese['Verifica_ok'].min())
    if df_tot_mese['Verifica_ok'].min() == False:
        contatore_capacità_ko +=1
        mesi_ko = ' | '.join(map(str,lista_linee_capacity_ko))
        st.markdown(f':red[Capacità produttiva insufficiente in: {mesi_ko} ]')

if contatore_capacità_ko >0:
    st.subheader(':red[Modificare volumi in linee e mesi indicati (file tgt_mese.xlsx)]')
    st.stop()

# Funzioni di appoggio

# Estrazione dizionari target sales per linea e modello per mese e divide per multipli 10 o 10.5
def dizionari_tgt (mese):
    sales_MTS = {'MTS_V2': df_tgt.loc['MTS_V2',mese]/10, 'MTS_V4':  df_tgt.loc['MTS_V4',mese]/10}
    sales_PAN_SF = {'PAN_V2': df_tgt.loc['PAN_V2',mese]/10.5, 'SF_V2': df_tgt.loc['SF_V2',mese]/10.5, 
                   'PAN_V4': df_tgt.loc['PAN_V4',mese]/10.5, 'SF_V4': df_tgt.loc['SF_V4',mese]/10.5}
    sales_SCR_HYM = {'SCR': df_tgt.loc['SCR',mese]/10, 'HYM': df_tgt.loc['HYM',mese]/10, 'HYM_698': df_tgt.loc['HYM_698',mese]/10}
    sales_MON_SS_DVL_DSX = {'MON': df_tgt.loc['MON',mese]/10, 'SS': df_tgt.loc['SS',mese]/10, 'X_DVL': df_tgt.loc['X_DVL',mese]/10,
                           'DVL': df_tgt.loc['DVL',mese]/10, 'DSX': df_tgt.loc['DSX',mese]/10}
    return sales_MTS, sales_PAN_SF, sales_SCR_HYM, sales_MON_SS_DVL_DSX

# Ordinamento mesi
df_avg_production = pd.DataFrame(columns=['Mese','Giorni','Produzione','Avg','Asc'])
df_avg_production['Mese']=mesi

lista_giorni_lavorativi = []
for mese in mesi:
    gg_lav = giorni_lav(mese)
    lista_giorni_lavorativi.append(gg_lav)

df_avg_production['Giorni']=lista_giorni_lavorativi

lista_produzione = []
for mese in mesi:
    totale_mese = df_tgt[mese].sum()
    lista_produzione.append(totale_mese)

df_avg_production['Produzione']=lista_produzione
df_avg_production['Avg']=df_avg_production['Produzione']/df_avg_production['Giorni']

for i in range (len (df_avg_production)):
    if i == 10:
        if df_avg_production.loc[i,'Avg'] >= df_avg_production.loc[0,'Avg']:
            df_avg_production.loc[i,'Asc'] = False
        else:
            df_avg_production.loc[i,'Asc'] = True
    else:
        if df_avg_production.loc[i,'Avg'] >= df_avg_production.loc[i+1,'Avg']:
             df_avg_production.loc[i,'Asc'] = False
        else:
            df_avg_production.loc[i,'Asc'] = True 

#st.write(df_avg_production)

#st.stop()

st.header('Ottimizzazione cadenze', divider='red')

# Funzione di ottimizzazione
# decorare
@st.cache_resource
def ottimizza (mese):
    model_linea = LpProblem(name='linea', sense=LpMinimize)
    # variabili decisionali: giorni nel mese e target sales
    giorni = list(range(1,giorni_lav(mese)+1))
    sales_MTS, sales_PAN_SF, sales_SCR_HYM, sales_MON_SS_DVL_DSX = dizionari_tgt(mese)
    # variabili decisionali: produzioni per linea
    produzione_MTS = LpVariable.dicts('produzione_MTS', [(v,g) for v in veicoli_MTS for g in giorni],lowBound=0, cat = 'Integer')
    produzione_PAN_SF = LpVariable.dicts('produzione_PAN_SF', [(v,g) for v in veicoli_PAN_SF for g in giorni],lowBound=0, cat = 'Integer')
    produzione_SCR_HYM = LpVariable.dicts('produzione_SCR_HYM', [(v,g) for v in veicoli_SCR_HYM for g in giorni],lowBound=0, cat = 'Integer')
    produzione_MON_SS_DVL_DSX = LpVariable.dicts('produzione_MON_SS_DVL_DSX', [(v,g) for v in veicoli_MON_SS_DVL_DSX for g in giorni],lowBound=0, cat = 'Integer')
    # variabili decisionali: totale produzione per linea
    produzione_totale_MTS = {(giorno): LpVariable(f"produzione_totale_MTS_{giorno}", lowBound=0) for giorno in giorni}
    produzione_totale_PAN_SF = {(giorno): LpVariable(f"produzione_totale_PAN_SF_{giorno}", lowBound=0) for giorno in giorni}
    produzione_totale_SCR_HYM = {(giorno): LpVariable(f"produzione_totale_SCR_HYM_{giorno}", lowBound=0) for giorno in giorni}
    produzione_totale_MON_SS_DVL_DSX = {(giorno): LpVariable(f"produzione_totale_MON_SS_DVL_DSX_{giorno}", lowBound=0) for giorno in giorni}
    # variabili decisionali: differenze
    differenza_totale_MTS = {(giorno): LpVariable(f"differenza_totale_MTS_{giorno}", lowBound=0) for giorno in giorni  if giorno > 1}
    differenza_totale_PAN_SF = {(giorno): LpVariable(f"differenza_totale_PAN_SF_{giorno}", lowBound=0) for giorno in giorni  if giorno > 1}
    differenza_totale_SCR_HYM = {(giorno): LpVariable(f"differenza_totale_SCR_HYM_{giorno}", lowBound=0) for giorno in giorni  if giorno > 1}
    differenza_totale_MON_SS_DVL_DSX = {(giorno): LpVariable(f"differenza_totale_MON_SS_DVL_DSX_{giorno}", lowBound=0) for giorno in giorni  if giorno > 1}
    # assegna valori alle produzioni totali
    for giorno in giorni:
        produzione_totale_MTS[giorno] = sum(produzione_MTS[veicolo,giorno] for veicolo in veicoli_MTS)
        produzione_totale_PAN_SF[giorno] = sum(produzione_PAN_SF[veicolo,giorno] for veicolo in veicoli_PAN_SF)
        produzione_totale_SCR_HYM[giorno] = sum(produzione_SCR_HYM[veicolo,giorno] for veicolo in veicoli_SCR_HYM)
        produzione_totale_MON_SS_DVL_DSX[giorno] = sum(produzione_MON_SS_DVL_DSX[veicolo,giorno] for veicolo in veicoli_MON_SS_DVL_DSX)
    # vincoli di capacità
    for giorno in giorni:
        model_linea += lpSum(produzione_MTS[veicolo,giorno] for veicolo in veicoli_MTS) <= max_capacity_MTS
        model_linea += lpSum(produzione_PAN_SF[veicolo,giorno] for veicolo in veicoli_PAN_SF) <= max_capacity_PAN_SF
        model_linea += lpSum(produzione_SCR_HYM[veicolo,giorno] for veicolo in veicoli_SCR_HYM) <= max_capacity_SCR_HYM
        model_linea += lpSum(produzione_MON_SS_DVL_DSX[veicolo,giorno] for veicolo in veicoli_MON_SS_DVL_DSX) <= max_capacity_MON_SS_DVL_DSX
    # vincoli di volume
    for veicolo in veicoli_MTS:
        model_linea += lpSum(produzione_MTS[veicolo,giorno] for giorno in giorni) == sales_MTS[veicolo]
    for veicolo in veicoli_PAN_SF:
        model_linea += lpSum(produzione_PAN_SF[veicolo,giorno] for giorno in giorni) == sales_PAN_SF[veicolo]
    for veicolo in veicoli_SCR_HYM:
        model_linea += lpSum(produzione_SCR_HYM[veicolo,giorno] for giorno in giorni) == sales_SCR_HYM[veicolo]
    for veicolo in veicoli_MON_SS_DVL_DSX:
        model_linea += lpSum(produzione_MON_SS_DVL_DSX[veicolo,giorno] for giorno in giorni) == sales_MON_SS_DVL_DSX[veicolo]
    # vincoli per le differenze
    for giorno in giorni:
        if giorno > 1:
            # MTS
            model_linea += differenza_totale_MTS[giorno] >= produzione_totale_MTS[giorno] - produzione_totale_MTS[giorno - 1]
            model_linea += differenza_totale_MTS[giorno] >= produzione_totale_MTS[giorno - 1] - produzione_totale_MTS[giorno]
            # PAN_SF
            model_linea += differenza_totale_PAN_SF[giorno] >= produzione_totale_PAN_SF[giorno] - produzione_totale_PAN_SF[giorno - 1]
            model_linea += differenza_totale_PAN_SF[giorno] >= produzione_totale_PAN_SF[giorno - 1] - produzione_totale_PAN_SF[giorno]
            # SCR_HYM
            model_linea += differenza_totale_SCR_HYM[giorno] >= produzione_totale_SCR_HYM[giorno] - produzione_totale_SCR_HYM[giorno - 1]
            model_linea += differenza_totale_SCR_HYM[giorno] >= produzione_totale_SCR_HYM[giorno - 1] - produzione_totale_SCR_HYM[giorno]
            # MON_SS_DVL_DSX
            model_linea += differenza_totale_MON_SS_DVL_DSX[giorno] >= produzione_totale_MON_SS_DVL_DSX[giorno] - produzione_totale_MON_SS_DVL_DSX[giorno - 1]
            model_linea += differenza_totale_MON_SS_DVL_DSX[giorno] >= produzione_totale_MON_SS_DVL_DSX[giorno - 1] - produzione_totale_MON_SS_DVL_DSX[giorno]
    # funzione di ottimizzazione
    model_linea += lpSum(differenza_totale_MTS[giorno] for giorno in giorni if giorno > 1) + lpSum(differenza_totale_PAN_SF[giorno] for giorno in giorni if giorno > 1) + lpSum(differenza_totale_SCR_HYM[giorno] for giorno in giorni if giorno > 1) + lpSum(differenza_totale_MON_SS_DVL_DSX[giorno] for giorno in giorni if giorno > 1)
    return model_linea, produzione_MTS, produzione_PAN_SF, produzione_SCR_HYM, produzione_MON_SS_DVL_DSX

# Tabella mese
def tabella_mese(modello, mese):
    df_tabella = pd.DataFrame(columns=['Mese','Veicolo','Giorno','Qty'])
    produzione_MTS = modello[1]
    produzione_PAN_SF = modello[2]
    produzione_SCR_HYM = modello[3]
    produzione_MON_SS_DVL_DSX = modello[4]
    i=0
    for veicolo in veicoli:
        for giorno in giorni:
            df_tabella.loc[i,'Mese']=mese
            df_tabella.loc[i,'Veicolo']=veicolo
            df_tabella.loc[i,'Giorno']=giorno
            if veicolo[:3]=='MTS':
                df_tabella.loc[i,'Qty']=produzione_MTS[veicolo,giorno].value()
            elif (veicolo[:3]=='PAN' or veicolo[:3]=='SF_'):
                df_tabella.loc[i,'Qty']=produzione_PAN_SF[veicolo,giorno].value()
            elif (veicolo[:3]=='SCR' or veicolo[:3]=='HYM'):
                df_tabella.loc[i,'Qty']=produzione_SCR_HYM[veicolo,giorno].value()
            else:
                df_tabella.loc[i,'Qty']=produzione_MON_SS_DVL_DSX[veicolo,giorno].value()
            i+=1
        i+=1
    return df_tabella

# Ciclo per un anno, non ordinato
df_scheduling = pd.DataFrame(columns=['Mese','Veicolo','Giorno','Qty'])
for mese in mesi:
    giorni = list(range(1,giorni_lav(mese)+1))
    model_mese = ottimizza(mese)
    model_mese[0].solve()
    tabella = tabella_mese(model_mese,mese)
    df_scheduling = pd.concat([df_scheduling,tabella])
df_scheduling.reset_index(inplace=True, drop=True)
for i in range (len(df_scheduling)):
    if df_scheduling.loc[i,'Veicolo'] in veicoli_PAN_SF:
        df_scheduling.loc[i,'Qty_reale'] = df_scheduling.loc[i,'Qty']*10.5
    else:
        df_scheduling.loc[i,'Qty_reale'] = df_scheduling.loc[i,'Qty']*10

# Pivot
df_pivot = pd.pivot_table(df_scheduling,index=['Mese','Giorno'], columns='Veicolo', values='Qty_reale',aggfunc='sum')
df_pivot.reset_index(inplace=True)

# Ordinamento
i=0
df_pivot_ordinato = pd.DataFrame(columns = df_pivot.columns)
for mese in mesi:
    df_mese_ordinato = df_pivot[df_pivot['Mese']==mese].sort_values(by='Giorno', ascending=df_avg_production.loc[i,'Asc'])
    df_pivot_ordinato = pd.concat([df_pivot_ordinato,df_mese_ordinato])
    i+=1
df_pivot_ordinato.reset_index(inplace=True, drop=True)
df_pivot_ordinato['Data']=df_giorni['Data']


# fare statistiche di ottimizzazione


df_final = df_pivot_ordinato.drop(columns=['Giorno'])
df_final = df_final[['Data','Mese', 'DSX', 'DVL', 'HYM', 'HYM_698', 'MON', 'MTS_V2', 'MTS_V4',
       'PAN_V2', 'PAN_V4', 'SCR', 'SF_V2', 'SF_V4', 'SS', 'X_DVL']]

df_final['Data']=df_final['Data'].dt.strftime('%d/%m/%Y')
df_final['Totale']=df_final[veicoli].sum(axis=1)
df_final['Tot_MTS_V2_MTS_V4'] = df_final[veicoli_MTS].sum(axis=1)
df_final['Tot_PAN_V2_SF_V2_PAN_V4_SF_V4']=df_final[veicoli_PAN_SF].sum(axis=1)
df_final['Tot_SCR_HYM_HYM_698']=df_final[veicoli_SCR_HYM].sum(axis=1)
df_final['Tot_MON_SS_X_DVL_DVL_DSX']=df_final[veicoli_MON_SS_DVL_DSX].sum(axis=1)
st.dataframe(df_final, use_container_width=True)


# scarica XLSX
def scarica_excel(df, filename):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.close()
    st.download_button(
        label="Download Excel workbook",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.ms-excel"
    )

scarica_excel(df_final, 'df_in_excel')


fig_line = px.line(df_final, x="Data", y="Totale", title='Produzione totale per giorno')
st.plotly_chart(fig_line, use_container_width=True)

fig_line_partizione = px.line(df_final, x="Data", y=['Tot_MTS_V2_MTS_V4','Tot_PAN_V2_SF_V2_PAN_V4_SF_V4','Tot_SCR_HYM_HYM_698','Tot_MON_SS_X_DVL_DVL_DSX'], title='Produzione totale per giorno per linea')
st.plotly_chart(fig_line_partizione, use_container_width=True)

# ciclo per grafici produzione per linea
df_final_chart = df_final.copy()
df_final_chart.set_index('Data', inplace=True)
lista_linee = [veicoli_MTS,veicoli_PAN_SF,veicoli_SCR_HYM ,veicoli_MON_SS_DVL_DSX]
for i in range (len (lista_linee)):
    df_final_linea = df_final_chart[df_final_chart.columns[df_final_chart.columns.isin(lista_linee[i])]]
    fig_linea_final = px.bar(df_final_linea, x= df_final_chart.index, y=lista_linee[i], title= f'Dettaglio produzione: {lista_linee[i]}')
    st.plotly_chart(fig_linea_final, use_container_width=True)

#st.stop()
