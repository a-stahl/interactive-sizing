import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title('Interactive Sizing App')

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache)")

# st.subheader('Sample Plot')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# Some number in the range 0-23
initial_level = 4
hour_to_filter = st.slider('diameter', 1, 10, initial_level)

#plot
lower_limit = hour_to_filter
upper_limit = 10
data = pd.read_csv('plot_data.csv', header=None)
data.columns = ['ocr_val', 'hours']
data['hours'] -= data['hours'][0]
data['date'] = pd.to_datetime(27,errors='ignore', unit='d',origin='2022-08')
data['date'] = data['date'] + pd.to_timedelta(data['hours'], unit='h')
data['lower_limit'] = lower_limit
data['upper_limit'] = upper_limit
data['minutes'] = data['hours']*60
fig1 = px.line(data, x='date', y='ocr_val',labels=dict(date="", ocr_val="OCR (GMP)"))
fig2 = go.Figure(data=[go.Scatter(
    x = data['date'],
    y = data['lower_limit'],
    stackgroup='one'),
                       go.Scatter(
    x = data['date'],
    y = data['upper_limit']-data['lower_limit'],
    stackgroup='one')
])
fig3 = go.Figure(data=fig1.data + fig2.data)
fig3.update_layout(
    title="",
    xaxis_title="",
    yaxis_title="OCR (GMP)",
    legend_title="",
    showlegend=False
)
st.plotly_chart(fig3, use_container_width=True)

#water volume
ind = data.index[data['ocr_val']<initial_level].tolist()
y = data['ocr_val'][ind].values
x = data['minutes'][ind].values
initial_water_volume = np.round_(np.trapz(y,x), decimals = 2)
ind = data.index[data['ocr_val']<lower_limit].tolist()
y = data['ocr_val'][ind].values
x = data['minutes'][ind].values
water_volume = np.round_(np.trapz(y,x), decimals = 2)
delta=water_volume-initial_water_volume
st.metric(label="Water Loss (Gallons)", value=water_volume, delta=delta, delta_color="inverse")