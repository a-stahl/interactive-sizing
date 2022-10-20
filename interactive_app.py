import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Interactive Sizing App', page_icon='wooper.ico')
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
line_rgb = st.text_input('Line rgb and transparency', 'rgba( 0, 204, 255, 0.4)')
bottom_zone_rgb = st.text_input('Bottom zone rgb', 'rgb(255, 51, 51)')
middle_zone_rgb = st.text_input('Middle zone rgb', 'rgb(255, 178, 102)')
top_zone_rgb = st.text_input('Top zone rgb', 'rgb(204, 255, 230)')
#plot
line = go.Scatter(
            x=data['date'], 
            y=data['ocr_val'], 
            mode="lines+markers",
            line=dict(width=5.0, color=line_rgb)
            # line={
            #     "color":"rgba( 102, 204, 255,0.7)"
            # }
            )
fig1 = go.Figure(
    data=line,
    layout=go.Layout(showlegend=False)
)
fig2 = go.Figure(data=[go.Scatter(
    x = data['date'],
    y = data['lower_limit'],
    line=dict(width=0.1, color=bottom_zone_rgb),
    stackgroup='one'),
                       go.Scatter(
    x = data['date'],
    y = data['upper_limit']-data['lower_limit'],
    line=dict(width=0.1, color=middle_zone_rgb),
    stackgroup='one')
])
fig3 = go.Figure(data=fig2.data + fig1.data)
fig3.update_layout(
    title="",
    xaxis_title="",
    yaxis_title="OCR (GPM)",
    legend_title="",
    showlegend=False,
    xaxis_range=[data['date'][0],data['date'][data.index[-1]]],
    plot_bgcolor = top_zone_rgb
)
st.plotly_chart(fig3, use_container_width=True)

#water volume
# ind = data.index[data['ocr_val']<initial_level].tolist()
# y = data['ocr_val'][ind].values
# x = data['minutes'][ind].values
# initial_water_volume = np.round_(np.trapz(y,x), decimals = 2)
# ind = data.index[data['ocr_val']<lower_limit].tolist()
# y = data['ocr_val'][ind].values
# x = data['minutes'][ind].values
# water_volume = np.round_(np.trapz(y,x), decimals = 2)
# delta=water_volume-initial_water_volume
# st.metric(label="Water Loss (Gallons)", value=water_volume, delta=delta, delta_color="inverse")