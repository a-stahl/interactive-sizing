import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
from PIL import Image

st.set_page_config(page_title='Interactive Sizing App', layout='wide',page_icon='wooper.ico')
col1, col2, col3 = st.columns([1, 1, 1])
image = Image.open('Olea_logo.png') 
new_image = image.resize((140, 100))
with col1:
    st.image(new_image)
with col2:
    st.markdown(""" <style> .font {
    font-size:40px ; font-family: 'Open Sans'; color: black;text-align: center;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"><strong>Sizing Error<strong></p>', unsafe_allow_html=True)
    st.markdown(""" <style> .font2 {
    font-size:20px ; font-family: 'Open Sans'; color: black;text-align: center;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">Meter size Affect on Accuracy</p>', unsafe_allow_html=True)
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 3rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

line_rgb = 'rgb(2, 135, 202)'
bottom_zone_rgb = 'rgb(242, 90, 90)'
middle_zone_rgb = 'rgb(255, 191, 128)'
top_zone_rgb = 'rgb(185, 223, 209)'
# Function to calculate time duration
def interpolated_intercepts(x, y1, y2):

    def intercept(point1, point2, point3, point4):

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idxs = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    xcs = []
    ycs = []

    for idx in idxs:
        xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
        xcs.append(xc)
        ycs.append(yc)
    return np.array(xcs), np.array(ycs)


initial_level = 8

col1, col2 = st.columns([2, 1])
with col1:
    plot_spot = st.empty()
with col2:
    video_spot = st.empty()    


metrics_spot = st.empty()


# input widgets
col1, col2 = st.columns([2, 1])

with col1:
    hour_to_filter = st.slider('Diameter (Inch)', 1, 8, initial_level)

with col2:
    meter_type = st.radio(
    "Sample meter",
    ('Meter #1', 'Meter #2', 'Meter #3'))

# input dependent calculations
if meter_type == 'Meter #1':
    data = pd.read_csv('plot_data.csv', header=None)
    data.columns = ['ocr_val', 'hours']
    data.ocr_val = data.ocr_val.values + 3.0
if meter_type == 'Meter #2':
    data = pd.read_csv('plot_data.csv', header=None)
    data.columns = ['ocr_val', 'hours']
    data.ocr_val = 2*data['ocr_val'].mean() - data.ocr_val.values + 3.2
if meter_type == 'Meter #3':
    data = pd.read_csv('plot_data.csv', header=None)
    data.columns = ['ocr_val', 'hours']
    data.ocr_val = data.ocr_val.values[::-1] + 7

data['hours'] -= data['hours'][0]
data['date'] = pd.to_datetime(27,errors='ignore', unit='d',origin='2022-08')
data['date'] = data['date'] + pd.to_timedelta(data['hours'], unit='h')
lower_limit = hour_to_filter
upper_limit = 6 + 1.5*lower_limit
limit = max(upper_limit + 4.5,data['ocr_val'].max() + 1.0)

data['lower_limit'] = lower_limit
data['upper_limit'] = upper_limit
data['limit'] = limit
data['minutes'] = data['hours']*60
#plot
line = go.Scatter(
            x=data['date'], 
            y=data['ocr_val'], 
            mode="lines+markers",
            line=dict(width=5.0, color=line_rgb),
            name="OCR"
            )
fig1 = go.Figure(
    data=line,
    layout=go.Layout(showlegend=False)
)
fig2 = go.Figure(data=[go.Scatter(
    x = data['date'],
    y = data['lower_limit'],
    line=dict(width=0.1, color=bottom_zone_rgb),
    name="<95% Accuracy",
    stackgroup='one'),
                    go.Scatter(
    x = data['date'],
    y = data['upper_limit']-data['lower_limit'],
    line=dict(width=0.1, color=middle_zone_rgb),
    name="<98.5% Accuracy",
    stackgroup='one'),
                    go.Scatter(
    x = data['date'],
    y = data['limit']-data['upper_limit'],
    line=dict(width=0.1, color=top_zone_rgb),
    showlegend=False,
    stackgroup='one')
])
fig3 = go.Figure(data=fig2.data + fig1.data)
fig3.update_layout(
    title="",
    xaxis_title="",
    yaxis_title="OCR (GPM)",
    legend_title="",
    showlegend=True,
    xaxis_range=[data['date'][0],data['date'][data.index[-1]]],
    yaxis_range=[0,limit],
    width = 800,
    height = 300,
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
)
    #plot_bgcolor = 'rgb(242, 242, 242)'
)
with plot_spot:
    st.plotly_chart(fig3)

video_logic_df = pd.read_csv('video_logic.csv')
video_no = video_logic_df[meter_type][hour_to_filter-1]
video_file = 'sizing' + str(video_no) + '.mp4'
with video_spot:
    with open(video_file, "rb") as f:

        video_content = f.read()

        video_str = f"data:video/mp4;base64,{base64.b64encode(video_content).decode()}"
        st.markdown(f"""
            <video controls width="500" autoplay="true" muted="true" loop="true">
                <source src="{video_str}" type="video/mp4">
            </video>
        """, unsafe_allow_html=True)
        
# calculate time_duration_red and time_duration_orange

xcs, ycs = interpolated_intercepts(data['minutes'].to_numpy(),data['ocr_val'].to_numpy(),data['lower_limit'].to_numpy())
if xcs.size == 0:
    if data['ocr_val'][0] < lower_limit:
        time_duration_red = data['minutes'][data.index[-1]]
    else:
        time_duration_red = 0
else:
    time_diff = np.diff(xcs.flatten())
    initial_sign = np.interp(xcs[0]+0.01, data['minutes'].to_numpy(), data['ocr_val'].to_numpy()) - lower_limit
    if initial_sign>0:
        time_duration_red = time_diff[1::2].sum()
        time_duration_red = time_duration_red + float(xcs[0]-data['minutes'][0])
    else:
        time_duration_red = time_diff[::2].sum()
    final_sign = np.interp(xcs[-1]+0.01, data['minutes'].to_numpy(), data['ocr_val'].to_numpy()) - lower_limit
    if final_sign<0:
        time_duration_red = time_duration_red + float(data['minutes'][data.index[-1]] - xcs[-1])
xcs, ycs = interpolated_intercepts(data['minutes'].to_numpy(),data['ocr_val'].to_numpy(),data['upper_limit'].to_numpy())
if xcs.size == 0:
    if data['ocr_val'][0] < upper_limit:
        time_duration_redorange = data['minutes'][data.index[-1]]
    else:
        time_duration_redorange = 0
else:
    time_diff = np.diff(xcs.flatten())
    initial_sign = np.interp(xcs[0]+0.01, data['minutes'].to_numpy(), data['ocr_val'].to_numpy()) - upper_limit
    if initial_sign>0:
        time_duration_redorange = time_diff[1::2].sum()
        time_duration_redorange = time_duration_redorange + float(xcs[0]-data['minutes'][0])
    else:
        time_duration_redorange = time_diff[::2].sum()
    final_sign = np.interp(xcs[-1]+0.01, data['minutes'].to_numpy(), data['ocr_val'].to_numpy()) - upper_limit
    if final_sign<0:
        time_duration_redorange = time_duration_redorange + float(data['minutes'][data.index[-1]] - xcs[-1])
time_duration_redorange_pct=np.round(time_duration_redorange/data['minutes'][data.index[-1]]*100,2)
time_duration_red_pct=np.round(time_duration_red/data['minutes'][data.index[-1]]*100,2)
with metrics_spot:
    col1, col2, col3, col4 = st.columns([1,2,2,3])
    col2.metric(label="Time duration <98.5% Accuracy", value="{} %".format(time_duration_redorange_pct), delta=None)
    col3.metric(label="Time duration <95% Accuracy", value="{} %".format(time_duration_red_pct), delta=None)

col1, col2, col3, col4 = st.columns([1,4,4,1])
with col4:
    st.write('Not to scale')