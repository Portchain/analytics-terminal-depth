import datetime

import streamlit as st
import numpy as np
from portcall.data import TerminalTable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from port_mapper.depth_map import create_draft_histogram_map, calculate_quay_aligned_map
from port_mapper.depth_profile import convert_curve_to_sections, clean_depth_profile, calculate_depth_profile
from port_mapper.utils import import_ais_data


@st.cache
def load_terminal(terminal: str):
    with TerminalTable() as tdb:
        terminals = tdb.fetch_terminals(code=terminal)
        terminal = terminals[0]
    return terminal


@st.cache
def load_tracks(terminal, ais_start_time, ais_end_time):
    tracks = import_ais_data(terminal, ais_start_time, ais_end_time)
    return tracks


@st.cache(allow_output_mutation=True)
def calculate_drafts(tracks, terminal, resolution, map_buffer, map_type):
    return create_draft_histogram_map(tracks,
                                      terminal,
                                      resolution=resolution,
                                      terminal_buffer=map_buffer,
                                      type=map_type)


st.sidebar.header('Data')
terminal_name = st.sidebar.selectbox('Select a terminal', ['DEHAM-BUR',
                                                           'DEHAM-CTA',
                                                           'MAPTM-EUROGATE',
                                                           'COCTG-CONTECAR'])

ais_start_time = st.sidebar.date_input('Start data', datetime.date(2019, 11, 1))
ais_end_time = st.sidebar.date_input('Start data', datetime.date(2019, 12, 1))

terminal = load_terminal(terminal_name)
st.title('Analysing %s' % terminal_name)
tracks = load_tracks(terminal, ais_start_time, ais_end_time)

st.sidebar.header('Map')
draft_type = st.sidebar.selectbox(
    'Select a map type',
    ['static', 'dynamic'])

resolution = st.sidebar.number_input('Map resolution [m]', min_value=2, max_value=50, value=10)

drafts = calculate_drafts(tracks,
                          terminal,
                          resolution, 100, draft_type)

map_type = st.sidebar.selectbox('Select a map type',
                                ['depth', 'count'])

if map_type == 'depth':
    agg = lambda x: np.nanquantile(x, 0.95, axis=0)
    agg_map = drafts.create_depth_map(agg=agg)
elif map_type == 'count':
    agg_map = drafts.create_count_map()
else:
    raise ValueError('Unknown map type requested: %s' % map_type)

st.sidebar.header('Quay profile')
quay_codes = [quay.code for quay in terminal.quays]
quay_code = st.sidebar.selectbox('Select a quay to analyse', quay_codes)
quay_idx = quay_codes.index(quay_code)
depth_unit = st.sidebar.number_input('Unit', min_value=0.2, max_value=2., value=1., step=0.2)
smooth_width = st.sidebar.number_input('Smoothing', min_value=1, max_value=10, value=3, step=1)
merge_width = st.sidebar.number_input('Merging', min_value=0, max_value=10, value=1, step=1)
plateau_width = st.sidebar.number_input('Plateau', min_value=0, max_value=10, value=1, step=1)

quay = terminal.quays[quay_idx]

quay_map = calculate_quay_aligned_map(quay, agg_map)
quay_position, quay_depth = calculate_depth_profile(quay_map)
quay_depth_cleaned = clean_depth_profile(quay_position, quay_depth,
                                         depth_unit=depth_unit,
                                         smooth_width=smooth_width,
                                         merge_width=merge_width,
                                         plateau_width=plateau_width
                                         )
depth_profile = convert_curve_to_sections(quay_position, quay_depth_cleaned, threshold=0.1)

fig1 = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scattermapbox"}, {"type": "scatter"}]],
    shared_yaxes=True,
    shared_xaxes=True)

fig1.add_trace(go.Scattermapbox(), row=1, col=1)
center = terminal.terminal_center
fig1.update_layout(
    mapbox={
        'center': {'lon': center[0], 'lat': center[1]},
        'style': "open-street-map",
        'zoom': 12,
        'layers': [{
            'source': terminal.geojson(),
            'type': "fill",
            'below': "traces", 'color': "blue", 'opacity': 0.5, 'name': terminal.code}]
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0})

fig1.add_trace(go.Scatter(x=terminal.outline[:, 0],
                          y=terminal.outline[:, 1],
                          fill='toself',
                          opacity=0.5,
                          hoveron='fills',
                          name=terminal_name,
                          hoverinfo='name'))

fig1.add_trace(go.Contour(z=agg_map.get_data(),
                          x=agg_map.get_first_axis(),  # horizontal axis
                          y=agg_map.get_second_axis(),  # vertical axis
                          colorscale='Blues',
                          colorbar=dict(lenmode='fraction', len=0.75,yanchor='bottom',y=0)),
               row=1, col=2)

for quay in terminal.quays:
    fig1.add_trace(go.Scatter(x=quay.line[:,0],
                              y=quay.line[:,1],
                              mode='lines',
                              name=quay.code,
                              hoveron='fills',
                              hoverinfo='x+y+name'))


fig1.update_xaxes(showticklabels=False)
fig1.update_yaxes(showticklabels=False)
fig1.update_layout()
st.write(fig1)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=quay_position, y=quay_depth,
                          mode='lines',
                          name='raw'),
               )
fig2.add_trace(go.Scatter(x=quay_position, y=quay_depth_cleaned,
                          mode='lines+markers',
                          name='clean'),
               )

st.write(fig2)

st.header('Quay profile')
df = {'start_position': depth_profile[0],
      'depth': depth_profile[1],
      'length': depth_profile[2]}
st.dataframe(df)
