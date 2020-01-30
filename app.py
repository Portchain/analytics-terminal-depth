import datetime, os, sys
import logging

import streamlit as st
import numpy as np
from portcall.data import TerminalTable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from port_mapper.depth_map import create_draft_histogram_map, calculate_quay_aligned_map, DraftContainer
from port_mapper.depth_profile import convert_curve_to_sections, clean_depth_profile, calculate_depth_profile
from port_mapper.utils import import_ais_data
from portcall.terminal import Terminal
from portcall.vessel import VesselTrack

FIGURE_WIDTH = 1000


@st.cache
def load_terminal(terminal: str):
    with TerminalTable() as tdb:
        terminals = tdb.fetch_terminals(code=terminal)
        terminal = terminals[0]
    return terminal


@st.cache
def load_tracks(terminal: str, ais_start_time: datetime, ais_end_time: datetime):
    tracks = import_ais_data(terminal, ais_start_time, ais_end_time)
    return tracks


@st.cache(hash_funcs={np.ufunc:str})
def calculate_drafts(tracks, terminal, resolution, map_buffer, map_type):
    return create_draft_histogram_map(tracks,
                                      terminal,
                                      resolution=resolution,
                                      terminal_buffer=map_buffer,
                                      type=map_type)


@st.cache
def load_all_terminal_names():
    with TerminalTable() as tdb:
        df = tdb.fetch_terminal_data()
        df = df.dropna(subset=['outline'])
    return sorted(df['code'].values.tolist())


if __name__ == "__main__":

    # password_str = st.sidebar.text_input('App Password')
    # if password_str == os.environ['APP_PASSWORD']:
    #     st.sidebar.text('App unlocked!')
    # else:
    #     st.sidebar.text('App locked.')
    #     sys.exit()

    # ---------- Collect histogram of drafts ---------------------------------------------------------------------------
    st.sidebar.header('Data')
    all_terminal_names = load_all_terminal_names()
    terminal_name = st.sidebar.selectbox('Select a terminal', [''] + all_terminal_names, index=0)

    if not terminal_name == '':
        ais_start_time = st.sidebar.date_input('Start data', datetime.date(2019, 11, 1))
        ais_end_time = st.sidebar.date_input('Start data', datetime.date(2019, 12, 1))

        terminal = load_terminal(terminal_name)
        st.title('Depth analysis of  %s' % terminal_name)
        tracks = load_tracks(terminal, ais_start_time, ais_end_time)

        # --------- Calculate the map ---------------------------------------------------------------------------------
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
        draft_agg = st.sidebar.selectbox('Select a map type',
                                         ['max', '95%', '90%', '75%'],
                                         index=1)

        if map_type == 'depth':
            if draft_agg == 'max':
                agg = lambda x: np.nanmax(x, axis=0)
            elif draft_agg == '95%':
                agg = lambda x: np.nanquantile(x, 0.95, axis=0)
            elif draft_agg == '90%':
                agg = lambda x: np.nanquantile(x, 0.90, axis=0)
            elif draft_agg == '75%':
                agg = lambda x: np.nanquantile(x, 0.75, axis=0)
            else:
                raise ValueError('Unknown value requestes: %s' % draft_agg)
            value_str = 'Depth [m]'
            agg_map = drafts.create_depth_map(agg=agg)
            zmin = 5
            zmax = 20
        elif map_type == 'count':
            agg_map = drafts.create_count_map()
            value_str = 'Vessel Count [#]'
            zmin, zmax = None, None
        else:
            raise ValueError('Unknown map type requested: %s' % map_type)

        # ---------- Calculate the quay profile ---------------------------------------------------------------------
        st.sidebar.header('Quay profile')
        quay_codes = [quay.code for quay in terminal.quays]
        quay_code = st.sidebar.selectbox('Select a quay to analyse', quay_codes)
        quay_idx = quay_codes.index(quay_code)
        depth_unit = st.sidebar.number_input('Unit', min_value=0.2, max_value=2., value=1., step=0.2)
        smooth_width = st.sidebar.number_input('Smoothing', min_value=1, max_value=10, value=3, step=1)
        merge_width = st.sidebar.number_input('Merging', min_value=0, max_value=10, value=2, step=1)
        plateau_width = st.sidebar.number_input('Plateau', min_value=0, max_value=10, value=2, step=1)

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

        # --------------- Present results in figures ------------------------------------------------------------------
        fig1 = make_subplots(rows=1,
                             cols=2,
                             specs=[[{"type": "scattermapbox"}, {"type": "scatter"}]],
                             shared_yaxes=True,
                             shared_xaxes=True)

        fig1.add_trace(go.Scattermapbox(mode='lines',
                                        lon=terminal.outline[:, 0],
                                        lat=terminal.outline[:, 1],
                                        line=dict(color='blue'),
                                        name=terminal_name,
                                        text=terminal_name,
                                        showlegend=False,
                                        hoverinfo='text'),
                       row=1, col=1)

        for quay in terminal.quays:
            fig1.add_trace(go.Scattermapbox(mode='lines',
                                            lon=quay.line[:, 0],
                                            lat=quay.line[:, 1],
                                            line=dict(color='red'),
                                            name=quay.code,
                                            text=quay.code,
                                            showlegend=False,
                                            hoverinfo='text'),
                           row=1, col=1)
        fig1.update_traces(textposition='top center')
        center = terminal.terminal_center
        fig1.update_layout(
            mapbox=dict(center=dict(lon=center[0], lat=center[1]),
                        style="open-street-map",
                        zoom=12),
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0})

        fig1.add_trace(go.Scatter(x=terminal.outline[:, 0],
                                  y=terminal.outline[:, 1],
                                  mode='lines',
                                  line=dict(color='black'),
                                  hoveron='fills',
                                  name=terminal_name,
                                  hoverinfo='name'))

        fig1.add_trace(go.Contour(z=agg_map.get_data(),
                                  x=agg_map.get_first_axis(),  # horizontal axis
                                  y=agg_map.get_second_axis(),  # vertical axis
                                  zmin=zmin, zmax=zmax,
                                  colorscale='Blues',
                                  colorbar=dict(lenmode='fraction',
                                                len=0.75,
                                                yanchor='bottom',
                                                y=0,
                                                title=dict(text=value_str)),
                                  line=dict(width=0)),
                       row=1, col=2)

        for quay in terminal.quays:
            fig1.add_trace(go.Scatter(x=quay.line[:, 0],
                                      y=quay.line[:, 1],
                                      mode='lines',
                                      name=quay.code,
                                      hoveron='fills',
                                      hoverinfo='x+y+name'))

        fig1.update_xaxes(showticklabels=False)
        fig1.update_yaxes(showticklabels=False)
        fig1.update_layout(width=FIGURE_WIDTH, height=FIGURE_WIDTH / 2)
        st.write(fig1)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=quay_position, y=quay_depth,
                                  mode='lines',
                                  name='raw'))
        fig2.add_trace(go.Scatter(x=quay_position, y=quay_depth_cleaned,
                                  mode='lines+markers',
                                  name='clean'))
        fig2.update_yaxes(title=value_str)
        fig2.update_xaxes(title='Quay Position [m]')
        fig2.update_layout(width=FIGURE_WIDTH, height=FIGURE_WIDTH / 2)
        st.write(fig2)

        st.header('Quay profile')
        df = {'start_position': depth_profile[0],
              'depth': depth_profile[1],
              'length': depth_profile[2]}
        st.dataframe(df)
