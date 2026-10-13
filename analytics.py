from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go


def create_radar_chart(labels: List[str], values: List[float]):
    fig = go.Figure(data=[go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        line=dict(color='#3B82F6', width=2),
        marker=dict(color='#22D3EE', size=6),
    )])
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        margin=dict(l=30, r=30, t=20, b=20),
    )
    return fig


def create_gauge_chart(value: float):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': 'Interview Readiness'},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': '#22C55E'},
            'bgcolor': '#111827',
            'borderwidth': 2,
            'bordercolor': '#3B82F6',
            'steps': [
                {'range': [0, 50], 'color': '#EF4444'},
                {'range': [50, 70], 'color': '#F59E0B'},
                {'range': [70, 100], 'color': '#22C55E'}
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def create_pie_chart(labels: List[str], values: List[float]):
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#3B82F6', '#22D3EE', '#22C55E', '#F59E0B', '#EF4444'])
    )])
    fig.update_layout(
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def create_bar_chart(labels: List[str], values: List[float], title: str):
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=['#3B82F6', '#2563EB', '#22D3EE', '#22C55E', '#F59E0B']
    )])
    fig.update_layout(
        title=title,
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_line_chart(x_values: List[str], y_values: List[float], title: str):
    fig = go.Figure(data=[go.Scatter(x=x_values, y=y_values, mode='lines+markers', line=dict(color='#3B82F6', width=3))])
    fig.update_layout(
        title=title,
        paper_bgcolor='#0B1120',
        plot_bgcolor='#0B1120',
        font=dict(color='white'),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1F2937'),
        margin=dict(l=30, r=30, t=40, b=20),
    )
    return fig
