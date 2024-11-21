from dash import Dash, dcc, html
import plotly.graph_objects as go
from Data_Loader import load_hist_clusters

app = Dash(__name__)

cluster_df = load_hist_clusters()


def scatterplot_wind_snow(cluster_df):
    df = cluster_df[cluster_df['alpine_region'] == 'Eastern Alps']
    if df.empty:
        raise ValueError("No data available for the selected alpine region.")

    fig = go.Figure()
    for cluster_id in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_id]
        fig.add_trace(go.Scatter(
            x=cluster_data['new_snow_mean_stations'],
            y=cluster_data['wind_speed_max_stations'],
            mode='markers',
            marker=dict(size=10),
            name=f'Cluster {cluster_id}',  # Cluster-Name für die Legende
            text=cluster_data['cluster']  # Tooltip-Inhalt
        ))
    fig.update_layout(
        xaxis_title='New Snow (cm)',
        yaxis_title='Max Wind Speed (m/s)',
        legend_title='Cluster',
        template='plotly_white'
    )
    return fig


# Erstelle das Plot-Objekt
fig = scatterplot_wind_snow(cluster_df)

# Layout der App
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

# Starte die App
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True)
