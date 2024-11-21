from dash import Dash, dcc, html
import plotly.express as px
import numpy as np

app = Dash(__name__)

# Erstelle eine zufällige Matrix als Beispiel
data = np.random.rand(100, 100)  # 100x100 Pixel

# Visualisiere die Daten als Pixelbild
fig = px.imshow(data, color_continuous_scale='Viridis', aspect='equal', origin='lower')

# Passe die Anzeigeparameter an
fig.update_xaxes(showticklabels=False)  # Achsenbeschriftungen entfernen
fig.update_yaxes(showticklabels=False)  # Achsenbeschriftungen entfernen
fig.update_layout(coloraxis_showscale=True)  # Farbskala anzeigen

    # Layout der App
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

# Starte die App mit einem benutzerdefinierten Host und Port
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True)  # Ändere Host und Port hier
