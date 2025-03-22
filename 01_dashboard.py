from dash import Dash
import dash_bootstrap_components as dbc
from dashboard.data_loader import load_data
from dashboard.layout import create_layout
from dashboard.callbacks import register_callbacks

# Daten laden
data = load_data()

# Anwendung initialisieren
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="CPA-Dashboard")
app.layout = create_layout(data)

# Callbacks registrieren
register_callbacks(app, data)

if __name__ == '__main__':
    app.run_server(debug=True)
