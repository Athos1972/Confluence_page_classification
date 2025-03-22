from dash import dcc, html, dash_table


def create_layout(data):
    return html.Div([
        html.H1("Confluence Page Analysis Dashboard", style={'textAlign': 'center',
                                                             'marginTop': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Label('Filter creators', style={'margin': '10px 0 5px 0', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='creator-dropdown',
                options=[{'label': name, 'value': name} for name in sorted(data['created_by'].unique())],
                multi=True,
                placeholder="Type any name to filter"
            ),

            html.Br(),  # Abstand zwischen den Zeilen

            html.Label('Start date:', style={'margin': '10px 20px 5px 0', 'fontSize': '16px', 'display': 'inline-block'}),
            dcc.DatePickerSingle(
                id='start-date-picker',
                min_date_allowed=data['created'].min(),
                max_date_allowed=data['created'].max(),
                initial_visible_month=data['created'].min(),
                date=str(data['created'].min().date()),
                display_format='DD-MM-YYYY',
                className='dash-datepicker',
                style={'marginRight': '20px', 'display': 'inline-block'}
            ),

            html.Label('End date:', style={'margin': '10px 20px 5px 0', 'fontSize': '16px', 'display': 'inline-block'}),
            dcc.DatePickerSingle(
                id='end-date-picker',
                min_date_allowed=data['created'].min(),
                max_date_allowed=data['created'].max(),
                initial_visible_month=data['created'].max(),
                date=str(data['created'].max().date()),
                display_format='DD-MM-YYYY',
                className='dash-datepicker',
                style={'display': 'inline-block'}
            ),

            html.P(" "),  # Abstand zwischen den Zeilen

            html.Label('Document length filter', style={'margin': '10px 0 5px 0', 'fontSize': '16px'}),
            dcc.RangeSlider(
                id='pure-length-slider',
                min=0,
                max=data['pure_length'].max(),
                value=[0, data['pure_length'].max()],
                marks={i: str(i) for i in range(0, int(data['pure_length'].max()) + 1, 1000)},
                step=100
            )
        ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px',
                  'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'}),

        html.Hr(style={'border': '1px solid #d3d3d3', 'margin': '20px 0'}),  # optische Trennung

        html.Div(id='page-count', style={'margin': '20px 0', 'fontSize': '16px', 'textAlign': 'center',
                                         'backgroundColor': '#f1f1f1', 'padding': '10px', 'borderRadius': '5px'}),

        html.Hr(style={'border': '1px solid #d3d3d3', 'margin': '20px 0'}),  # optische Trennung

        # Flexbox-Container für die Grafiken
        html.Div(id='graphs-container', style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),

        html.Hr(style={'border': '1px solid #d3d3d3', 'margin': '20px 0'}),  # optische Trennung

        html.Div([
            html.Label('Number of records to display:', style={'marginRight': '10px', 'fontSize': '16px'}),
            dcc.Input(id='record-count', type='number', value=100, min=1, style={'fontSize': '16px'})
        ], style={'marginBottom': '20px'}),

        dash_table.DataTable(
            id='details-table',
            columns=[
                {'name': 'Title/Url', 'id': 'title', 'presentation': 'markdown'},
                {'name': 'Creator', 'id': 'created_by'},
                {'name': 'Versions', 'id': 'version_count'},
                {'name': 'Links-In', 'id': 'incoming_links_count'},
                {'name': 'Viewers', 'id': 'overall_page_viewers'},
                {'name': 'Views', 'id': 'overall_page_views'},
                {'name': 'QP', 'id': 'quality_points_total'},
                {'name': 'Length', 'id': 'pure_length'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'Arial, sans-serif', 'fontSize': '13px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold',
                          'fontFamily': 'Arial, sans-serif', 'fontSize': '12px'},
            page_size=100,
            filter_action='native',
            sort_action='native'
        ),

        html.Button("Download Current Selection", id="download-button", style={'marginTop': '20px',
                                                                               'marginBottom': '20px',
                                                                               'alignSelf': 'center', }),
        dcc.Download(id="download-dataframe-xls")
    ], style={'padding': '20px', 'backgroundColor': '#f5f5f5'})
