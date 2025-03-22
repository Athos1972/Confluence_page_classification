from dash import dcc, html, Output, Input, State
import plotly.express as px
import pandas as pd
import io


def register_callbacks(app, data):
    def update_slider(creators, start_date, end_date):
        print("update_slider Callback aufgerufen")
        print(f"Creators: {creators}, Start Date: {start_date}, End Date: {end_date}")

        filtered_data = data.copy()  # Make a copy of the data to avoid SettingWithCopyWarning
        if creators:
            filtered_data = filtered_data[filtered_data['created_by'].isin(creators)]
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['created'] >= pd.to_datetime(start_date)) &
                                          (filtered_data['created'] <= pd.to_datetime(end_date))]

        if filtered_data.empty:
            return 0, 0, [0, 0], {}

        min_length = filtered_data['pure_length'].min()
        max_length = filtered_data['pure_length'].max()
        marks = {i: str(i) for i in range(0, int(max_length) + 1, int(max_length / 10))}

        return min_length, max_length, [min_length, max_length], marks

    def create_pie_chart(filtered_data):
        top_creators = filtered_data['created_by'].value_counts().nlargest(10)
        pie_fig = px.pie(names=top_creators.index, values=top_creators.values,
                         title='Top 10 Creators by Number of Pages')
        pie_fig.update_layout({'plot_bgcolor': '#e0e0e0', 'paper_bgcolor': '#e0e0e0'})
        return pie_fig

    def create_line_chart(filtered_data):
        filtered_data['month_year'] = filtered_data['created'].dt.to_period('M')
        monthly_counts = filtered_data.groupby('month_year').size().reset_index(name='count')
        monthly_counts['month_year'] = monthly_counts['month_year'].astype(str)

        line_fig = px.line(monthly_counts, x='month_year', y='count', title='Pages Created Over Time', markers=True)
        line_fig.update_layout(
            xaxis_title='Month and Year',
            yaxis_title='Number of Pages',
            xaxis_tickformat='%b %Y',
            plot_bgcolor='#d0d0d0',
            paper_bgcolor='#d0d0d0'
        )

        return line_fig

    def create_network_graph(filtered_data):
        network_fig = px.scatter(filtered_data, x='pure_length', y='incoming_links_count', color='created_by',
                                 size='pure_length', title='Network of Page Connections')
        network_fig.update_layout({'plot_bgcolor': '#e0e0e0', 'paper_bgcolor': '#e0e0e0'})
        return network_fig

    def create_simple_distribution_chart(filtered_data):
        filtered_data = filtered_data[
            filtered_data['pure_length'] > 0].copy()  # Make a copy to avoid SettingWithCopyWarning
        if filtered_data.empty:
            print("Keine Daten nach Bereinigung verfügbar.")
            empty_fig = {
                "layout": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [
                        {
                            "text": "Keine Daten nach Bereinigung verfügbar.",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 28
                            }
                        }
                    ]
                }
            }
            return empty_fig

        # Gruppiere die Daten in 5000er-Schritten und benenne die Gruppen
        filtered_data['length_group'] = ((filtered_data['pure_length'] // 5000) * 5000).astype(int)
        filtered_data['length_group'] = filtered_data['length_group'].apply(lambda x: f"{x + 1}-{x + 5000}")

        length_distribution = filtered_data.groupby('length_group').size().reset_index(name='count')

        # Sortiere die length_groups nach ihrem numerischen Anfangswert
        length_distribution['length_group_start'] = length_distribution['length_group'].apply(
            lambda x: int(x.split('-')[0]))
        length_distribution = length_distribution.sort_values(by='length_group_start').drop(
            columns=['length_group_start'])

        # Erstelle das Liniendiagramm
        dist_fig = px.line(length_distribution, x='length_group', y='count', title='Distribution of Page Lengths',
                           markers=True)
        dist_fig.update_layout(
            xaxis_title='Page Length (Grouped by 5000s)',
            yaxis_title='Number of Pages',
            plot_bgcolor='#d0d0d0',
            paper_bgcolor='#d0d0d0'
        )

        return dist_fig

    @app.callback(
        [Output('pure-length-slider', 'min'),
         Output('pure-length-slider', 'max'),
         Output('pure-length-slider', 'value'),
         Output('pure-length-slider', 'marks')],
        [Input('creator-dropdown', 'value'),
         Input('start-date-picker', 'date'),
         Input('end-date-picker', 'date')]
    )
    def slider_callback(creators, start_date, end_date):
        return update_slider(creators, start_date, end_date)

    @app.callback(
        [Output('graphs-container', 'children'),
         Output('page-count', 'children'),
         Output('details-table', 'data'),
         Output('details-table', 'page_size')],
        [Input('creator-dropdown', 'value'),
         Input('start-date-picker', 'date'),
         Input('end-date-picker', 'date'),
         Input('pure-length-slider', 'value'),
         Input('record-count', 'value')]
    )
    def update_graphs(creators, start_date, end_date, pure_length_range, record_count):
        print("update_graphs Callback aufgerufen")
        print(
            f"Creators: {creators}, Start Date: {start_date}, End Date: {end_date}, Pure Length Range: {pure_length_range}")

        if start_date and end_date and pd.to_datetime(end_date) < pd.to_datetime(start_date):
            print("Fehler: Enddatum liegt vor dem Startdatum.")
            error_message = "Enddatum liegt vor dem Startdatum. Keine Anzeige möglich."
            empty_fig = {
                "layout": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [
                        {
                            "text": error_message,
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 28
                            }
                        }
                    ]
                }
            }
            return [dcc.Graph(figure=empty_fig)], "0 pages of {total} selected".format(total=len(data)), [], 100

        filtered_data = data.copy()  # Make a copy to avoid SettingWithCopyWarning
        if creators:
            filtered_data = filtered_data[filtered_data['created_by'].isin(creators)]
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['created'] >= pd.to_datetime(start_date)) &
                                          (filtered_data['created'] <= pd.to_datetime(end_date))]
        if pure_length_range:
            filtered_data = filtered_data[(filtered_data['pure_length'] >= pure_length_range[0]) &
                                          (filtered_data['pure_length'] <= pure_length_range[1])]

        print(f"Gefilterte Daten: {filtered_data.shape}")

        if filtered_data.empty:
            print("Keine Daten nach Filterung verfügbar.")
            empty_fig = {
                "layout": {
                    "xaxis": {"visible": False},
                    "yaxis": {"visible": False},
                    "annotations": [
                        {
                            "text": "Keine Daten verfügbar.",
                            "xref": "paper",
                            "yref": "paper",
                            "showarrow": False,
                            "font": {
                                "size": 28
                            }
                        }
                    ]
                }
            }
            return [dcc.Graph(figure=empty_fig)], "0 pages of {total} selected".format(total=len(data)), [], 100

        graphs = []
        colors = ['#f0f0f0', '#e8e8e8']
        index = 0

        if not creators or (creators and len(creators) > 1):
            fig = create_pie_chart(filtered_data)
            fig.update_layout({'plot_bgcolor': colors[index % 2], 'paper_bgcolor': colors[index % 2]})
            graphs.append(html.Div(dcc.Graph(figure=fig),
                                   style={'backgroundColor': colors[index % 2], 'padding': '10px',
                                          'borderRadius': '5px', 'marginBottom': '10px'}))
            index += 1

        fig = create_line_chart(filtered_data)
        fig.update_layout({'plot_bgcolor': colors[index % 2], 'paper_bgcolor': colors[index % 2]})
        graphs.append(html.Div(dcc.Graph(figure=fig),
                               style={'backgroundColor': colors[index % 2], 'padding': '10px', 'borderRadius': '5px',
                                      'marginBottom': '10px'}))
        index += 1

        fig = create_network_graph(filtered_data)
        fig.update_layout({'plot_bgcolor': colors[index % 2], 'paper_bgcolor': colors[index % 2]})
        graphs.append(html.Div(dcc.Graph(figure=fig),
                               style={'backgroundColor': colors[index % 2], 'padding': '10px', 'borderRadius': '5px',
                                      'marginBottom': '10px'}))
        index += 1

        fig = create_simple_distribution_chart(filtered_data)
        fig.update_layout({'plot_bgcolor': colors[index % 2], 'paper_bgcolor': colors[index % 2]})
        graphs.append(html.Div(dcc.Graph(figure=fig),
                               style={'backgroundColor': colors[index % 2], 'padding': '10px', 'borderRadius': '5px'}))

        page_count = "{filtered} pages of {total} selected with length {min_length} to {max_length}".format(
            filtered=len(filtered_data), total=len(data), min_length=pure_length_range[0],
            max_length=pure_length_range[1]
        )

        # Bereite die Daten für das Grid vor
        filtered_data = filtered_data.head(record_count)
        filtered_data['quality_points_total'] = filtered_data['quality_points_total'].round(0).astype(int)
        table_data = filtered_data[
            ['title', 'created_by', 'version_count', 'incoming_links_count', 'overall_page_viewers',
             'overall_page_views', 'quality_points_total', 'pure_length', 'url']].to_dict('records')

        # Formatieren der Links in der ersten Spalte
        for row in table_data:
            row['title'] = f"[{row['title'][0:50]}]({row['url']})"

        # Remove URL as we don't need it anymore
        table_data = [{k: v for k, v in row.items() if k != 'url'} for row in table_data]

        return graphs, page_count, table_data, record_count

    @app.callback(
        Output("download-dataframe-xls", "data"),
        Input("download-button", "n_clicks"),
        State('creator-dropdown', 'value'),
        State('start-date-picker', 'date'),
        State('end-date-picker', 'date'),
        State('pure-length-slider', 'value'),
        prevent_initial_call=True,
    )
    def download_filtered_data(n_clicks, creators, start_date, end_date, pure_length_range):
        filtered_data = data.copy()
        if creators:
            filtered_data = filtered_data[filtered_data['created_by'].isin(creators)]
        if start_date and end_date:
            filtered_data = filtered_data[(filtered_data['created'] >= pd.to_datetime(start_date)) &
                                          (filtered_data['created'] <= pd.to_datetime(end_date))]
        if pure_length_range:
            filtered_data = filtered_data[(filtered_data['pure_length'] >= pure_length_range[0]) &
                                          (filtered_data['pure_length'] <= pure_length_range[1])]

        if filtered_data.empty:
            return None

        # Create a BytesIO buffer to save the Excel file
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            filtered_data.to_excel(writer, index=False, sheet_name='Filtered Data')

        buffer.seek(0)
        return dcc.send_bytes(buffer.read(), "filtered_data.xlsx")
