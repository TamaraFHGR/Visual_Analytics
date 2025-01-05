@app.callback(
    Output('accidents_stats', 'figure'),
    [Input('region_dropdown', 'value'),
     Input('risk_group_checklist', 'value')]
)
def accidents_stats(selected_parameter, selected_risk_groups):
    filtered_df = kmeans_df.copy()

    if selected_parameter == 'Risks Group and Influencing Factors':
        if selected_risk_groups:
            filtered_df = filtered_df[filtered_df['risk_group'].isin(selected_risk_groups)]

        # Achsen und Labels
        parameters = ['air_temp_mean_stations', 'wind_speed_max_stations', 'snow_height_mean_stations', 'new_snow_mean_stations']
        labels = ['Air Temperature (°C)', 'Wind Speed (m/s)', 'Snow Height (cm)', 'New Snow (cm)']
        angles = [0, 90, 180, 270]  # Für die vier Parameter

        # Normierte Werte für jeden Parameter vorbereiten
        normalized_means = {}
        for param in parameters:
            param_min = filtered_df[param].min()
            param_max = filtered_df[param].max()
            normalized_means[param] = filtered_df.groupby('k_cluster')[param].mean().apply(
                lambda x: (x - param_min) / (param_max - param_min) if param_max > param_min else 0
            )

        fig = go.Figure()

        # Iteriere durch jeden Parameter und füge Balken pro k_cluster hinzu
        for param, label, angle in zip(parameters, labels, angles):
            cluster_means = normalized_means[param]  # Normierte Mittelwerte pro k_cluster

            for cluster, norm_mean_value in cluster_means.items():
                fig.add_trace(go.Barpolar(
                    r=[norm_mean_value],  # Normierter Wert
                    theta=[angle],       # Der Winkel für den Parameter
                    name=f"{label} - Cluster {cluster}",
                    marker_color=color_map[cluster],  # Farbe aus der color_map
                    width=50,
                    opacity=0.7
                ))

        # Layout anpassen
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    showline=True,  # Zeige die Radialachse
                    ticks='',       # Entferne Ticks
                    gridcolor='lightgray',
                    range=[0, 1],   # Normierte Werte auf [0, 1] begrenzen
                ),
                angularaxis=dict(
                    tickvals=angles,
                    ticktext=labels,
                    rotation=90  # Optional: Startwinkel anpassen
                )
            ),
            margin=dict(l=0, r=0, t=24, b=24),
            template='plotly_white',
            showlegend=False,
            height=300
        )
        return fig

    elif selected_parameter == 'Number of Accidents by Month':
        filtered_acc_df = acc_df

        # Convert dates to datetime and add year and month columns:
        filtered_acc_df['date'] = pd.to_datetime(filtered_acc_df['date'])
        filtered_acc_df['year'] = filtered_acc_df['date'].dt.year
        filtered_acc_df['month'] = filtered_acc_df['date'].dt.month

        # Define winter months and corresponding names:
        winter_months = [11, 12, 1, 2, 3, 4]
        month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February', 3: 'March', 4: 'April'}

        # Create subplot structure for two small multiples:
        fig = make_subplots(rows=3, cols=3, subplot_titles=[month_names[m] for m in winter_months],
                            vertical_spacing=0.05, horizontal_spacing=0.05)

        # Set the font size for subplot titles specifically
        for i, month in enumerate(winter_months):
            fig['layout']['annotations'][i].update(font=dict(size=8))

        # Plot accident counts as bar charts (default):
        for idx, month in enumerate(winter_months):
            monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]
            yearly_counts = monthly_data.groupby('year').size()

            # Calculate row and column for the subplot
            row = (idx // 3) + 1
            col = (idx % 3) + 1

            # Add bar chart to the respective subplot
            fig.add_trace(
                go.Bar(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    name=month_names[month],
                    marker_color='darkblue'
                ),
                row=row, col=col
            )
            # Set y-axis range for the subplot
            fig.update_yaxes(range=[0, 80], row=row, col=col)

        return fig


