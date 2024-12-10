def weather_trend(active_button, selected_region, selected_canton, selected_stations, moving_window):
    filtered_data = daily_snow_df.copy()
    # Filter data based on selected region:
    if selected_region and selected_region != 'Entire Alpine Region':
        valid_regions = all_regions.get(selected_region, [])
        filtered_data = filtered_data[filtered_data['canton_code'].isin(valid_regions)]
    # Filter data based on selected canton:
    if selected_canton and selected_canton != 'None':
        filtered_data = filtered_data[filtered_data['canton_code'] == selected_canton]
    # Filter data based on selected stations:
    if selected_stations:
        filtered_data = filtered_data[filtered_data['station_code'].isin(selected_stations)]

    # Function to calculate mean for any given value column:
    def calculate_mean(df, value_column):
        return df.groupby('measure_date')[value_column].mean()

    # Function to plot the trend data (global aggregation, no stations selected):
    def plot_trend(df, value_column, y_axis_label, line_color, avg_mean_color, is_bar=False, with_moving_avg=False):
        mean_data = calculate_mean(df, value_column)

        if with_moving_avg is True:
            moving_avg = mean_data.rolling(window=moving_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=mean_data.index,
                y=mean_data.values,
                mode='lines',
                line=dict(color=line_color, width=2),
                name=y_axis_label))
            fig.add_trace(go.Scatter(
                x=moving_avg.index,
                y=moving_avg.values,
                mode='lines',
                line=dict(dash='dot', color=avg_mean_color, width=1),
                name=f'{moving_window}-Day Moving Average'))
        elif with_moving_avg is False and is_bar is False:
            fig.add_trace(go.Scatter(
                x=mean_data.index,
                y=mean_data.values,
                mode='lines',
                line=dict(color=line_color, width=2),
                name=y_axis_label))

        elif with_moving_avg is False and is_bar is True:
            fig.add_trace(go.Bar(
                x=mean_data.index,
                y=mean_data.values,
                name=y_axis_label)
            )
            fig.update_layout(
                barmode='stack')

    # Function to plot individual station trends
    def plot_station_trends(df, value_column, y_axis_label, color_scale):
        if selected_stations:
            for i, station in enumerate(selected_stations):
                station_data = df[df['station_code'] == station]
                color=color_scale[i % len(color_scale)]

                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data[value_column],
                    mode='lines',
                    name=f"{station}",
                    line=dict(color=color)))

                moving_avg = station_data[value_column].rolling(window=moving_window, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=station_data['measure_date'],
                        y=moving_avg,
                        mode='lines',
                        line=dict(dash='dot', color='black', width=1),
                        name=f"{station} ({moving_window}-day MA)"))
        else:  # Plot aggregated data if no stations are selected
            plot_trend(df, value_column, y_axis_label, line_color='blue', avg_mean_color='black', is_bar=False, with_moving_avg=True)

    fig = go.Figure()

    # Define color scales based on the active button
    color_scales = {
        'temp_button': ['#0D4235','#15BF94','#16745C','#9CBFB6'],  # Green tones (blugrn)
        'wind_button': ['#5F3515', '#FF6C00', '#FFC497', '#9CBFB6'],  # Brown tones (brwnyl)
        'snow_height_button': ['#003741', '#2ADDFF', '#0000AB', '#9CBFB6'],  # Blue tones (Blues)
        'new_snow_button': ['#601860', '#78A8F0', '#A8D8D8', '#9CBFB6'],  # Blue-Violet tones (Dense)
    }

    # Handle the active button conditions
    if active_button == 'default_button':
        plot_trend(filtered_data, 'air_temp_mean_stations', 'Mean Temperature [°C]', 'green', 'black', is_bar = False, with_moving_avg=False)
        plot_trend(filtered_data, 'wind_speed_max_stations', 'Mean Wind Speed [m/s]', 'orange', 'black', is_bar = False, with_moving_avg=False)
        plot_trend(filtered_data, 'snow_height_mean_stations', 'Mean Snow Height [cm]','blue', 'black', is_bar=True,with_moving_avg=False)
        plot_trend(filtered_data, 'new_snow_mean_stations', 'Mean New Snow [cm]', 'cyan', 'black', is_bar=True, with_moving_avg=False)

    elif active_button == 'temp_button':
        plot_station_trends(filtered_data, 'air_temp_mean_stations', "Air Temperature [°C]", color_scales['temp_button'])
    elif active_button == 'wind_button':
        plot_station_trends(filtered_data, 'wind_speed_max_stations', "Wind Speed [m/s]", color_scales['wind_button'])
    elif active_button == 'snow_height_button':
        plot_station_trends(filtered_data, 'snow_height_mean_stations', "Height of Snowpack [cm]", color_scales['snow_height_button'])
    elif active_button == 'new_snow_button':
        plot_station_trends(filtered_data, 'new_snow_mean_stations', "Height of New Snow [cm]", color_scales['new_snow_button'])

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        template="plotly_white",
        height=190,
        legend=dict(
            font=dict(color='gray', size=8, family='Arial, sans-serif'),
            bgcolor='rgba(0, 0, 0, 0)',
            orientation='h',
            yanchor='bottom',
            y=-0.4,
            xanchor='center',
            x=0.5),
        showlegend=True,
    )

    fig.update_xaxes(
        autorange=True,
        showgrid=False,
        showticklabels=True,
        tickformat="%d-%b",
        tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

    fig.update_yaxes(
        autorange=True,
        showgrid=False,
        showticklabels=True,
        tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
    )

    return fig
