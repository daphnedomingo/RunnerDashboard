import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import numpy as np
import math
import calendar
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objs as go  # Added this import
import datetime# Added this import



# Create a Dash web application
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX])

## Initial variables needed #####
df = pd.read_csv("running log_cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'])  # Convert the "Date" column to datetime
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
click_data_output = html.Div()
click_data_temp = None
click_data_hour = None
click_data_shoes = None
runner_name= 'Jane Doe'
min_year= df['Date'].dt.year.min()
total_runs= len(df)
my_year= ""

# Define the layout of the web page
app.layout = dbc.Container([
#============== Row 1: Dashboard Name, Year Dropdown, and User Info ==========================
    dbc.Row([
        #======= R1 Col 1: Dashboard Title=======
        dbc.Col([
            html.H1("RUN PERFORMANCE VISUALISER"),
            html.P( "This dashboard offers a comprehensive view of the runner's performance, including details on distance covered, speed, calories burned, and more. With interactive graphs and filters, you can fine-tune your analysis and uncover hidden patterns in your data.",)
        ], className= 'mt-3 mx-3 w-auto'),
        #======== R1, Col2: Year Dropdown=======
        dbc.Col([
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in sorted(df['Date'].dt.year.unique())],
                value=df['Date'].dt.year.max())
        ], align='center', className='text-right', width= 2),
        #======== R1, Col3: User Info=================
        dbc.Col([html.Img(src=app.get_asset_url('hi.png'),
                         className='set_img',
                         style={'aria-hidden': True, 'margin-left': '170px','width': '100px', 'height': '100px'}), 
        ], align='center', className='text-left', width=2 ),
        dbc.Col([
            html.H6(f'Runner: {runner_name}'),
            html.H6(f'Running since: {min_year}'),
            html.H6(f'Total runs completed: {total_runs}'), 
        ], align='center', className='text-left', width=3 )
    ], className='mb-3 sticky-top bg-light'),

#============== Row 2: text to let user know that the 3 graphs can be used as filter ==========================
    dbc.Row([
        html.H6('Use the three graphs below to filter performance by temperature, hour, and shoe-type'),
    ], className='mb-3 align-content-center text-center'),

#============== Row 3: Graph filters and reset buttons ==========================================
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dcc.Graph(id='temperature-plot', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '1', 'border': 'primary', 'border-radius': '2px','padding': '3px'}),
            ],className="h-30"),
            dbc.Row([
                html.Button('Reset Temperature Filter', id='reset-button-temp', className='btn btn-outline-primary', style={'padding': '3px'}),
            ]),
        ], className='w-30'),
        dbc.Col([
            dbc.Row([
                dcc.Graph(id='hour-plot', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '2', 'border': 'primary', 'border-radius': '2px','padding': '3px'}),
            ],className="h-30"),
            dbc.Row([
                html.Button('Reset Hour Filter', id='reset-button-hour', className='btn btn-outline-primary', style={'padding': '3px'}),
            ]),
        ], className='w-30'),
        dbc.Col([
            dbc.Row([
                dcc.Graph(id='shoes-plot', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '3', 'border': 'primary', 'border-radius': '2px','padding': '3px'}),
            ],className="h-30"),
            dbc.Row([
                html.Button('Reset Shoes Filter', id='reset-button-shoes', className='btn btn-outline-primary', style={'padding': '3px'}), 
            ]),
        ], className='mb-3 w-30')
    ], className= 'mx-1 mr-1 mb-3 align-content-center', style={'display': 'grid', 'grid-template-columns': '1fr 1fr 1fr', 'height': '20'}),

#===================== Row 4: Reset All Filters button=======================
    dbc.Row([
        html.Button('Reset All Filters', id='reset-button-all', className='btn btn-outline-primary', style={'padding': '3px'}),
    ], className= 'mx-1 mr-1 mb-3 align-content-center'),

#===================== Row 5: Year Drilldown Header ================================
    dbc.Row([
        html.H1('My Running Performance', style={'text-align': 'center'}),
    ],className= 'mb-3 align-content-center'),

#==================== Row 6: Statistics display ===================================
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Average Distance (km)',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='average_distance', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ],  ),
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Average Speed (km/hr)',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='average_speed', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ]),
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Average Time Taken (mins)',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='averagetimetaken', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ]),
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Total Calories Burned (Cal)',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='totalcalories', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ]),
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Total Distance Covered (km)    ',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='totaldistance', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ]),
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Total Time Taken (mins)',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='totaltimetaken', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ]),
        ]),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Total Number of Runs Completed',style={'text-align': 'center'}, className='text-white bg-primary'),
                dbc.CardBody([
                    dcc.Markdown(id='Runningcount', style={'text-align': 'center'}),
                ], className= 'w-auto'),
            ]),
        ]),
    ], className='mb-3 align-content-center', style={'display': 'grid', 'grid-template-columns': '1fr 1fr 1fr 1fr 1fr 1fr 1fr'}),

#=================Row 7: Year Drilldown Graphs================================
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='distance', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '1', 'padding': '3px'}),
        ]),
        dbc.Col([
            dcc.Graph(id='monthly-distance', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '2', 'padding': '3px'}),
        ]),
    ], className= 'mb-3 align-content-center', style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'height': '20'}),
    dbc.Row([
        dcc.Graph(id='day-of-week-plot', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '1','padding': '3px'}),
    ], className='mx-1 mr-1 mb-3 align-content-center'),

#================== Rows 8, 9, 10, 11: Month Drilldown Header, Dropdown, Graphs, Log Table==============================
    dbc.Row([
        html.H2(f'Drill down to the monthly performance!', style={'text-align': 'center'}),
    ],className= 'mb-3 align-content-center'),
    dbc.Row([
        dcc.Dropdown(
            id='month-dropdown',
            options=[{'label': month, 'value': month} for month in calendar.month_name[1:]],
            value=calendar.month_name[df['Date'].dt.month.min()],
        ),
    ],className= 'mb-3 align-content-center'),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='stacked-bar-plot', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '1', 'padding': '3px'}),
        ]),
        dbc.Col([
            dcc.Graph(id='weekly-run-plot', figure = go.Figure(layout=dict(template='plotly')), style={'grid-row': '1', 'grid-column': '1', 'padding': '3px'}),
        ]),
    ], className= 'mb-3 align-content-center', style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'height': '20'}),
    dbc.Container([
        dbc.Row([
            html.H3('Run logs',),
        ],className= 'w-100 text-center'),
        dbc.Row([
            dash_table.DataTable(
                    id='data-table',
                    style_table={'height': '300px', 'overflowY': 'auto'},
                ),
        ],className= 'mb-3 w-100'),
    ], className= 'mb-3 align-content-center'),
    dbc.Row()
], className= 'mw-10', fluid=True, style={'backgroundColor': '#F0E6FA'})



#====== function for day of week graph================
def reorder_days_of_week(day_of_week_counts):
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return day_of_week_counts.reindex(days_of_week, fill_value=0)

#====== function to get weekly run frequency within month drilldown (fig7)===============
def get_week_frequency(year, month, df):
    # Get the first and last day of the specified month
    _, last_day = calendar.monthrange(year, month)
    first_day = f"{year}-{month:02d}-01"
    last_day = f"{year}-{month:02d}-{last_day}"
    runs_df = df


    # Determine the weekday (0 = Monday, 6 = Sunday) of the first day of the month
    weekday_of_first_day = pd.Timestamp(first_day).dayofweek

    # Adjust the start date to the beginning of the first week
    if weekday_of_first_day != 0:  # If the month starts on a day other than Monday
        start = pd.Timestamp(first_day) - pd.DateOffset(days=weekday_of_first_day)

    else:
        start = pd.Timestamp(first_day)

    end_date= []
    start_date=[]
    end= start
    temp_start= start


    while end<= pd.Timestamp(last_day):
        start_date.append(temp_start.strftime('%y-%m-%d'))
        end = temp_start + pd.DateOffset(days=6)
        end_date.append(end.strftime('%y-%m-%d'))
        temp_start= pd.Timestamp(temp_start) + pd.DateOffset(days=7)

    weeks_df = pd.DataFrame({'Week': [str(a) + " to " + str(b) for a, b in zip(start_date, end_date)], "Start": start_date, "End": end_date})


    # Convert 'Start' and 'End' columns to datetime objects
    weeks_df['Start'] = pd.to_datetime(weeks_df['Start'], format='%y-%m-%d')
    weeks_df['End'] = pd.to_datetime(weeks_df['End'], format='%y-%m-%d')

    # Convert 'Date' column in 'runs_df' to datetime objects
    runs_df['Date'] = pd.to_datetime(runs_df['Date'])

    # Create an empty list to store the total run counts
    total_runs = []

    # Iterate through 'Weeks' DataFrame
    for index, row in weeks_df.iterrows():
        start_date = row['Start']
        end_date = row['End']
        
        # Filter 'runs_df' to get runs within the current week
        runs_in_week = runs_df[(runs_df['Date'] >= start_date) & (runs_df['Date'] <= end_date)]
        
        # Get the count of runs for the current week
        run_count = len(runs_in_week)
        
        # Append the run count to the 'total_runs' list
        total_runs.append(run_count)

    # Add the 'Total Runs' column to 'Weeks' DataFrame
    weeks_df['Total Runs'] = total_runs

    # Fill missing values with 0
    weeks_df['Total Runs'].fillna(0, inplace=True)

    return weeks_df

#============= function for weekly run distance plot (fig8)====================
def get_week_distance(year, month, df):
    # Get the first and last day of the specified month
    _, last_day = calendar.monthrange(year, month)
    first_day = f"{year}-{month:02d}-01"
    last_day = f"{year}-{month:02d}-{last_day}"
    runs_df = df

    # Determine the weekday (0 = Monday, 6 = Sunday) of the first day of the month
    weekday_of_first_day = pd.Timestamp(first_day).dayofweek

    # Adjust the start date to the beginning of the first week
    if weekday_of_first_day != 0:  # If the month starts on a day other than Monday
        start = pd.Timestamp(first_day) - pd.DateOffset(days=weekday_of_first_day)

    else:
        start = pd.Timestamp(first_day)

    end_date= []
    start_date=[]
    end= start
    temp_start= start


    while end<= pd.Timestamp(last_day):
        start_date.append(temp_start.strftime('%y-%m-%d'))
        end = temp_start + pd.DateOffset(days=6)
        end_date.append(end.strftime('%y-%m-%d'))
        temp_start= pd.Timestamp(temp_start) + pd.DateOffset(days=7)

    weeks_df = pd.DataFrame({'Week': [str(a) + " to " + str(b) for a, b in zip(start_date, end_date)], "Start": start_date, "End": end_date})

    # Convert 'Start' and 'End' columns to datetime objects
    weeks_df['Start'] = pd.to_datetime(weeks_df['Start'], format='%y-%m-%d')
    weeks_df['End'] = pd.to_datetime(weeks_df['End'], format='%y-%m-%d')

    # Convert 'Date' column in 'runs_df' to datetime objects
    runs_df['Date'] = pd.to_datetime(runs_df['Date'])

    # Create an empty list to store the total run counts
    total_distance = []
    ave_distance=[]

    # Iterate through 'Weeks' DataFrame
    for index, row in weeks_df.iterrows():
        start_date = row['Start']
        end_date = row['End']
        
        # Filter 'runs_df' to get runs within the current week
        runs_in_week = runs_df[(runs_df['Date'] >= start_date) & (runs_df['Date'] <= end_date)]
        
        # Get the count of runs for the current week
        run_count = runs_in_week['Distance'].sum()
        ave_dist= runs_in_week['Distance'].mean()
        
        # Append the run count to the 'total_runs' list
        total_distance.append(run_count)
        ave_distance.append(ave_dist)

    # Add the 'Total Runs' column to 'Weeks' DataFrame
    weeks_df['Total Distance'] = total_distance
    weeks_df['Average Distance']= ave_distance

    # Fill missing values with 0
    weeks_df['Total Distance'].fillna(0, inplace=True)
    weeks_df['Average Distance'].fillna(0, inplace=True)

    return weeks_df

#======= function for statistics to avoid NaN display
def if_empty(temp):
    if np.isnan(temp):
        return 0
    else:
        return temp


# Callback to update the filter plots based on the selected year and clickData
@app.callback(
    Output('temperature-plot', 'figure'),
    Output('shoes-plot', 'figure'),
    Output('hour-plot', 'figure'),
    Input('year-dropdown', 'value'),
    Input('temperature-plot', 'clickData'),
    Input('hour-plot', 'clickData'),
    Input('shoes-plot', 'clickData'),
)
def update_plot_filters(selected_year, clickData1, clickData2, clickData3):
    global my_year
    my_year= selected_year
    filtered_df = df[df['Date'].dt.year == selected_year]
    click_data_temp = clickData1
    click_data_hour = clickData2
    click_data_shoes = clickData3

    
    #========Temp Graph===============
    max_temp = int(filtered_df['Temp'].max())

    # Calculate the end of the temperature range as the closest multiple of 5 to max_temp
    end_of_range = 5 * math.ceil(max_temp / 5)

    # Create temperature intervals with 5-degree increments
    temperature_ranges = list(range(0, end_of_range + 1, 5))

    # Initialize a dictionary to store the counts for each temperature range
    temperature_counts = {f"{lower}-{upper}": 0 for lower, upper in zip(temperature_ranges, temperature_ranges[1:])}

    # Update the counts based on the DataFrame
    for i in range(len(temperature_ranges) - 1):
        lower_temp = temperature_ranges[i]
        upper_temp = temperature_ranges[i + 1]
        count = len(filtered_df[(filtered_df['Temp'] >= lower_temp) & (filtered_df['Temp'] < upper_temp)])
        temperature_counts[f"{lower_temp}-{upper_temp}"] = count

    # Create a new DataFrame for the bar chart
    data = {'Temperature Range': list(temperature_counts.keys()), 'Number of Runs': list(temperature_counts.values()), 'Lower Temp': [int(label.split('-')[0]) for label in list(temperature_counts.keys())]}
    df_bar = pd.DataFrame(data)  

    # Create the bar chart with 'Lower Temp' determining the color
    fig2 = px.bar(
        df_bar,
        x='Temperature Range',
        y='Number of Runs',
        labels={'x': 'Temperature Range', 'y': 'Number of Runs'},
        title=f"Number of Runs by Temperature in {selected_year}",
        color='Lower Temp',
        color_continuous_scale="burg",
        color_continuous_midpoint=df_bar['Lower Temp'].mean(),  # Set midpoint using mean of Lower Temp
        )
    fig2.update_layout(title_x=0.5)
    fig2.layout.coloraxis.colorbar.title = 'Temperature'

    if clickData1:
        selected_temp = clickData1['points'][0]['x']
        # Define the color for each bar based on the selected category
        colors = ['orange' if temp == selected_temp else 'aliceblue' for temp in df_bar['Temperature Range']]

        # Update the marker colors for the bars
        fig2.update_traces(marker=dict(color=colors))    

    #================Shoes graph=========================
    shoe_counts = filtered_df['Shoes'].value_counts().reset_index()
    shoe_counts.columns = ['Shoes', 'Count']

    # Create a horizontal bar graph
    fig3 = px.bar(
        shoe_counts,
        x='Count',
        y='Shoes',
        color = 'Count',
        orientation='h',
        color_continuous_scale='Purp',
        labels={'Shoes': 'Shoe Type', 'Count': 'Usage'},
        title=f"Shoe Usage in {selected_year}"
    )
    fig3.update_layout(title_x=0.5)

    if click_data_shoes:
        selected_shoes = click_data_shoes['points'][0]['y']
        # Define the color for each bar based on the selected category
        colors = ['orange' if shoes == selected_shoes else 'aliceblue' for shoes in shoe_counts['Shoes']]

        # Update the marker colors for the bars
        fig3.update_traces(marker=dict(color=colors))    

    #================Time Graph=====================
    hour_df = filtered_df.groupby('Hour').size().reset_index(name='Count')
    hour_df['Hour'] = hour_df['Hour'].apply(lambda x: f'{x:02d}:00')


    # Create a bar graph
    fig4 = px.bar(
        hour_df,
        x='Hour',
        y='Count',
        color = 'Count',
        color_continuous_scale='mint',
        labels={'Hour': 'Hour', 'Count': 'Number of Runs'},
        title=f"Runs by Hour in {selected_year}"
    )
    fig4.update_layout(
        title_x=0.5, 
        xaxis=dict(
            tickangle=-45)
    )


    if click_data_hour:
        selected_hour = click_data_hour['points'][0]['x']
        # Define the color for each bar based on the selected category
        colors = ['orange' if hour == selected_hour else 'aliceblue' for hour in hour_df['Hour']]

        # Update the marker colors for the bars
        fig4.update_traces(marker=dict(color=colors))    

    return fig2, fig3,fig4


# Callback for displaying all the other graphs and statistics (other than filter graphs)
@app.callback(
    Output('day-of-week-plot', 'figure'),
    Output('distance', 'figure'),  # Changed the id to 'distance'
    Output('monthly-distance', 'figure'),
    Output('data-table', 'data'),
    Output('stacked-bar-plot', 'figure'),
    Output('weekly-run-plot', 'figure'),
    Output('average_distance', 'children'),
    Output('average_speed', 'children'),
    Output('totalcalories', 'children'),
    Output('totaldistance', 'children'),
    Output('averagetimetaken', 'children'),
    Output('totaltimetaken', 'children'),
    Output('Runningcount', 'children'),
  
    Input('year-dropdown', 'value'),
    Input('month-dropdown', 'value'),
    Input('temperature-plot', 'clickData'),
    Input('hour-plot', 'clickData'),
    Input('shoes-plot', 'clickData'),
)
def display_click_data(selected_year,selected_month, clickData1, clickData2, clickData3):
    lower = 0
    upper = 0
    selected_hour = 0
    selected_shoes = ""
    show_month=selected_month
    global click_data_temp, click_data_hour, click_data_shoes
    click_data_temp = clickData1
    click_data_hour = clickData2
    click_data_shoes = clickData3

#===========updating the dataframe based on clickData from graphical filters as well as chosen month and year from dropdowns
    filtered_df = df[df['Date'].dt.year == selected_year]

    if not click_data_temp:
        pass
    else:
        selected_temp = click_data_temp['points'][0]['x']
        lower, upper = map(int, selected_temp.split('-'))
        filtered_df = filtered_df[(filtered_df['Temp'] >= lower) & (filtered_df['Temp'] <= upper)]

    if not click_data_hour:
        pass
    else:
        selected_hour = click_data_hour['points'][0]['x']
        selected_hour_int = int(selected_hour.split(":")[0])
        filtered_df = filtered_df[filtered_df['Hour'] == selected_hour_int]

    if not click_data_shoes:
        pass
    else:
        selected_shoes = click_data_shoes['points'][0]['y']
        filtered_df = filtered_df[filtered_df['Shoes'] == selected_shoes]
#=================================================================

### ==== day of week graph=======
    day_of_week_counts = filtered_df['Date'].dt.day_name().value_counts()
    day_of_week_counts = reorder_days_of_week(day_of_week_counts) #see function in functions section above after layout
    day_of_week_percentage = (day_of_week_counts / day_of_week_counts.sum() * 100).fillna(0)

    fig1 = px.bar(
        x=day_of_week_percentage.index,
        y=day_of_week_percentage.values,
        labels={'x': 'Day of the Week', 'y': 'Percentage (%) of Runs'},
        title=f"Percentage of Runs for Each Day of the Week in {selected_year}", color_discrete_sequence=['mistyrose'] 
    )

    for i, val in enumerate(day_of_week_percentage.values):
        fig1.add_annotation(
            text=f"{val:.2f}%",
            x=day_of_week_percentage.index[i],
            y=val + 1,  # Adjust the vertical offset
            showarrow=False
        )

    fig1.update_layout(title_x=0.5)

#===============Speed vs Distance Graph
    fig5 = px.scatter(filtered_df, x='Distance', y='Avg Pace', title='Speed Vs Distance', trendline='ols')
    fig5.update_xaxes(title_text='Distance (KM)')
    fig5.update_yaxes(title_text='Average Pace')
    fig5.update_layout(title_x=0.5)
    fig5.update_traces(marker=dict(color='darkseagreen'),line=dict(color='slateblue'))
   


#================ Monthly Distance Graph
    month_names = [calendar.month_name[month] for month in range(1, 13)]
    monthly_distance = filtered_df.groupby(filtered_df['Date'].dt.month)['Distance'].sum().reindex(range(1, 13), fill_value=0)
    monthly_pace = filtered_df.groupby(filtered_df['Date'].dt.month)['Avg Pace'].mean().reindex(range(1, 13), fill_value=0)

    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=month_names, y=monthly_distance, name='Distance',marker=dict(color='lightskyblue')))
    fig6.add_trace(go.Scatter(x=month_names, y=monthly_pace, mode='lines+markers', name='Average Pace', yaxis='y2', line=dict(color='midnightblue')))

    fig6.update_layout(
        title='Monthly Mileage and Average Pace',
        xaxis=dict(title='Month'),
        yaxis=dict(title='Total Distance (KM)'),
        yaxis2=dict(title='Average Pace', overlaying='y', side='right', showgrid=False),
        title_x=0.5, 
    )

#=================Data Table
    selected_columns = ["Date", "Title", "Distance", "Avg Pace", "Calories", "Avg HR", "Max HR", "Shoes"]
    selected_month = list(calendar.month_name).index(selected_month)
    filtered_data = filtered_df[filtered_df['Date'].dt.month == selected_month]
    filtered_data['Date'] = sorted(filtered_data['Date'].dt.day)
    # filtered_data['Date'] = sorted(filtered_data['Date'])
    table_data = filtered_data[selected_columns].to_dict('records')

#================  Weekly run frequency Plot
    filtered_data = filtered_df[filtered_df['Date'].dt.month == selected_month]
    weekly_run_frequency = get_week_frequency(selected_year, selected_month, filtered_data) #see function in functions section above after layout
    weekly_run_frequency.columns= ['Week', 'Start', 'End', 'Total Runs Completed']

    fig7 = px.bar(weekly_run_frequency, x='Week', y='Total Runs Completed', title=f'Weekly Run Frequency in {show_month} {selected_year}')
    fig7.update_layout(
        title_x=0.5,
        xaxis=dict(
            tickangle=-45)
    )
    
#=============================Weekly Run Distance Plot
    filtered_data = filtered_df[filtered_df['Date'].dt.month == selected_month]
    weekly_run_distance = get_week_distance(selected_year, selected_month, filtered_data) #see function in functions section above after layout
    weekly_run_distance.column=['Week', 'Start', 'End', 'Total Distance', 'Average Distance']
    week_names= weekly_run_distance['Week']
    T_dist= weekly_run_distance['Total Distance']
    A_dist= weekly_run_distance['Average Distance']

    fig8_title= f'Weekly Run Distance in {show_month} {selected_year}'

    fig8 = go.Figure()
    fig8.add_trace(go.Bar(x=week_names, y=T_dist, name='Total Distance',marker=dict(color='orchid')))
    fig8.add_trace(go.Scatter(x=week_names, y=A_dist,  mode='lines+markers', name='Average Distance', yaxis='y2', line=dict(color='orange')))

    fig8.update_layout(
        title=fig8_title,
        xaxis=dict(title='Week',tickangle=-45),
        yaxis=dict(title='Total Distance (KM)'),
        yaxis2=dict(title='Average Distance', overlaying='y', side='right', showgrid=False),
        title_x=0.5,
        legend=dict(x=1.05, y=1)
    )
   
#=========================Statistics
    average_distance = round(if_empty(filtered_df['Distance'].mean()),2)
    average_speed = round(if_empty(filtered_df['Avg Pace'].mean()),2)
    totalcalories = round(if_empty(filtered_df['Calories'].sum()),2)
    totaldistance = round(if_empty(filtered_df['Distance'].sum()),2)
    averagetimetaken= round(if_empty(filtered_df['Time taken'].mean()),2)
    totaltimetaken= round(if_empty(filtered_df['Time taken'].sum()),2)
    Runningcount = round(if_empty(filtered_df['Date'].nunique()),2)

    return fig1, fig5, fig6, table_data, fig7, fig8, f'{average_distance}',f'{average_speed}', f'{totalcalories}', f'{totaldistance}', f'{averagetimetaken}', f'{totaltimetaken}', f'{Runningcount}'

#============ callbacks to enable reset of graph filters using reset buttons
@app.callback(
    Output('temperature-plot', 'clickData'),
    Input('reset-button-temp', 'n_clicks'),
    Input('reset-button-all', 'n_clicks'),
    State('temperature-plot', 'id'),
)
def reset_click_data_temp(n_clicks,n_clicks_all, graph_id):
    global click_data_temp
    if n_clicks is not None:
        click_data_temp = None
    elif n_clicks_all is not None:
        click_data_temp = None
    return None

@app.callback(
    Output('hour-plot', 'clickData'),
    Input('reset-button-hour', 'n_clicks'),
    Input('reset-button-all', 'n_clicks'),
    State('hour-plot', 'id'),
)
def reset_click_data_hour(n_clicks,n_clicks_all, graph_id):
    global click_data_hour

    if n_clicks is not None:
        click_data_hour = None
    elif n_clicks_all is not None:
        click_data_hour = None
    return None

@app.callback(
    Output('shoes-plot', 'clickData'),
    Input('reset-button-shoes', 'n_clicks'),
    Input('reset-button-all', 'n_clicks'),
    State('shoes-plot', 'id'),
)
def reset_click_data_shoes(n_clicks,n_clicks_all, graph_id):
    global click_data_shoes

    if n_clicks is not None:
        click_data_shoes = None
    elif n_clicks_all is not None:
        click_data_shoes = None     
    return None




if __name__ == '__main__':
    app.run_server(debug=True, port=8090)

