# import dash
# from dash import dcc
# # import dash_html_components as html
# from dash import html
# from dash.dependencies import Input, Output
# import plotly.express as px
# import pandas as pd
# import numpy as np
#
#
# # Sample data for demonstration
# # Replace this with actual data pulled from your database or API
#
#
# appt_type_df = pd.read_excel('CMG PROVIDERS-APPTYPES-LOCATION_RawData_10.2024.xlsx', sheet_name=0)
# provider_df = pd.read_excel('CMG PROVIDERS-APPTYPES-LOCATION_RawData_10.2024.xlsx', sheet_name=1)
#
# data = pd.DataFrame({
#     'Appointment Type': np.random.choice(appt_type_df['APPOINTMENT_TYPE_NAME'], 100),
#     'Doctor': np.random.choice(provider_df['Last Name'], 100),
#     'Appointment Duration': np.random.normal(30, 10, 100),  # in minutes
#     'Scheduled Time': pd.date_range("2023-10-01", periods=100, freq='h')
# })
#
# # print(provider_df.columns)  # Index(['Location', 'Last Name', 'First Name', 'Credentials', 'Specilaity',
# #        'Clinic Schedule(s):', 'Notes:'],
# # print(appt_type_df.columns)
# app = dash.Dash(__name__)
#
# app.layout = html.Div([
#     html.H1("Appointment Scheduling Dashboard"),
#
#     # Dropdown for Appointment Type
#     html.Label("Select Appointment Type"),
#     dcc.Dropdown(
#         id='appointment-type-dropdown',
#         options=[{'label': typ, 'value': typ} for typ in appt_type_df['APPOINTMENT_TYPE_NAME'].unique()],
#         value='Initial Consultation',
#         searchable=True
#     ),
#
#     # Dropdown for Doctor
#     html.Label("Select Doctor"),
#     dcc.Dropdown(
#         id='doctor-dropdown',
#         options=[{'label': doc, 'value': doc} for doc in provider_df['Last Name'].unique()], #+ [{'label': 'All', 'value': 'All'}],
#         value='Dr. Smith',
#         searchable=True,
#         # multi=True
#     ),
#
#     # html.Label("Enter Minimum Appointment Duration"),
#     # dcc.Dropdown(
#     #     id='duration-dropdown',
#     #     options=[10, 20, 30],
#     #     value=30,
#     #     searchable=True
#     # ),
#
#     # Graph for displaying the distribution of appointment durations
#     dcc.Graph(id='duration-distribution'),
#
#     # Text output for expected appointment duration and suggested times
#     html.Div(id='suggested-times'),
#     html.Div(id='probability-no-show')
# ])
#
#
# # Callback to update the graph and recommendations based on user input
# @app.callback(
#     [Output('duration-distribution', 'figure'),
#      Output('suggested-times', 'children'),
#      Output('probability-no-show', 'children')],
#     [Input('appointment-type-dropdown', 'value'),
#      Input('doctor-dropdown', 'value')]
# )
# def update_dashboard(selected_type, selected_doctor):
#     # Filter data based on the selected appointment type and doctor
#     filtered_data = data[(data['Appointment Type'] == selected_type) &
#                          (data['Doctor'] == selected_doctor)]
#
#     # Plot the distribution of appointment durations (Histogram)
#     fig = px.histogram(filtered_data, x='Appointment Duration',
#                        nbins=10, title=f'Appointment Duration for {selected_type} with {selected_doctor}')
#
#     # Calculate expected appointment duration (mean)
#     expected_duration = filtered_data['Appointment Duration'].mean()
#
#     # Recommend best time to schedule based on availability (e.g., using scheduled times)
#     recommended_times = filtered_data['Scheduled Time'].sample(3).dt.strftime('%Y-%m-%d %H:%M:%S').values
#
#     # Create a recommendation message
#     recommendation_message = f"Expected duration: {expected_duration:.2f} minutes.\n"
#     recommendation_message += "Suggested appointment times: "
#     recommendation_message += ", ".join(recommended_times)
#
#     return fig, recommendation_message, "The probability of a no show or cancellation is 0.1"
#
#
# if __name__ == '__main__':
#     app.run_server(debug=True)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px

# Sample data for demonstration


# Load data for appointment types and providers
appt_type_df = pd.read_excel('CMG PROVIDERS-APPTYPES-LOCATION_RawData_10.2024.xlsx', sheet_name=0)
provider_df = pd.read_excel('CMG PROVIDERS-APPTYPES-LOCATION_RawData_10.2024.xlsx', sheet_name=1)

data = pd.DataFrame({
    'Appointment Type': np.random.choice(appt_type_df['APPOINTMENT_TYPE_NAME'], 100000),
    'Doctor': np.random.choice(appt_type_df['RESOURCES_AVAILABLE'], 100000),
    'Appointment Duration': np.random.normal(30, 10, 100000),  # in minutes
    'Scheduled Time': pd.date_range("2023-10-01", periods=100000, freq='h')
})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(
        "Healthcare Appointment Scheduling Dashboard",
        style={
            'textAlign': 'center',
            'color': '#003366',  # Dark blue color
            'fontSize': '36px',
            'fontFamily': 'Arial, sans-serif'
        }
    ),

    # Dropdown for Appointment Type
    html.Label("Select Appointment Type"),
    dcc.Dropdown(
        id='appointment-type-dropdown',
        options=[{'label': typ, 'value': typ} for typ in appt_type_df['APPOINTMENT_TYPE_NAME'].unique()],
        placeholder="Select an appointment type",
        searchable=True
    ),

    # Dropdown for Doctor
    html.Label("Select Doctor"),
    dcc.Dropdown(
        id='doctor-dropdown',
        options=[{'label': doc, 'value': doc} for doc in appt_type_df['RESOURCES_AVAILABLE'].unique()],
        placeholder="Select a doctor",
        searchable=True
    ),

    # Graph for displaying the distribution of appointment durations
    dcc.Graph(id='duration-distribution'),

    # Text output for expected appointment duration and suggested times
    html.Div(id='suggested-times'),
    html.Div(id='probability-no-show')
])

# Callback to update Doctor options based on Appointment Type selection and vice versa
@app.callback(
    [Output('doctor-dropdown', 'options'),
     Output('appointment-type-dropdown', 'options')],
    [Input('appointment-type-dropdown', 'value'),
     Input('doctor-dropdown', 'value')]
)
def update_dropdowns(selected_type, selected_doctor):
    # Filter doctor options based on selected appointment type
    if selected_type:
        filtered_doctors = appt_type_df[appt_type_df['APPOINTMENT_TYPE_NAME'] == selected_type]['RESOURCES_AVAILABLE'].unique()
        doctor_options = [{'label': doc, 'value': doc} for doc in filtered_doctors]
    else:
        doctor_options = [{'label': doc, 'value': doc} for doc in appt_type_df['RESOURCES_AVAILABLE'].unique()]

    # Filter appointment type options based on selected doctor
    if selected_doctor:
        filtered_types = appt_type_df[appt_type_df['RESOURCES_AVAILABLE'] == selected_doctor]['APPOINTMENT_TYPE_NAME'].unique()
        appt_type_options = [{'label': typ, 'value': typ} for typ in filtered_types]
    else:
        appt_type_options = [{'label': typ, 'value': typ} for typ in appt_type_df['APPOINTMENT_TYPE_NAME'].unique()]

    return doctor_options, appt_type_options


# Callback to update the graph and recommendations based on dropdown selections
@app.callback(
    [Output('duration-distribution', 'figure'),
     Output('suggested-times', 'children'),
     Output('probability-no-show', 'children')],
    [Input('appointment-type-dropdown', 'value'),
     Input('doctor-dropdown', 'value')]
)
def update_dashboard(selected_type, selected_doctor):
    filtered_data = data
    if selected_type:
        filtered_data = filtered_data[filtered_data['Appointment Type'] == selected_type]
    if selected_doctor:
        filtered_data = filtered_data[filtered_data['Doctor'] == selected_doctor]

    if len(filtered_data) == 0:
        return

    # Plot the distribution of appointment durations
    fig = px.histogram(filtered_data, x='Appointment Duration',
                       nbins=10, title=f'Appointment Duration for {selected_type} with {selected_doctor}')

    # Calculate expected appointment duration (mean)
    expected_duration = filtered_data['Appointment Duration'].mean()

    # Recommend best times based on availability
    recommended_times = filtered_data['Scheduled Time'].sample(3).dt.strftime('%Y-%m-%d %H:%M:%S').values
    # recommendation_message = f"Expected duration: {expected_duration:.2f} minutes.\n"
    # recommendation_message += "Suggested appointment times: "
    # recommendation_message += ", ".join(recommended_times)
    recommendation_message = html.Div([
        html.P(f"**Expected Duration:** {expected_duration:.2f} minutes", style={'fontWeight': 'bold'}),
        html.P("**Suggested Appointment Times:**", style={'fontWeight': 'bold'}),
        html.Ul([html.Li(time) for time in recommended_times])])

    return fig, recommendation_message, "The probability of a no show or cancellation is 0.1"

if __name__ == '__main__':
    app.run_server(debug=True)

