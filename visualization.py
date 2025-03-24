# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.dates as mdates
# # import numpy as np
# # import matplotlib.patches as mpatches
# #
# # # Load the CSV file
# # df = pd.read_csv('sample_schedule.csv', parse_dates=['start_time', 'end_time'])
# #
# # # Sort by start time to process overlaps
# # df = df.sort_values(by=['machine', 'start_time'])
# #
# # # Assign colors for different tasks
# # task_ids = df['task_id'].unique()
# # task_colors = {task_id: plt.cm.tab10(i % 10) for i, task_id in enumerate(task_ids)}
# #
# # # Create a figure and axis
# # fig, ax = plt.subplots(figsize=(12, 6))
# #
# # overlap_offsets = {machine: 0 for machine in df['machine'].unique()}
# # previous_end_times = {machine: None for machine in df['machine'].unique()}
# #
# # # Iterate over rows to plot bars
# # for index, row in df[:30].iterrows():
# #     machine = row['machine']
# #     color = task_colors[row['task_id']]
# #
# #     if previous_end_times[machine] is not None and row['start_time'] < previous_end_times[machine]:
# #         overlap_offsets[machine] += 0.2  # Slightly offset overlapping tasks
# #     else:
# #         overlap_offsets[machine] = 0  # Reset offset when no overlap
# #
# #     ax.barh(y=machine - overlap_offsets[machine],
# #             width=(row['end_time'] - row['start_time']).total_seconds() / 60,
# #             left=row['start_time'], height=0.4, color=color, edgecolor='black')
# #     previous_end_times[machine] = row['end_time']
# #
# # # Set x-axis limits
# # ax.set_xlim(df['start_time'][:30].min(), df['end_time'][:30].max())
# #
# # # Format the x-axis
# # ax.set_xlabel('Time')
# # ax.set_ylabel('Machine')
# # ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
# # ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# # plt.xticks(rotation=45)
# #
# # # Create a custom legend
# # legend_patches = [mpatches.Patch(color=color, label=f"Task {task_id}") for task_id, color in task_colors.items()]
# # ax.legend(handles=legend_patches, title="Tasks", loc='upper right')
# #
# # # Display the Gantt chart
# # plt.title('Machine Task Timeline with Overlaps')
# # plt.show()
#
# import pandas as pd
# import plotly.express as px
#
# # Load the CSV file
# df = pd.read_csv('sample_schedule.csv', parse_dates=['start_time', 'end_time'])[:30]
#
# # Ensure machine column is treated as categorical for better plotting
# df['machine'] = df['machine'].astype(str)
#
# # Create the Gantt chart
# fig = px.timeline(df, x_start='start_time', x_end='end_time', y='machine', color='task_id',
#                   title='Machine Task Timeline', labels={'machine': 'Machine', 'task_id': 'Task ID'})
#
# # Format the x-axis
# fig.update_xaxes(title_text='Time', tickformat='%H:%M', showgrid=True)
# fig.update_yaxes(title_text='Machine')
#
# # Improve layout
# fig.update_layout(
#     height=600,
#     margin=dict(l=100, r=50, t=50, b=50),
#     legend_title_text='Tasks',
#     xaxis=dict(showline=True, linewidth=1, linecolor='black')
# )
#
# # Show the chart
# fig.show()
#

import pandas as pd
import plotly.express as px
import plotly.io as pio
import calendar
from datetime import datetime

df = pd.read_csv('sample_schedule.csv')
df = df[df['machine']==0]
for col in ['start_time', 'end_time']:
    df[col] = pd.to_datetime(df[col])


import pandas as pd
import plotly.express as px
import plotly.io as pio
import calendar
from datetime import datetime


# Extract necessary time components
df['date'] = df['start_time'].dt.date
df['day'] = df['start_time'].dt.day
df['weekday'] = df['start_time'].dt.strftime('%A')
df['week'] = df['start_time'].dt.strftime('%U').astype(int)  # Get week number

df['task_info'] = df['task_id'].astype(str) + ' (' + df['start_time'].dt.strftime('%H:%M') + '-' + df['end_time'].dt.strftime('%H:%M') + ')'

# Get the month and year for title
month_name = df['start_time'].dt.strftime('%B %Y').unique()[0]

# Create a structured calendar format
calendar_data = df.groupby(['week', 'weekday', 'day'])['task_info'].apply('<br>'.join).reset_index()

# Convert weekday order into a list
weekday_order = list(calendar.day_name)

# Create a timeline chart for a structured weekly calendar visualization
fig = px.timeline(calendar_data, x_start='day', x_end='day', y='week', text='task_info',
                  title=f'Scheduled Appointments - {month_name}', labels={'week': 'Week', 'day': 'Day of Month'})

# Adjust layout for a calendar-like view
fig.update_traces(textposition='inside')
fig.update_yaxes(dtick=1, tickmode='linear', title='Week', autorange='reversed')
fig.update_xaxes(categoryorder='array', categoryarray=weekday_order, title='Day of Week')
fig.update_layout(height=600, margin=dict(l=50, r=50, t=50, b=50))

# Show the chart
pio.show(fig)
