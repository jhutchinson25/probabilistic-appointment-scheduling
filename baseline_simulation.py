import pandas as pd
from simulator import ScheduleSimulator
from settings import settings
import os
import random


def baseline_model(df):
    mean_duration = df['duration'].mean()
    df_sorted = df.sort_values(by=['machine', 'task_id']).copy()
    print(mean_duration)


    schedules = []
    machine_times = {}
    start_date = df['StartTime'].min()
    end_hour = settings['work_end']

    for _, row in df_sorted.iterrows():
        machine = row['machine']
        task_id = row['task_id']

        if machine not in machine_times:
            machine_times[machine] = start_date

        start_time = machine_times[machine]
        if start_time.hour >= end_hour:
            start_time = start_time.replace(hour=settings['work_start'], minute=0) + pd.Timedelta(days=1)
        slot = round(start_time.minute/settings['time_slot_duration']) * settings['time_slot_duration']
        if slot == 60:
            start_time = start_time.replace(
                minute=0, hour=start_time.hour + 1)
        else:
            start_time = start_time.replace(minute=round(start_time.minute/settings['time_slot_duration']) * settings['time_slot_duration'])
        end_time = start_time + pd.Timedelta(minutes=mean_duration)

        schedules.append({
            'task_id': task_id,
            'machine': machine,
            'start_time': start_time,
            'end_time': end_time
        })

        machine_times[machine] = end_time
    pd.DataFrame(schedules).to_csv('baseline_schedule.csv')
    return pd.DataFrame(schedules)


if __name__ == '__main__':

    df = pd.read_csv('appointment_duration_data (2).csv').rename(columns={'ProviderID': 'machine', 'ServTime': 'duration'}).dropna()
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df['task_id'] = df.index
    df['duration'] /= 60
    df['machine'] = random.choices(list(range(6)),k=len(df))

    print(df)
    print(df.columns)

    simulator = ScheduleSimulator(baseline_model)

    # Run the simulator and calculate metrics
    results = simulator.run(df, df)
    print(results)
    directory_path = './results'

    # write the results to the file
    os.makedirs(directory_path, exist_ok=True)
    results.to_csv("baseline_results")
