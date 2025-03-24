import pandas as pd
import numpy as np
from scheduler import schedule_model
from settings import settings
import random
import os


class ScheduleSimulator:
    def __init__(self, model):
        """
        Initialize the simulator with a scheduling model.

        :param model: A function that returns a DataFrame with scheduled appointments.
                      The DataFrame should have columns ['task_id', 'machine', 'start_time', 'end_time'].
        """
        self.model = model
        self.schedule = None  # To hold the schedule once it's generated
        self.results = None  # To hold performance metrics

    def run(self, appointments, actual_durations):
        """
        Run the simulation, scheduling the appointments and calculating performance metrics.

        :param appointments: DataFrame of appointments (only has 'task_id', 'machine', and columns for estimating
                            appointment durations).
                            The start and end times will be filled in by the model.
        :param actual_durations: DataFrame of the actual durations for each machine.
                                 It should have columns ['task_id', 'machine', 'duration'].

        :return: results: Dictionary of performance metrics in a DataFrame format.
        """
        # Generate the schedule using the model
        self.schedule = self.model(appointments)

        # Merge appointments with actual durations

        # avoid overlapping columns names, except for task_id as key
        cols = [col for col in actual_durations.columns if col not in self.schedule.columns] + ['task_id']
        merged = self.schedule.merge(actual_durations[cols], on='task_id', how='left')

        # Calculate the actual start and end times
        merged = self._calculate_actual_times(merged)
        merged.to_csv('sample_merged_df.csv')

        # Metrics Calculation
        utilization, idle_time = self._calculate_utilization(merged)
        total_wait_time = self._calculate_total_wait_time(merged)
        avg_wait_time = self._calculate_average_wait_time(merged)
        makespan = self._calculate_makespan(merged)
        task_bumping_rate = self._calculate_task_bumping_rate(merged)

        # Store the results
        self.results = {
            'utilization': utilization,
            'idle_time': idle_time,
            'total_wait_time': total_wait_time,
            'average_wait_time': avg_wait_time,
            'makespan': makespan,
            'task_bumping_rate': task_bumping_rate,
        }

        self.results = pd.DataFrame(list(self.results.items()), columns=["Metric", "Value"])

        return self.results

    @staticmethod
    def _calculate_actual_times(merged_df):
        """
        Calculate the actual start and end times based on durations and machine availability.

        :param merged_df: Merged DataFrame of scheduled appointments and actual durations.

        :return: merged_df: Updated DataFrame with actual start and end times.
        """
        # Track machine availability (i.e., when the machine is next free)
        machine_availability = {machine: merged_df['start_time'].iloc[0] for machine in merged_df['machine'].unique()}

        for idx, row in merged_df.iterrows():
            machine = row['machine']
            scheduled_start = row['start_time']
            if np.isnan(row['duration']):
                print(row['duration'])

            # account for no shows and cancellations
            if random.random() < settings['no_show_rate']:
                duration = pd.Timedelta(0)

            else:
                duration = pd.Timedelta(minutes=row['duration'])

            # Calculate actual start time as the max of scheduled start or when the machine becomes available
            actual_start = max(scheduled_start, machine_availability[machine])
            actual_end = actual_start + duration

            # Update the machine availability to reflect the new end time
            machine_availability[machine] = actual_end

            # Assign the actual start and end times back to the DataFrame
            merged_df.at[idx, 'actual_start'] = actual_start
            merged_df.at[idx, 'actual_end'] = actual_end

        return merged_df

    @staticmethod
    def _calculate_utilization(merged_df):
        """
        Calculate machine utilization and idle time based on actual appointments.

        Utilization is the percentage of time that machines were actively working on tasks
        compared to the total available time. Idle time is the amount of time machines
        were not used.

        :param merged_df: Merged DataFrame of scheduled and actual times.

        :return: tuple (utilization, idle_time):
                 - utilization: The percentage of time machines were in use.
                 - idle_time: The amount of time machines were idle.
        """
        start_of_period = merged_df['start_time'].min()
        end_of_period = merged_df['end_time'].max()
        # find total working time in minutes
        total_time_period = 60 * (settings['work_end'] - settings['work_start']) * ((end_of_period-start_of_period).days + 1)

        machine_metrics = {}

        for machine in merged_df['machine'].unique():
            machine_df = merged_df[merged_df['machine'] == machine]
            total_machine_time = total_time_period  # Total time available for each machine

            # Calculate actual usage
            actual_usage = 0
            for _, row in machine_df.iterrows():
                actual_start = row['actual_start']
                actual_end = row['actual_end']
                if not pd.isna(actual_start) and not pd.isna(actual_end):
                    actual_duration = (actual_end - actual_start).total_seconds() / 60
                    actual_usage += actual_duration

            utilization = (actual_usage / total_machine_time) * 100
            idle_time = total_machine_time - actual_usage
            machine_metrics[machine] = {'utilization': utilization, 'idle_time': idle_time}

        # Total utilization and idle time
        total_actual_usage = sum([v['utilization'] * total_time_period / 100 for v in machine_metrics.values()])
        total_utilization = (total_actual_usage / (total_time_period * len(machine_metrics))) * 100
        total_idle_time = total_time_period * len(machine_metrics) - total_actual_usage

        return total_utilization, total_idle_time

    @staticmethod
    def _calculate_total_wait_time(merged_df):
        """
        Calculate the total wait time for all tasks that started after their scheduled time.
        Total wait time sums the differences between actual start times and scheduled start times
        for all tasks that were delayed. This metric can help identify delays in the scheduling process.

        :param merged_df: Merged DataFrame of scheduled and actual times.
        :return: total_wait_time: The total wait time in minutes for all appointments.
        """
        total_wait_time = 0

        for idx, row in merged_df.iterrows():
            scheduled_start = row['start_time']
            actual_start = row['actual_start']

            if not pd.isna(actual_start) and actual_start > scheduled_start:
                wait_time = (actual_start - scheduled_start).total_seconds() / 60
                total_wait_time += wait_time

        return total_wait_time

    @staticmethod
    def _calculate_average_wait_time(merged_df):
        """
        Calculate the average wait time for tasks that started after their scheduled time.

        Average wait time provides insight into how long tasks are typically delayed after
        their scheduled start time. A lower average suggests better scheduling efficiency.

        :param merged_df: Merged DataFrame of scheduled and actual times.

        :return: avg_wait_time: The average wait time in minutes for all appointments.
        """
        total_wait_time = ScheduleSimulator._calculate_total_wait_time(merged_df)
        num_tasks = len(merged_df)
        return total_wait_time / num_tasks if num_tasks > 0 else 0

    @staticmethod
    def _calculate_makespan(merged_df):
        """
        Calculate the makespan of all tasks, which is the total time from the start of the first task
        to the completion of the last task.
        Makespan is a crucial measure in scheduling as it represents the total time required to complete a set of tasks.

        :param merged_df: Merged DataFrame of scheduled and actual times.
        :return: makespan: The total duration from the start of the first task to the end of the last task in minutes.
        """
        start_times = merged_df['actual_start'].min()
        end_times = merged_df['actual_end'].max()
        return (end_times - start_times).total_seconds() / 60 if not pd.isna(start_times) and not pd.isna(end_times) else 0

    @staticmethod
    def _calculate_task_bumping_rate(merged_df):
        """
        Calculate the task bumping rate, indicating how often tasks are rescheduled or delayed.

        Task bumping rate is important for understanding scheduling efficiency and the impact of delays on workflow.

        :param merged_df: Merged DataFrame of scheduled and actual times.

        :return: task_bumping_rate: The percentage of tasks that experienced bumping or rescheduling.
        """
        bumped_tasks = merged_df[merged_df['actual_start'] > merged_df['start_time']].shape[0]
        total_tasks = merged_df.shape[0]
        task_bumping_rate = (bumped_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        return task_bumping_rate


if __name__ == '__main__':

    df = pd.read_csv('appointment_duration_data (2).csv').rename(columns={'ProviderID': 'machine', 'ServTime': 'duration'}).dropna()[:500]
    df['task_id'] = df.index
    df['duration'] /= 60
    print(df)

    simulator = ScheduleSimulator(schedule_model)

    # Run the simulator and calculate metrics
    results = simulator.run(df, df)
    directory_path = './results'

    # write the results to the file
    os.makedirs(directory_path, exist_ok=True)
    results.to_csv(f"{directory_path}/{round(settings['alpha'], 2)}_{settings['time_slot_duration']}")
