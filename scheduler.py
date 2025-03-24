import heapq
import numpy as np
# import random
# from scipy.stats import expon
# from typing import Callable
import json
from duration_model import DurationModel
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
from settings import settings


# TO DO: calculate first time slot based on the expected duration of appointments up to that point in the day
# save the expected durations in the schedule

@dataclass
class Task:
    name: str
    compatible_machines: list
    distribution_samples: list  # instance of scipy.stats rvs distribution
    failure_rate: float
    priority: int


@dataclass
class ScheduledItem:
    task_id: int
    task: Task
    start_time_slot: int
    end_time_slot: int


def calculate_mean_weighted_absolute_error(preds, values, alpha):
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    diff = values - preds
    return np.mean(np.where(diff<0, np.abs(diff) * alpha, diff))


# joint optimization of idle time and patient waiting time
class TaskScheduler:

    def __init__(self, machines: list[int], num_time_slots: int, alpha=settings['alpha'],
                 slot_size=settings['time_slot_duration'], work_start: int = settings['work_start'],
                 work_end: int = settings['work_end']):
        """
        :param machines: the available machines
        :param num_time_slots: the number of time slots to include in the future for scheduling
        :param current_schedule:
        :param alpha: weight to determine the relationship between machine idle time and part wait time
        :param slot_size: the length of the basic scheduling block
        :param estimation_mode: determines the input type for calculated the expected time. Can be 'distribution' for
        scipy.stats distribution
        """
        
        self.machines = machines
        self.service_time_distributions = []  # service_time_distributions
        self.compatibility = None
        self.failure_rates = []
        self.current_time_slot = 0
        self.num_time_slots = num_time_slots
        self.schedule = [[] for _ in machines]  # Schedule for each machine
        self.task_queue = []  # Priority queue for incoming tasks
        self.last_task_id = -1
        self.alpha = alpha
        self.service_lengths = [1, 2, 3, 4]
        self.slot_size = slot_size
        self.tasks = []
        self.slots_per_day = (work_end - work_start) * 60 // slot_size

        # self.duration_model = DurationModel()
        
    def update_compatibility_matrix(self, compatible_machines: list):
        """
        Updates the compatibility matrix with a row for each task. The compatibility matrix is an array row indexed by
        task and column indexed by machines with an entry of 1 where the machine is compatible and 0 if it is not.

        :param compatible_machines: list
        :return: None
        """
        if self.compatibility is None:
            self.compatibility = np.array([[1 if machine in compatible_machines else 0 for machine in self.machines]])
        else:
            self.compatibility = np.append(self.compatibility, [[1 if machine in compatible_machines else 0 for machine in self.machines]], axis=0)
    
    def add_task(self, priority: int, compatible_machines: list):
        """
        Adds a new task given a priority and what machines can service it.

        :param priority: the importance of the task, lower is better
        :param compatible_machines: list of machines that can handle the task
        :return: None
        """
        # Add a new task to the priority queue
        current_task_id = self.last_task_id + 1
        heapq.heappush(self.task_queue, (priority, current_task_id))
        self.update_compatibility_matrix(compatible_machines)
        self.last_task_id += 1

    def optimize_duration(self, task_id: int) -> int:
        """
        Finds the optimal number of blocks to schedule for the task
        :param task_id: The id of the task to schedule
        :return: opt_duration: The number of blocks that minimizes the objective function
        """

        opt_duration = 60 // self.slot_size
        opt_z = np.inf  # the weighted expected value for the best duration

        # find the service time that optimizes the weighted sum of idle and wait time
        for duration in self.service_lengths:

            allotted_service_time = self.slot_size * duration
            dist = self.service_time_distributions[task_id]
            z = calculate_mean_weighted_absolute_error(allotted_service_time, dist, self.alpha)

            if z < opt_z:
                opt_duration = duration
                opt_z = z
        return opt_duration

    def find_available_slot(self, machine_id: int, task_id: int) -> tuple[int, int] | tuple[None, None]:
        """
        Finds the duration of the appointment by optimizing the tradeoff between machine idle time and part times given
        the current alpha

        :param machine_id: the id of the machine servicing the task
        :param task_id: the id of the task being scheduled
        :return: start_time, end_time_slot
        """
        # Find the first available time slot on a given machine for the new task
        for start_time in range(self.num_time_slots):
            if self.is_time_slot_available(machine_id, start_time, task_id):
                opt_duration = self.optimize_duration(task_id)
                end_time_slot = start_time + opt_duration
                if end_time_slot <= self.num_time_slots:
                    return start_time, end_time_slot
        return None, None
    
    def is_time_slot_available(self, machine_id: int, start_time: int, end_time_slot: int) -> bool:
        """
        Checks whether a machine is available at a certain time

        :param machine_id: the id of the machine servicing the task
        :param start_time: the beginning of the service
        :param end_time_slot: the end of the service
        :return: bool
        """
        # Check if a time slot is available for the new task

        # Check if the start and end times conflict with existing tasks
        for scheduled_task in self.schedule[machine_id]:
            scheduled_start, scheduled_end = scheduled_task.start_time_slot, scheduled_task.end_time_slot
            if not (end_time_slot <= scheduled_start or start_time >= scheduled_end):
                return False
        return True

    def find_start_of_day_time_slot(self, current_time_slot: int) -> int:
        """
        Finds the time slot that began the day for the current time slot. Used to find the expected time of appointments
        since the start of day.

        :param current_time_slot: The current time slot
        :return: start_of_day_time_slot: The time slot that started the day of the current time slot
        """
        start_of_day_time_slot = current_time_slot // self.slots_per_day * self.slots_per_day
        return start_of_day_time_slot

    def find_optimal_start_time(self, machine):
        """

        :param machine:
        :return:
        """

        if len(self.schedule[machine]) > 0:
            current_time_slot = self.schedule[machine][-1].end_time_slot

        else:
            current_time_slot = 0
        start_of_day_time_slot = self.find_start_of_day_time_slot(current_time_slot)

        if start_of_day_time_slot == current_time_slot:
            opt_start = current_time_slot
        else:
            day_appointment_durations = [scheduled_item.task.distribution_samples for scheduled_item in
                                         self.schedule[machine] if scheduled_item.start_time_slot >= start_of_day_time_slot]

            if len(day_appointment_durations) > 0:
                agg_dist_samples = np.sum(np.vstack(day_appointment_durations), axis=0)
                opt_z = np.inf
                opt_start = 0
                for i in range(self.slots_per_day):
                    z = calculate_mean_weighted_absolute_error(i * self.slot_size, agg_dist_samples, alpha=self.alpha)
                    if z < opt_z:
                        opt_z = z
                        opt_start = start_of_day_time_slot + i

            else:
                opt_start = start_of_day_time_slot
        return opt_start
    
    def schedule_task(self, task_id: int):

        best_schedule = None
        earliest_start = np.inf
        
        # Find machine with earliest available appointment
        for machine_id in range(len(self.machines)):

            if self.compatibility[task_id, machine_id] == 0:
                continue
            start_time_slot = self.find_optimal_start_time(machine_id)
            end_time_slot = start_time_slot + self.optimize_duration(task_id)

            if start_time_slot < earliest_start:
                earliest_start = start_time_slot
                best_schedule = (task_id, machine_id, start_time_slot, end_time_slot)

        #
        if best_schedule:
            task_id, machine_id, start_time_slot, end_time_slot = best_schedule

            self.schedule[machine_id].append(ScheduledItem(task_id, self.tasks[task_id], start_time_slot, end_time_slot))
            self.current_time_slot = max(self.current_time_slot, end_time_slot)
            self.current_time_slot = max(self.current_time_slot, end_time_slot)
            print(f"Scheduled Task {task_id} on Machine {machine_id} from {start_time_slot} to {end_time_slot}")
        # else:
        #     print(f"Unable to schedule {task_id}")

    def handle_new_task(self, task: Task):
        """
        Adds a new task to the schedule
        :param task: instance of Task dataclass
        :return: None
        """
        # When a new task arrives

        # 1 - shows up, 0 - no show or cancellation
        failures = np.random.binomial(1, 1 - task.failure_rate, len(task.distribution_samples))

        # modify service time distribution to account for failure rate
        self.service_time_distributions.append(list(failures * np.array(task.distribution_samples)))
        self.failure_rates.append(task.failure_rate)
        self.add_task(task.priority, task.compatible_machines)
        self.tasks.append(task)
        self.schedule_tasks()
        # print(self.schedule)
    
    def schedule_tasks(self):
        """
        Schedule each task in the queue
        :return:
        """
        while self.task_queue:
            _, task_id = heapq.heappop(self.task_queue)
            self.schedule_task(task_id)

    def dump_schedule(self):
        """
        Dump the schedule into a file
        :return:
        """
        # {machine: [[task id, start time slot, end time slot], ...], ...}
        d = {i: self.schedule[i] for i in range(len(self.schedule))}
        return d


def get_working_time(start_minutes, initial_time, work_start, work_end, time_slot_duration):
    """
    Adjusts time calculation to skip non-working hours

    :param start_minutes:
    :param initial_time:
    :param work_start:
    :param work_end:
    :param time_slot_duration:
    :return:
    """
    total_minutes = 0
    current_time = initial_time
    while total_minutes < start_minutes:
        # If within working hours, add time slot duration
        if work_start <= current_time.hour < work_end:
            total_minutes += time_slot_duration
        # Move to the next time slot
        current_time += timedelta(minutes=time_slot_duration)

    return current_time


def convert_to_dataframe(machine_data: dict[int: ScheduledItem], time_slot_duration: int, initial_time, work_start=settings['work_start'],
                         work_end=settings['work_end']):
    """
    Converts a machine-task dictionary into a Pandas DataFrame with actual timestamps.

    Parameters:
    - machine_data: dict {machine: [[task_id, start_time_slot, end_time_slot], ...]}
    - time_slot_duration: int, duration of each time slot in minutes
    - initial_time: datetime, the reference start time
    - work_start: int, starting hour of the working day (default: 9 AM)
    - work_end: int, ending hour of the working day (default: 5 PM)

    Returns:
    - Pandas DataFrame with columns: ['machine', 'task_id', 'start_time', 'end_time']
    """
    slots_per_day = (work_end - work_start) * 60 // time_slot_duration  # Total slots per working day
    rows = []

    for machine, scheduled_items in machine_data.items():

        # for task_id, start_slot, end_slot in tasks:
        for item in scheduled_items:
            task_id = item.task_id
            start_slot = item.start_time_slot
            end_slot = item.end_time_slot
            # Determine the workday index and in-day slot position
            start_day = start_slot // slots_per_day  # Which day the task falls on
            start_in_day_slot = start_slot % slots_per_day  # Slot within that day

            # end_day = end_slot // slots_per_day
            # end_in_day_slot = end_slot % slots_per_day

            # Compute actual start and end times
            start_time = initial_time + timedelta(days=start_day)
            start_time = start_time.replace(hour=work_start, minute=0) + timedelta(
                minutes=start_in_day_slot * time_slot_duration)

            # calculate appointment length (end slot inclusive)
            appointment_length = (end_slot-start_slot+1) * time_slot_duration
            end_time = start_time + timedelta(minutes=appointment_length)
            # end_time = initial_time + timedelta(days=end_day)
            # end_time = end_time.replace(hour=work_start, minute=0) + timedelta(
            #     minutes=end_in_day_slot * time_slot_duration)

            rows.append([machine, task_id, start_time, end_time])

    return pd.DataFrame(rows, columns=['machine', 'task_id', 'start_time', 'end_time'])


def schedule_model(appointment_df, condition_cols=['Month', 'DayOfWeek', 'WorkingDay', 'AM_PM', 'Gender']):
    """
    Schedules appointments based on a trained duration model.

    Args:
        appointment_df (pd.DataFrame): A DataFrame containing appointment data.
            - Required columns:
                - 'task_id': Unique identifier for each appointment.
                - 'machine': The machine assigned to the appointment.
                - Additional columns needed for estimating appointment durations (discrete-valued, either int or string dtype).

    Returns:
        pd.DataFrame: A DataFrame with scheduled appointment times and any additional scheduling details
        Includes columns 'start_time' and 'end_time'


    Notes:
        - The function uses `trained_duration_model` to predict appointment durations based on the provided features.
        - It then schedules the appointments accordingly using the TaskScheduler Engine to handle constraints
    """

    duration_model = DurationModel()

    X = appointment_df[condition_cols]
    y = appointment_df['duration']
    duration_model.train(X, y)

    # condition_cols = [col for col in appointment_df.columns if col not in ['appointment_id', 'machine']]
    machines = appointment_df['machine'].unique()
    num_blocks = appointment_df['task_id'].max() * 8
    scheduler = TaskScheduler(machines, num_blocks)

    for i, row in appointment_df.iterrows():

        compatible_machines = list(range(6))
        sample_durations = duration_model.predict(tuple(row[condition_cols]))
        no_show_rate = settings['no_show_rate']
        task = Task(str(row['task_id']), compatible_machines, list(sample_durations), no_show_rate, 1)
        scheduler.handle_new_task(task)

    schedule = convert_to_dataframe(scheduler.dump_schedule(), scheduler.slot_size, pd.to_datetime(appointment_df['date'].iloc[0]))
    # print((schedule['end_time'] - schedule['start_time']).unique())
    schedule.to_csv('sample_schedule.csv')
    return schedule


if __name__ == "__main__":

    appointment_df = pd.read_csv('appointment_duration_data (2).csv').rename(columns={'ProviderID': 'machine', 'ServTime': 'duration'})[:300]
    appointment_df['appointment_id'] = appointment_df.index
    print(appointment_df)

    appointment_df['duration'] /= 60
    print(appointment_df['duration'])
    print(schedule_model(appointment_df))
