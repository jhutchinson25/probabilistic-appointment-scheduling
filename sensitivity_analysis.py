import os
import subprocess
import sys
import numpy as np


# Define the ranges for time_slot_duration and alpha
time_slot_durations = [1] + list(np.arange(5, 60, 5))
alpha_values = [1/i for i in np.arange(1, 6, 0.1)] + [i for i in np.arange(2, 6, 0.1)]

# Loop through each combination of time_slot_duration and alpha
for time_slot_duration in time_slot_durations:
    for alpha in alpha_values:
        print(alpha, time_slot_duration)
        # Update settings.py with the current values
        settings_content = f"""settings = {{
                            'work_start': 8,
                            'work_end': 17,
                            'time_slot_duration': {time_slot_duration},
                            'alpha': {alpha},
                            'no_show_rate': 0.1
                             }}"""

        with open('settings.py', 'w') as file:
            file.write(settings_content)

        # Run simulator.py with the current settings
        subprocess.run([sys.executable, 'simulator.py'], env=os.environ.copy(), cwd=os.getcwd())
