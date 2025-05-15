# Healthcare Appointment Scheduling Optimization
## Overview
This project implements a bilateral weighted scheduling optimization model with probabilistic start time estimation to improve appointment scheduling efficiency in a healthcare settings. The model dynamically schedules appointments based on historical appointment durations, provider availability, and no-show probabilities to reduce makespan (time to first available appointment) and increase provider utilization.
## Key Features
* Data-driven duration estimation using Kernel Density Estimation (KDE)
* Weighted Optimization based on relative value of provider vs. patient time
* Stochastic simulation engine to evaluate and benchmark schedules
* Sensitivity Analysis for optimizing block sizes and scheduling priorities
## Architecture
The system has three core components:
### 1. Duration Model
* Uses KDE to model appointment durations based on appointment type and provider.
* Accounts for on-shows by incorporating a binomial random variable
* Provides a probability density function of expected durations for each appointment
### 2. Scheduling Engine
* Builds a schedule that minimizes the weighted mean absolute error based on a sample of predicted durations from the duration model
* Optimizes for provider utilization by prioritizing provider time using a tunable α parameter, which represents the ratio of the importance of the provider to patient time.
* Supports scheduling at varying block sizes (the granularity with which appointments can be scheduled)
* Output: A CSV file of scheduled appointments including start time and provider.
### 3. Simulator
* Simulates the execution of the proposed schedule using actual appointment durations and no-show outcomes.
* Calculates performance metrics:
    * Makespan (the time from start of first task to the end of the last)
    * Utilization (percentage of provider time used)
    * Average patient wait time
## Sensitivity Analysis 
* Varying α controls the trade-off between provider idle time and patient wait time.
* Smaller block sizes (e.g., 1 minute) increase schedule granularity and precision but may be harder to implement operationally.
## Results Summary
| Model                            | Utilization | Makespan (min) | Avg. Wait Time (min) |
| -------------------------------- | ----------- | -------------- | -------------------- |
| Baseline (18-min slots)          | 77%         | 26,385         | 5.4                  |
| Optimized (α=2.9, 1-min blocks)  | 93.5%       | 21,708         | 14.7                 |
| Optimized (α=2.5, 10-min blocks) | 92.98%      | 21,926         | 14.4                 |

* Utilization improved by 21%, while maintaining reasonable wait times.

* Makespan reduced by ~17.7%, enabling quicker access to care.
## Considerations
* Full implementation would require deployment of the KDE-based prediction model and operational changes to allow finer block scheduling.
* Model generalizes well and can be adapted for what-if analyses or multiple locations/providers.
## Authors
This project was developed as part of a capstone at Liberty University's Industrial & Systems Engineering program
