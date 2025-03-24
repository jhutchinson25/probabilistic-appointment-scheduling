# Welcome to the Centra Scheduling Documentation


## Project Description
This site documents a capstone project completed by Devan Walhof, Kaden Skarda, Henry Thomas, 
and John Hutchinson of Liberty University for Centra in an effort to increase utilization by
improving scheduling. The project consists of several components:

### The Duration Model
The duration model estimates the length of an appointment based on historical data. 
It estimates the duration by fitting kernel density estimation on the lengths of 
appointments separated by appointment type and doctor. This allows a probabilistic 
prediction of the duration of future appointments for that appointment type and doctor 
in the future.

### The Scheduler
The scheduler uses the duration model and then assigns patients to doctors based on which 
doctors are available and any imposed constraints - i.e. patient x can only see doctor y.  
It uses a heuristic greedy algorithm of simply scheduling patients with the first available
doctor that can handle an appointment of the length estimated by the duration model.

### The Simulator
The simulator tests the scheduler and the duration model by finding their metrics on historical
data. It calculates these metrics by using a scheduling model to create a schedule and then 
finding the realizations of this schedule based on the actual durations of the appointments.
It allows testing of the viability of different scheduling models, both against each other and 
against the baseline.  

### The Dashboard
The dashboard conveys the scheduling model to the consumer. It allows users to input
data about the appointment type and receive a predicted duration and suggested appointment 
times, along with relevant reporting and descriptive statistics. 

::: scheduler
::: simulator
