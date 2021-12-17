# BET_MT_model_Identification
A set of Notebooks dedicated to propeller model identification 

Scripts are named in the same order they're supposed to be used.

The pipeline takes a .ulg file as an input.
The file shoud be located in ./logs/target.

To be used, the scripts expect a .csv file located in ./logs/target, named "log_real.csv"

Its columns should be:

	* the timestamps, with column name "t"
	* the drone's center of mass speed relative to NED frame, named speed[0],speed[1],speed[2]
	* the drone's orientation, modelled via a quaternion, the columns' names should be q[0],q[1],q[2],q[3]
	* the PWM sent to the motors, with columns names PWM_motor[1],...,PWM_motor[max_number_of_rotor]

If working with a raw .ulg file, the 0_0_log_class.py may help to parse it to produce the log_real.csv, but its results may depend on the PX4 version/mixer/sensor suite used to produce the .ulg file. Thus, it may be more safe to generate the log_real.csv yourself before using the scripts.

If using 0_0_log_class.py, you should put the .ulg file into ./logs/target:
	- parse it using log2csv, output csv are stored in ./logs/target/donnees_brut
	- preprocess and reinterpolate so that all datas are at the same timestamp
	- save the data to ./logs/target/log_real.csv

Once you have log_real.csv, it is advised to remove the first and last rows (preserve the header) to clip take off and landing phases.

Then use <b>0_5_process_log_real</b> to generate body vel / acc. This will generate log_real_processed.csv next to log_real.csv. This step is mandatory.

If interrested, use 1_0_compute_dataset_variance to estimate the variance of the acceleration from the dataset with a Kalman filter, but this is optionnal.


Then:
	-2_id_BEM_preliminary proceeds with model identification axis per axis 
	-3_id_BEM_acc_parallel proceeds with model identification on all axes, using a various set of hypothesis. The results are stored in ./results/acc
	-4_id_BEM_speed_parallel proceeds with model identification on all axes via speed prediction, using a various set of hypothesis. The results are stored in ./results/speed. The speed prediction is performed on a subsample of the data, which may be tweaked with the "nsecs" parameter. nsecs='all' will use all the dataset for speed prediction, where as nsecs=1 will use one second long batches for speed prediction, which speeds up the optimisation but generates suboptimal coefficients. Identification with the speed if MUCH LONGER than with acceleration.

For the two last scripts, parallelization is implemented to speed up calculations.

