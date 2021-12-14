# BET_MT_model_Identification
A set of Notebooks dedicated to propeller model identification 

Scripts are named in the same order they're supposed to be used.

The pipeline takes a .ulg file as an input.
The file shoud be located in ./logs/target.

Use 0_0_log_class.py to:
	- parse it using log2csv, output csv are stored in ./logs/target/donnees_brut
	- preprocess and reinterpolate so that all datas are at the same timestamp
	- save the data to ./logs/target/log_real.csv

Due to PX4 changing the topics, it is advised to check that all processed columns correspond to what they're supposed to.

Once the log_real.csv is created, feel free to remove the first and last rows (preserve the header) to clip take off and landing phases.

Then use 0_5_process_log_real to generate body vel / acc , that will generate log_real_processed.csv next to log_real.csv

If interrested, use 1_0_compute_dataset_variance to estimate the variance of the acceleration from the dataset with a Kalman filter.


Next, fire up 2_id_BEM to proceed with model identification.
In this script, identification is processed as a two parts process, using various hypothesis.

	-Axes per axes identifications will be summed up in bilan.csv, generated in the current directory.
	-Global identif used the whole dataset. A lot of hypothesis are tested, computation can be parallelized to be speeded up. Output will be saved to results.
	-Global identif also has the code for speed error minimisation based identification. This takes MUCH longer. Parallelization is also recommended. Output is saved in results_speed directory.
