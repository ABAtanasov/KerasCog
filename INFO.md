#Guidelines for Naming Convention for Consistency across Models


I figured it would be good to have a file where we explicitly declare all of the notation we use throughout our code and future write-ups in terms of what variable names signify. This is always one of the more confusing parts of computational neuroscience packages in general, so hopefully this will improve user experience.

If anyone thinks that there is a better naming convention, feel free to directly modify this .md

For each new model we make lets add it to this .md. Here is the current status:


##XOR on Two Inputs:
	Two input neurons,  A    and B
	Two output neurons, SAME and DIFF
	
	X and Y are random variables that can each be either A or B
	SAME fires if X xor Y == 1
	DIFF fires if X xor Y == 0
	
	There is an initial input_wait time interval.
	Afterwards an interval is either STIMULUS
	for a set time stim_dur or mem_gap, respectively.

	Timeline: input_wait | stim_dur | mem_gap | stim_dur | out_gap
	Input: 	  0000000000 | XXXXXXXX | 0000000 | YYYYYYYY | 0000000
	Output:   	          ****** Masked ******			 | ZZZZZZZ
	----------------------------seq_dur---------------------------

##Flip Flop Test by Inputs:
	Two input neurons,  A  and B
	Two output neurons, A' and B'
	
	There is an initial input_wait time interval.
	Afterwards an interval is either STIMULUS or QUIET
	for a set time stim_dur or quiet_dur, respectively.
	
	Timeline: input_wait | stim_dur | quiet_gap | stim_dur | quiet_gap  ...
	Input: 	  0000000000 | XXXXXXXX | 000000000 | YYYYYYYY | 000000000  ...
	Output:   	   ** Masked ** 	| XXXXXXXXX | *Masked* | YYYYYYYYY  ... 
	-----------------------------------seq_dur-------------------------------
	
	This is repeated n_rounds times. 

##Flip Flop Test by Magnitude
	One input neuron,  firing either HI or LO
	One output neuron, firing either HI or LO
	
	Everything is as in the prior case, with X and Y random variables representing either HI or LO.
	
	
	