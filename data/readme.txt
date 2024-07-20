#######################################################################################################################
# Format
#######################################################################################################################
Data can be either supplied using '.xlsx', '.csv', '.mat', or '.pkl' files. Additionally, the '.h5' files converted with
nilmtk can be used directly as input for the BaseNILM toolkit, but this option is not fully supported. To create a new
dataset please see the attached templates. The input (aggregated signals) and output (device signals) data must be 2D
or 3D tensors with the following shape:

	1) Input Tx(F+2) (samples (T) times number of input features (F), with the first column being time and second
	   column being the id.
	2) Output Tx(D+2)xF (samples (T) times number of devices (D) time features (F), with the first column being 
	   time and the second column being the id.


#######################################################################################################################
# Templates
#######################################################################################################################
For each of the options to load data a respective template is provided the filled with example input features and device
level power consumption values. To make usage more intuitive the redd2 data set is provided in all 5 versions of the
dataformat. When creating a new dataset these templates can be used as a starting point. Not all templates have full
functionality, please see the restrictions below:

1) XLSX (dataTemplate.xlsx): Cannot be used for 3D output features, e.g. as in AMPDs where for each appliance several
                             features are available

2) CSC (dataTemplate.csv):   Cannot be used for 3D output features, e.g. as in AMPDs where for each appliance several
                             features are available

3) MAT (dataTemplate.mat):   No restrictions

4) PKL (dataTemplate.pkl):   No restrictions

5) H5 (dataTemplate.h5):     Not fully support many nilmtk converted datasets do work though. Manually creating datasets
                             is not supported here.


#######################################################################################################################
# Description Datasets
#######################################################################################################################
# ==============================================================================
# AMPds2
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
- Status:               completed, checked
- Reference:            [1]
- Location:             Canada
- Year:			        2012 - 2014
- Duration: 		    2 years
- House:		        1
- Sampling: 		    60 sec
- Samples:		        1048575
- Appliances: 		    20
- Deferrable Loads:     dishwasher (DWE), fridge (FRE), heat pump (HPE), wall oven (WOE), clothe dryer (CDE)
- Transferable Loads:   -
- Dimensionality:	    3D
- Quality:		        clean, no measurement gaps
- Pre-processing:	    none
- Input Features:	    Active Power (W), Current (A), Reactive Power (VAr), Apparent Power (VA)
- Output Features:	    Active Power (W), Current (A), Reactive Power (VAr), Apparent Power (VA)
- Notes:                -

# ------------------------------------------
# Input
# ------------------------------------------
1) Active Power (W):		P-agg
2) Current (A):			    I-agg
3) Reactive Power (VAr):	Q-agg
4) Apparent Power (VA):		S-agg

# ------------------------------------------
# Output (repeated from [1])
# ------------------------------------------
1)  RSE: The basement rental suite sub-meter which has a line voltage of 240V and has a 100A double-pole breaker.
2)  GRE: The detached garage sub-meter which has a line voltage of 240 V and has a 60A double-pole breaker.
3)  MHE: The main house soft-meter which is calculated by MHE=WHE—(RSE+GRE).
4)  B1E: Sub-meter for the north bedroom plugs and lights which has a line voltage of 120V and has a 15A single-pole breaker.
5)  B2E: Sub-meter for the master and south bedroom plugs and lights which has a line voltage of 120 V and has a 15A single-pole breaker.
6)  BME: Sub-meter for the some of the basement plugs and lights which has a line voltage of 120 V and has a 15A single-pole breaker. A small freezer is powered by this breaker.
7)  CWE: Sub-meter for the front loading clothes washer which has a line voltage of 120 V and has a 15A single-pole breaker.
8)  DWE: Sub-meter for the kitchen dishwasher which has a line voltage of 120 V and has a 15A single-pole breaker.
9)  EQE: Sub-meter for the security and network equipment which has a line voltage of 120 V and has a 15A single-pole breaker.
10) FRE: Sub-meter for the forced air furnace fan and thermostat which has a line voltage of 120 V and has a 15A single-pole breaker.
11) HPE: Sub-meter for the heat pump which has a line voltage of 240 V and has a 40A double-pole breaker.
12) OFE: Sub-meter for the home office lights and plugs which has a line voltage of 120 V and has a 15A single-pole breaker.
13) UTE: Sub-meter for the utility room plug which has a line voltage of 120 V and has a 15A single-pole breaker.
14) WOE: Sub-meter for the kitchen convection wall oven which has a line voltage of 240 V and has a 30A double-pole breaker.
15) CDE: Sub-meter for the clothes dryer which has a line voltage of 240 V and has a 30A double-pole breaker.
16) DNE: Sub-meter for the dining room plugs which has a line voltage of 120 V and has a 15A single-pole breaker.
17) EBE: Sub-meter for the electronics workbench which has a line voltage of 120 V and has a 15A single-pole breaker.
18) FGE: Sub-meter for the kitchen fridge which has a line voltage of 120 V and has a 15A single-pole breaker.
19) HTE: Sub-meter for the instant hot water unit which has a line voltage of 120 V and has a 15A single-pole breaker.
20) OUE: Sub-meter for the outside plug which has a line voltage of 120 V and has a 15A single-pole breaker.
21) TVE: Sub-meter for the entertainment equipment (TV, PVR, amplifier, and Blu-Ray) which has a line voltage of 120 V and has a 15A single-pole breaker.
22) UNE: The unmetered soft-meter amount with is calculated by UNE=MHE—sum(all sub-meters under MHE).


# ==============================================================================
# REDD
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
- Status:               completed, not checked
- Reference:            [2]
- Location:		        US
- Year:			        2011
- Duration: 		    2-4 weeks
- Houses:		        6
- Sampling: 		    3 sec + HF data (15 kHz) for REDD-3 and REDD-5
- Samples:		        104177 - 544886
- Appliances: 		    9-24
- Deferrable Loads:     refrigerator (FRE), lighting (LIx), dishwasher (DWE), microwave (MIC), and furnace (FUR)
- Transferable Loads:   refrigerator (houses: 1,2,3,5,6), dishwasher (houses: all), washer dryer (houses: all)
- Dimensionality:	    2D (LF data) and 3D (HF data)
- Quality:		        measurement gaps and NaNs
- Pre-processing:	    data converted using nilmtk, NaNs removed (whole column is removed if there is any NaN value)
- Input Features:	    Active Power (W)
- Output Features:	    Active Power (W)
- Notes:                There are additional files with 1 min sampling resolutions, in these only NaNs in the main
                        metering have been removed after conversion with nilmtk. Also metering groups have not been
                        merged. Remaining NaNs in the data have been replaced with zeros. Also the unix timestamp has
                        been preserved.

# ------------------------------------------
# Input
# ------------------------------------------
# LF-Data
1) Active Power Total (W):	    P-agg
2) Active Power Phase-1 (W):	P1-agg
3) Active Power Phase-2 (W):	P2-agg

# HF-Data
1) Current (A):			I-agg
2) Voltage (V):			V-agg

# ------------------------------------------
# Output
# ------------------------------------------
# House-1
1)  OVE:	oven
2)  FRE:	refrigerator
3)  DWE:	dishwasher
4)  SO1:	kitchen_outlets
5)  SO2:	kitchen_outlets
6)  LI1:	lighting
7)  WAD:	washer_dryer
8)  MIC:	microwave
9)  BAT:	bathroom_gfi
10) ESH:	electric_heat
11) EST:	stove
12) SO3:	kitchen_outlets
13) SO4:	kitchen_outlets
14) LI2:	lighting
15) LI3:	lighting

# House-2
1) SO1:		kitchen_outlets
2) LI1:		lighting
3) EST:		stove
4) MIC:		microwave
5) WAD:		washer_dryer
6) SO2:		kitchen_outlets
7) FRE:		refrigerator
8) DWE:		dishwasher
9) WDU:		disposal

# House-3
1)  SO1:	outlets_unknown
2)  SO2:	outlets_unknown
3)  LI1:	lighting
4)  ELE:	electronics
5)  FRE:	refrigerator
6)  WDU:	disposal
7)  DWE:	dishwasher
8)  FUR:	furnace
9)  LI2:	lighting
10) SO3:	outlets_unknown
11) WAD:	washer_dryer
12) LI3:	lighting
13) MIC:	microwave
14) LI4:	lighting
15) SMO:	smoke_alarms
16) LI5:	lighting
17) BAT:	bathroom_gfi
18) SO4:	kitchen_outlets
19) SO5:	kitchen_outlets

# House-4
1)  LI1:	lighting
2)  FUR:	furnace
3)  SO1:	kitchen_outlets
4)  SO2:	outlets_unknown
5)  WAD:	washer_dryer
6)  EST:	stove
7)  AC1:	air_conditioning
8)  MIS:	miscellaneous
9)  SMO:	smoke_alarms
10) LI2:	lighting
11) SO3:	kitchen_outlets
12) DWE:	dishwasher
13) BAT1:	bathroom_gfi
14) BAT2:	bathroom_gfi
15) LI3:	lighting
16) LI4:	lighting
17) AC2;	air_conditioning

# House-5
1)  MIC:	microwave
2)  LI1:	lighting
3)  SO1:	outlets_unknown
4)  FUR:	furnace
5)  SO2:	outlets_unknown
6)  WAD:	washer_dryer
7)  SU1:	subpanel
8)  SU2:	subpanel
9)  ESH:	electric_heat
10) LI2:	lighting
11) SO3:	outlets_unknown
12) BAT:	bathroom_gfi
13) LI3:	lighting
14) FRE:	refrigerator
15) LI4:	lighting
16) DWE:	dishwasher
17) WDU:	disposal
18) ELE:	electronics
19) LI5:	lighting
20) SO4:	kitchen_outlets
21) SO5:	kitchen_outlets
22) SO6:	outdoor_outlets

# House-6
1)  SO1:	kitchen_outlets
2)  WAD:	washer_dryer
3)  EST:	stove
4)  ELE:	electronics
5)  BAT:	bathroom_gfi
6)  FRE:	refrigerator
7)  DWE:	dishwasher
8)  SO2:	outlets_unknown
9)  SO3:	outlets_unknown
10) ESH:	electric_heat
11) SO4:	kitchen_outlets
12) LI1:	lighting
13) AC1:	air_conditioning
14) AC2:	air_conditioning


# ==============================================================================
# UKDALE
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
- Reference:            [3]
- Location:		        UK
- Year:			        2014 - 2017
- Duration: 		    2-4 weeks
- Houses:		        5
- Sampling: 		    6 sec
- Samples:		        1520000 - 90438000
- Appliances: 		    5-40
- Deferrable Loads:     -
- Transferable Loads:   kettle, microwave, dishwasher, fridge, washing machine
- Dimensionality:	    2D
- Quality:		        measurement gaps and NaNs
- Pre-processing:	    data converted using nilmtk, NaNs removed and replaced by zeros
- Input Features:	    Active Power (W)
- Output Features:	    Active Power (W)
- Notes:                Due to the size of house 1, the focus has been on only five appliance

# ------------------------------------------
# Input
# ------------------------------------------
1) Active Power Total (W):	    P-agg

# ------------------------------------------
# Output
# ------------------------------------------
Due to the size of the dataset only five appliances have been used (please note that they are not always present in
each house). The unit of all output values are Watt. The following five appliances have been used.
1) KET:     kettle
2) MIC:     microwave
3) DWE:     dishwasher
4) FRE:     fridge
5) WME:     washing machine


# ==============================================================================
# REFIT
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
- Reference:            [4]
- Location:		        UK
- Year:			        2013 - 2015
- Duration: 		    2 years
- Houses:		        20
- Sampling: 		    8 sec
- Samples:		        503474 - 849364
- Appliances: 		    9
- Deferrable Loads:     -
- Transferable Loads:   TV (except house: 11), Washing Machine (except house: 12), Microwave (except house: 1, 7, 16, 21)
                        Kettle (except houses: 1, 10, 16, 18), Dishwasher (except houses: 4, 8, 12, 17, 19)
- Dimensionality:	    2D
- Quality:		        measurement gaps and NaNs
- Pre-processing:	    data converted using nilmtk, NaNs removed (whole column is removed if there is any NaN value)
- Input Features:	    Active Power (W)
- Output Features:	    Active Power (W)
- Notes:                Data down-sampled to 1 min resolution all NaNs removed

# ------------------------------------------
# Input
# ------------------------------------------
1) Active Power Total (W):	    P-agg

# ------------------------------------------
# Output
# ------------------------------------------
Each house contains exactly nine devices, which all have the output unit Watt (W). Since the number of houses is larger
only the different appliances and their labels are listed here:
1)  TVE:    Television
2)  HIF:    Hi-Fi
3)  FFE:    Fridge-Freezer
4)  FRE:    Fridge
5)  FRZ:    Freezer
6)  MIC:    Microwave
7)  COK:    Cooker
8)  KET:    Kettle
9)  TOA:    Toaster
10) ---:    Different Kitchen Appliances
11) WME:    Washing Machine
12) WAD:    Washer Dryer
13) TUD:    Tumble Dryer
14) DWE:    Dishwasher
15) CSE:    Computer
16) ROU:    Router
17) ESH:    Electric Heat
18) LAM:    Lamp
19) MIS:    Unknown
20) NAx:    Not available


# ==============================================================================
# ECO
# ==============================================================================
# ------------------------------------------
# General
# ------------------------------------------
- Reference:            [5]
- Location:		        CH
- Year:			        2013 - 2013
- Duration: 		    230 days
- Houses:		        6
- Sampling: 		    1 sec
- Samples:
- Appliances: 		    7-12
- Deferrable Loads:     -
- Transferable Loads:   Fridge, Kettle (except house 4), Entertainment (except house 1)
- Dimensionality:	    2D
- Quality:		        measurement gaps
- Pre-processing:	    data converted using nilmtk
- Input Features:	    Active Power (W), Current (A), Voltage (V), Phase Angle (deg)
- Output Features:	    Active Power (W)
- Notes:                Due to the size of the dataset it has been down-sampled to 60 sec resolution

# ------------------------------------------
# Input
# ------------------------------------------
1)  powerallphases:                 Sum of real power over all phases
2)  powerl1:                        Real power phase 1
3)  powerl2:                        Real power phase 2
4)  powerl3:                        Real power phase 3
5)  currentneutral:                 Neutral current
6)  currentl1:                      Current phase 1
7)  currentl2:                      Current phase 2
8)  currentl3:                      Current phase 3
9)  voltagel1:                      Voltage phase 1
10) voltagel2:                      Voltage phase 2
11) voltagel3:                      Voltage phase 3
12) phaseanglevoltagel2l1:          Phase shift between voltage on phase 2 and 1
13) phaseanglevoltagel3l1:          Phase shift between voltage on phase 3 and 1
14) phaseanglecurrentvoltagel1:     Phase shift between current/voltage on phase 1
15) phaseanglecurrentvoltagel2:     Phase shift between current/voltage on phase 2
16) phaseanglecurrentvoltagel3:     Phase shift between current/voltage on phase 3

# ------------------------------------------
# Output
# ------------------------------------------
# House-1
1) FRE:     Fridge (no. days: 231, coverage: 98.53%)
2) DRE:     Dryer (no. days: 231, coverage: 98.56%)
3) COM:     Coffee Maker (no. days: 113, coverage: 85.36%)
4) KET:     Kettle (no. days: 203, coverage: 77.65%)
5) WME:     Washing machine (no. days: 231, coverage: 98.56%)
6) CSE:     PC (no. days: 66, coverage: 84.77%)
7) FRZ:     Freezer (no. days: 231, coverage: 98.56%)

# House-2
1)  TAB:     Tablet (no. days: 240, coverage: 97.43%)
2)  DWE:     Dishwasher (no. days: 240, coverage: 97.09%)
3)  AIR:     Air Exhaust (no. days: 240, coverage: 96.18%) (*)
4)  FRE:     Fridge (no. days: 240, coverage: 98%)
5)  ENT:     Entertainment (no. days: 240, coverage: 96.18%) (**)
6)  FRZ:     Freezer (no. days: 240, coverage: 96.39%)
7)  KET:     Kettle (no. days: 240, coverage: 88.5%)
8)  LAM:     Lamp (no. days: 240, coverage: 90.21%) (***)
9)  LAP:     Laptops (no. days: 240, coverage: 83.36%)
10) STO:     Stove (no. days: 28, coverage: 100%) (****)
11) TV0:     TV (no. days: 240, coverage: 100%) (**)
12) STE:     Stereo (no. days: 240, coverage: 95.95%) (**)

# House-3
1) TAB:     Tablet (no. days: 104, coverage: 94.5%)
2) FRZ:     Freezer (no. days: 104, coverage: 90.71%)
3) COM:     Coffee machine (no. days: 67, coverage: 70.79%)
4) CSE:     PC (no. days: 42, coverage: 64%)
5) FRE:     Fridge (no. days: 47, coverage: 56%)
6) KET:     Kettle (no. days: 42, coverage: 67.82%)
7) ENT:     Entertainment (no. days: 48, coverage: 67.65%) (*)

# House-4
1) FRE:     Fridge (no. days: 194, coverage: 97.01%)
2) KIT:     Kitchen appliances (no. days: 194, coverage: 96.81%) (*)
3) LAM:     Lamp (no. days: 170, coverage: 93.54%) (**)
4) STE:     Stereo and laptop (no. days: 169, coverage: 90.98%)
5) FRE:     Freezer (no. days: 192, coverage: 93.08%)
6) TAB:     Tablet (no. days: 189, coverage: 93.6%)
7) ENT:     Entertainment (no. days: 186, coverage: 94.69%) (***)
8) MIC:     Microwave (no. days: 194, coverage: 97.08%)

# House-5
1) TAB:     Tablet (no. days: 218, coverage: 97.87%)
2) COM:     Coffee machine (no. days: 218, coverage: 95.16%)
3) FOU:     Fountain (no. days: 71, coverage: 99.43%) (*)
4) MIC:     Microwave (no. days: 218, coverage: 97.87%)
5) FRE:     Fridge (no. days: 218, coverage: 97.87%)
6) ENT:     Entertainment (no. days: 192, coverage: 89.14%) (**)
7) CSE:     PC (no. days: 218, coverage: 97.87%) (***)
8) KET:     Kettle (no. days: 25, coverage: 76.64%)

# House-6
1) LAM:     Lamp (no. days: 166, coverage: 67.2%)
2) LAP:     Laptop (no. days: 185, coverage: 97.3%) (*)
3) ROU:     Router (no. days: 88, coverage: 96.73%) (**)
4) COM:     Coffee machine (no. days: 179, coverage: 86.03%)
5) ENT:     Entertainment (no. days: 181, coverage: 95.86%) (***)
6) FRE:     Fridge (no. days: 179, coverage: 95.78%)
7) KET:     Kettle (no. days: 147, coverage: 82.54%)


#######################################################################################################################
# References
#######################################################################################################################
[1] Makonin, S., Ellert, B., Bajić, I. et al. Electricity, water, and natural gas consumption of a residential house in
    Canada from 2012 to 2014. Sci Data 3, 160037 (2016). https://doi.org/10.1038/sdata.2016.37
[2] J.Zico Kolter and Matthew J. Johnson. REDD: A public data set for energy disaggregation research. In proceedings of
    the SustKDD workshop on Data Mining Applications in Sustainability, 2011.
[3] Kelly, J. and Knottenbelt, W. The UK-DALE dataset, domestic appliance-level electricity demand and whole-house
    demand from five UK homes. Sci. Data 2:150007 doi: 10.1038/sdata.2015.7
[4] Murray, D., Stankovic, L. & Stankovic, V. An electrical load measurements dataset of United Kingdom households from
    a two-year longitudinal study. Sci Data 4, 160122 (2017). https://doi.org/10.1038/sdata.2016.122
[5] C. Beckel, W. Kleiminger, R. Cicchetti, T. Staake, S. Santini: The ECO Data Set and the Performance of Non-Intrusive
    Load Monitoring Algorithms. Proceedings of the 1st ACM International Conference on Embedded Systems for
    Energy-Efficient Buildings (BuildSys 2014). Memphis, TN, USA. ACM, November 2014.