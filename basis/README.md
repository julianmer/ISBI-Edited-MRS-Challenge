The contained metabolite bases were simulated with FID-A (run_simMegaPressShapedEdit.m) and following parameters:
 

% ************INPUT PARAMETERS**********************************
editWaveform='sampleEditPulse.pta'; %name of editing pulse waveform.
editOnFreq=1.88; %freqeucny of edit on pulse[ppm]
editOffFreq=7.4; %frequency of edit off pulse[ppm]
editTp=20; %duration of editing pulses[ms]
Npts=2048; %number of spectral points
sw=2000; %spectral width [Hz]
Bfield=3; %magnetic field strength [Tesla]
lw=2; %linewidth of the output spectrum [Hz]
taus=[5,... %Time from excitation to 1st refoc pulse [ms]
    17,...  %Time from 1st refoc pulse to 1st editing pulse [ms]
    17,...  %Time from 1st editing pulse to 2nd refoc pulse [ms]
    17,...  %Time from 2nd refoc pusle to 2nd editing pulse [ms]
    12];    %Time from 2nd editing pulse to ADC onset [ms]
spinSys='GABA'; %spin system to simulate
centreFreq=3.0; %Centre Frequency of MR spectrum [ppm];
editPhCyc1=[0 90]; %phase cycling steps for 1st editing pulse [degrees]
editPhCyc2=[0 90 180 270]; %phase cycling steps for 2nd editing pulse [degrees]
% ************END OF INPUT PARAMETERS**********************************
