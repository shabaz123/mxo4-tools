%% LoRa Generator
%% Rev 1.0 - shabaz - April 2023, modified from an example at the link below.
%% This MATLAB code generates a CSV file that can be used with the MXO 4 Waveform Generator
%% to transmit LoRa signals.
%% Note: You will need to prepend the CSV file contents with 43 lines of configuration, which can be 
%% copy-pasted from any waveform CSV file saved by the MXO 4.
%% The code below uses LoRaMatlab.
%% See https://uk.mathworks.com/matlabcentral/fileexchange/81166-loramatlab for credits


clear
clc

SF = 10 ;
BW = 125e3 ;
fc = 10e6 ;
Power = 14 ;

message = "Hello World!" ;

%% Sampling
Fs = 0.5*1e6 ;
Fc = 16.5e6 ;
%% Transmit Signal
signalIQ = LoRa_Tx(message,BW,SF,Power,Fs,Fc - fc) ;
Sxx = 10*log10(rms(signalIQ).^2) ;
disp(['Transmit Power   = ' num2str(Sxx) ' dBm'])
%% Plots
figure(1)
spectrogram(signalIQ,500,0,500,Fs,'yaxis','centered')
figure(2)
obw(signalIQ,Fs) ;
%% Received Signal
message_out = LoRa_Rx(signalIQ,BW,SF,2,Fs,Fc - fc) ;
%% Message Out
disp(['Message Received = ' char(message_out)])

ft=20e6;
ltime = (0:(1/ft):(1/ft)*max(size(signalIQ))-(1/ft))';
lsig = sqrt(2)*real(signalIQ.*exp(2j*pi*Fc*ltime));
M=[ltime lsig];
csvwrite('mylora.csv',M)
