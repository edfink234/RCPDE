%%Fast Fourier Transform
signal = xpeak_vals;
time = alltCM;

Ts = alltCM(2)-alltCM(1);%Sampling Period
Fs = 1/Ts;%Sampling Frequency

L = length(signal);%Length of signal

y = (fft(signal));%Fast Fourier Transform of the Signal
f = Fs/L*(0:L-1);%Frequency vector
% 
% figure
% stem(f,y)

z = (fftshift(y));
f_shift = Fs/L*(-floor(L/2):floor(L/2));

% figure
% plot(f_shift,z)

DSAS = (y/L);%Double Sided Amplitude Spectrum
SSAS = 2*DSAS(1:floor(L/2)+1);%Single Sided Amplitude Spectrum
f_ss = Fs/L*(0:floor(L/2));%Single Sided Frequency
% 
figure
stem(f_ss,abs(SSAS))
xlabel("Frequency")
ylabel("Amplitude")
title("Fourier Spectrum")

[pks, locs]=findpeaks(abs(SSAS));

% figure
% plot(locs,pks);
% 
% [pks,locs] = findpeaks(pks);
% figure 
% plot(locs,pks)

pk1 = max(pks);%First Peak

loc1 = find(pks == pk1);
index1 = locs(loc1);

[pkspks,locslocs] = findpeaks(pks);
% figure
% plot(locslocs,pkspks)

pk2 = max(pkspks);

loc2 = find(pks == pk2);
index2 = locs(loc2);

phase = angle(SSAS);
% figure 
% plot(f_ss,phase)
phase1=phase(index1);
phase2=phase(index2);









