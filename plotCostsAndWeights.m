clear; clc; clf;

addpath('results/');

% -------------------------------- FILE NAME EXTRACTION --------------------------------

% number of scatter points
scatterPoints = input('Number of Scatter Points: ', 's');
if length(scatterPoints) < 2
    disp('Number of scatter points must be be formatted using 2 or more digits (i.e. 02 for 2)');
    return
end

% number of time units
timeUnits = input('Number of Time Units: ', 's');
if length(scatterPoints) < 2
    disp('Number of time units must be be formatted using 2 or more digits (i.e. 02 for 2)');
    return
end

% concatenate string for filename using scatterPoints and timeUnits
unmaskedFileName = strcat('Unmasked Losses_', scatterPoints, '_T', timeUnits, '_all.csv')
maskedFileName = strcat('Masked Losses_', scatterPoints, '_T', timeUnits, '_all.csv')




% -------------------------------- DATA READ --------------------------------

% read in Unmasked Network's CSV data (epoch #, loss, W1, W2, W3, ... , Wn)
unmaskedData = csvread(unmaskedFileName,1,1);
maskedData = csvread(maskedFileName,1,1);

epochs = unmaskedData(:,1);                     % extract epoch column

unmaskedLosses = unmaskedData(:,2);             % least-squares loss @ said epoch
maskedLosses = maskedData(:,2);

unmaskedWeightMatrix = unmaskedData(:,3:end);   % Network weights @ said epoch (each row corresponds to an epoch)
maskedWeightMatrix = maskedData(:,3:end);


% -------------------------------- PLOT TRAINING LOSS PER EPOCH --------------------------------
lossFig = figure(1);

% plotting range for loss
lossrange = 1:length(epochs);

% regular scale (loss vs. epoch)
subplot(3,1,1);
plot(epochs(lossrange), unmaskedLosses(lossrange), 'DisplayName', 'normal');
hold on;
plot(epochs(lossrange), maskedLosses(lossrange), 'DisplayName', 'masked');
hold off;
ylabel('normal scale');
legend('show');
grid on;

% y-log scale (loss-log vs. epoch)
subplot(3,1,2);
semilogy(epochs(lossrange), unmaskedLosses(lossrange), 'DisplayName', 'normal');
hold on;
semilogy(epochs(lossrange), maskedLosses(lossrange), 'DisplayName', 'masked');
hold off;
ylabel('log-y scale');
legend('show');
grid on;

% log-log scale (loss-log vs. epoch-log)
subplot(3,1,3);
loglog(epochs(lossrange), unmaskedLosses(lossrange), 'DisplayName', 'normal');
hold on;
loglog(epochs(lossrange), maskedLosses(lossrange), 'DisplayName', 'masked');
hold off;
ylabel('log-log scale');
grid on;
legend('show');

xlabel('epoch');
suptitle('Training Loss');



% -------------------------------- PLOT WEIGHT CHANGE THROUGH TRAINING --------------------------------
unmaskedWeightFig = figure(2);
start = 9;
step = 10;
N = 401;
epochWeightRange = start:step:N;   % get range of row indices for weights to plot

[m,n] = size(unmaskedWeightMatrix);

% plot the weights after epoch 1
plot(1:n, unmaskedWeightMatrix(1,:));
% plot at subsequent epochs until N
hold on;
for k = epochWeightRange
    plot(1:n, unmaskedWeightMatrix(k,:));
end
hold off;
title(strcat('Weights from Epochs (Normal Version) ', ' ', num2str(epochs(epochWeightRange(1))), '-', num2str(epochs(epochWeightRange(end))) ))

maskedWeightFig = figure(3);
% plot the weights after epoch 1
plot(1:n, maskedWeightMatrix(1,:));
% plot at subsequent epochs until N
hold on;
for k = epochWeightRange
    plot(1:n, maskedWeightMatrix(k,:));
end
hold off;
title(strcat('Weights from Epochs (Masked Version) ', ' ', num2str(epochs(epochWeightRange(1))), '-', num2str(epochs(epochWeightRange(end))) ));

saveas(lossFig, strcat('results/Training_Loss_' , scatterPoints, '_T', timeUnits,'_all.pdf'));
saveas(unmaskedWeightFig, strcat('results/Unmasked_Training_Weights_' , scatterPoints, '_T', timeUnits,'_all.pdf'));
saveas(unmaskedWeightFig, strcat('results/Masked_Training_Weights_' , scatterPoints, '_T', timeUnits,'_all.pdf'));
