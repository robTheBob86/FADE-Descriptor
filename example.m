%%---------------------------------------------------%%
% HDM05 actions classification with knn and 10-fold crossvalidation
%%---------------------------------------------------%%

%% 1. Load all the data

close all;
clear;
warning('off','all');
load('allSegmentedDataset.mat');

%% 2. Select a dataset

% group classes with same number in to one class, they are too similar. 
class = [ 1 1 1 2 2 3 3 4 5 6 7 8 8 8 8 9 10 11 12 13 14 14 14 15 15 15 16 16 16 17 17 18 18 18 18 18 19 19 19 19 20 21 21 22 22 23 23 24 24 25 25 26 27 27 28 28 29 29 30 30 31 31 32 32 33 33 34 34 35 35 36 36 37 37 37 37 37 38 38 38 38 39 40 41 42 43 43 44 44 44 44 45 45 46 47 48 49 50 51 52 53 54 55 55 56 56 57 58 59 59 59 59 60 60 61 61 62 62 62 62 63 63 63 63 64 64 64 64 65 65]; 

%% 3. Count number of samples and design low pass filter
tic

N   = 400;       
Fp  = 20;       
Fs  = 1000;       
Rp  = 0.00057565; 
Rst = 1e-4;      

filter = firceqrip(N,Fp/(Fs/2),[Rp Rst],'passedge');
%fvtool(filter,'Fs',Fs,'Color','White'); % Visualize filter

numOfSamples = zeros(1,max(class));

for i = 1:max(class)
    indexes = find(class == i);
    for j = indexes
        numOfSamples(i) = numOfSamples(i) + size(allData(j).jointAngles, 2);
    end
end

%% 3. Compute the descriptors and save the labels in labels vector

f_th = 10;  % Cut at 10Hz
f_s  = 120;  % Sampling frequency
K    = 500; % Desired dimensionality
numOfClasses = max(class);

delta_f = size(allData(1).jointAngles{1,1},2)*f_th/K; %size() = 59, formula see presentation slides

sumCounter = 0;
labels = zeros(1, sum(numOfSamples));

FADE = zeros(size(allData(1).jointAngles{1,1},2),sum(numOfSamples));
UFADE = zeros(K*size(allData(1).jointAngles{1,1},2),sum(numOfSamples)); 
for i = 1:numOfClasses
    labels(1, (sumCounter + 1):(sumCounter + numOfSamples(i))) = i;
    descriptorFADE = zeros(size(allData(1).jointAngles{1,1},2), numOfSamples(i));
    descriptorUFADE = zeros(K*size(allData(1).jointAngles{1,1},2), numOfSamples(i)); %size should return 59, no. of joint angles
    indexes = find(class == i);
    counter = 1;
    for j = indexes
        for k = 1:size(allData(j).jointAngles,2)
            
            %filtering for smoothing
            for l = 1:size(allData(j).jointAngles{1,k},2)
                size_t = conv(filter, allData(j).jointAngles{1,k}(:,l));
                allData(j).jointAngles{1,k}(1:size(allData(j).jointAngles{1,k},1),l) = size_t((N+1)/2:end-(N+1)/2);
            end
            %end of filtering
            
            rawFourier = fft(allData(j).jointAngles{1,k});
            cutOffIndex = ceil((size(rawFourier,1)*f_th)/f_s); %compare ft and dft, cuts off all frequencies above 10Hz

            % upsampled fourier transformed
            fourierTransformed = interp1(linspace(0, 20, cutOffIndex), rawFourier(1:cutOffIndex, :), linspace(0,20,K), 'pchip');

            descriptorUFADE(:,counter) = reshape(abs(fourierTransformed), [], 1);
            Vomega = pca(abs(fourierTransformed));
            descriptorFADE(:,counter) = Vomega(:,1);
            counter = counter+1;
        end
    end
    %fftMatrix(i) = {cell(1,numOfSamples(i))};
    FADE(:,(sumCounter + 1):(sumCounter + numOfSamples(i))) = descriptorFADE;
    UFADE(:,(sumCounter + 1):(sumCounter + numOfSamples(i))) = descriptorUFADE;
    sumCounter = sumCounter + numOfSamples(i);
end

clear allActionsIndex allData sumCounter counter fourierTransformed rawFourier inverseTransform indexes Vomega descriptorFADE descriptorUFADE;



%% 4. Start (kNN/K fold cross validation)
numOfNN = 1;
Kfold = 10;

%intervalNum = floor(sum(numOfSamples)/10);
indices = crossvalind('Kfold', sum(numOfSamples), Kfold);
rightPredFADE = 0;
totalPredFADE = 0;
rightPredUFADE = 0;
totalPredUFADE = 0;


%% 5. Classification and validation
confmatFADE = zeros(max(class));
confmatUFADE = zeros(max(class));

% Clear all NaN in descriptors, not needed on this set in paricular, but for robustness

meanVec = mean(FADE);
[row, col] = find(isnan(FADE));
FADE(row,col) = meanVec(col);

meanVec = mean(UFADE);
[row, col] = find(isnan(UFADE));
UFADE(row,col) = meanVec(col);

for i = 1:Kfold
    
    
    testIndexes = find(indices == i);
    trainIndexes = find(indices ~= i);
    testFADE = FADE(:,testIndexes);
    trainFADE = FADE(:, trainIndexes);
    testUFADE = UFADE(:, testIndexes);
    trainUFADE = UFADE(:, trainIndexes);
    
    %Covariances for mahalanobis distance
%     trainFADECov = cov(trainFADE');
%     trainFADECov = trainFADECov + eye(length(trainFADECov));

%     trainUFADECov = cov(trainUFADE');
%     trainUFADECov = trainUFADECov + eye(length(trainUFADECov));
    
    %other distances: 'cityblock' 'euclidean' 'chebychev' 'mahalanobis'
    %'minkowski' ('Exponent', ...)
    Model = fitcknn(trainFADE',labels(trainIndexes)', 'NumNeighbors',numOfNN,'Standardize',1, 'Distance', 'cityblock');
    newLabels = predict(Model, testFADE');
    rightPredFADE = rightPredFADE + sum(newLabels' == labels(testIndexes));
    totalPredFADE = totalPredFADE + length(testIndexes);
    confmatFADE = confmatFADE + confusionmat(newLabels, labels(testIndexes)', 'order', 1:max(class));
    
    Model2 = fitcknn(trainUFADE',labels(trainIndexes)', 'NumNeighbors',numOfNN,'Standardize',1, 'Distance', 'cityblock');
    newLabels2 = predict(Model2, testUFADE');
    rightPredUFADE = rightPredUFADE + sum(newLabels2' == labels(testIndexes));
    totalPredUFADE = totalPredUFADE + length(testIndexes);
    confmatUFADE = confmatUFADE + confusionmat(newLabels2, labels(testIndexes)', 'order', 1:max(class));
end

%% 6. Plot the confmats, compute recognition rates

figure('Name', 'FADE Using 1-NN');
b = bar3(confmatFADE);
colormap(hot); %parula
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
colorbar;

figure('Name', 'UFADE Using 1-NN');
b = bar3(confmatUFADE);
colormap(parula); %parula
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
colorbar;

recognitionRate1 = rightPredFADE/totalPredFADE;
recognitionRate2 = rightPredUFADE/totalPredUFADE;

fprintf('Recognition rate using FADE: %s\n', recognitionRate1);
fprintf('Recognition rate using UFADE: %s\n', recognitionRate2);

toc

clear recognitionRate1 recognitionRate2 confmatUFADE confmatFADE trainUFADE testUFADE testFADE trainFADE

%% 7. Try a Bayesian Classifier on FADE
Kfold = 10;

meanVec = zeros(size(FADE, 1), numOfClasses);
covMat = zeros(size(FADE,1), size(FADE,1), numOfClasses);
sumOfSamples = 0;
rightPredBayes = 0;
confmatBayes = zeros(numOfClasses);

indices = crossvalind('Kfold', sum(numOfSamples), Kfold);

for j = 1:Kfold
    
    testIndexes = find(indices == j);
    trainIndexes = find(indices ~= j); % TODO: shift these cases
    likelihood = zeros(length(testIndexes), max(class));
    
    for i = 1:numOfClasses
    
        currentIndexes = find(labels(trainIndexes) == i);
        trainFADE = FADE(:, trainIndexes(currentIndexes));
    
        meanVec(:,i) = mean(trainFADE, 2)'; %TOCHECK
        covMat(:,:,i) = cov(trainFADE')+ .0001 * eye(length(covMat(:,1,1)));
        likelihood(:,i) = mvnpdf(FADE(:,testIndexes)', meanVec(:,i)', covMat(:,:,i));
    end
    
    [~,predictedLabels] = max(likelihood, [], 2);
    rightPredBayes = rightPredBayes + sum(predictedLabels' == labels(testIndexes));
    sumOfSamples = sumOfSamples + length(testIndexes);
    confmatBayes = confmatBayes + confusionmat(predictedLabels, labels(testIndexes)', 'order', 1:max(class));
end

recognitionRateBayes = rightPredBayes/sumOfSamples;

figure('Name', 'FADE Using Bayes-Classifier');
b = bar3(confmatBayes);
colormap(jet); %jet, summer, autumn
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
colorbar;

fprintf('Recognition rate using Bayes on FADE: %s\n', recognitionRateBayes);
