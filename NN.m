% this code is for a comparison with a conventional NN (MLP)
clear all;

material = "H2O"
switch material
    case "H2O"
        data = dlmread('h2o.dat');
        x = data(:,1:3)';  % Radau coordinates
        t = data(:,4)';    % potential energy in cm-1
        factor = 1;
    case "ZPE"
        data = readtable('ECM_QM9.csv');
        x = table2array(data(2:end,3:18))';  % this selects 16 ECM features
        t = table2array(data(2:end,20))';    % ZPE is column 20, in Hartree
        factor = 1000;
end;
    
[D Npts] = size(x)
for i=1:D,
    x(i,:) = rescale(x(i,:));
end;

NtestSetDraws = 5;        % run NtestSetDraws times to account for local minima
no_of_samples = size(t,2);

MaxNeurons = 200
MaxLayers = 5
results = [];
Results = table(100,100, 'VariableNames',{'trainRMSE', 'testRMSE'});
for j = 1:MaxLayers,    % layers scan
for i = [5 10 15 20 25 30 35 50 75 100 125 150 175 200],   % neurons per layer scan
if i*j<301,
    for k = 1:NtestSetDraws,
        hiddenLayerSize = [];
        for jj = 1:j,
            hiddenLayerSize = [hiddenLayerSize i]
        end;
        trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
        net = fitnet(hiddenLayerSize, trainFcn);
        for l=1:j,
            net.layers{l}.transferFcn = 'tansig';
        end;
        net.layers{j+1}.transferFcn = 'purelin';
        net.trainParam.epochs = 2000;
        
        no_of_samples = size(t,2);
           rng(2,"twister") 
           rand_Ind = randperm(Npts);
        val_ratio = 0.0;
        no_of_valset = 0;
        switch material
            case "H2O"
                no_of_trainset = 1000;
                no_of_testset = 3000;
            case "ZPE"
                no_of_trainset = 3000;
                no_of_testset = 10000;
        end;
        
        train_Ind = rand_Ind(1:no_of_trainset);
        val_Ind = rand_Ind(no_of_trainset+1:no_of_trainset+no_of_valset);
        test_Ind = rand_Ind(no_of_trainset+1:no_of_trainset+no_of_testset);
        net.divideFcn = 'divideind';
        net.divideParam.trainInd = train_Ind;
        net.divideParam.valInd = val_Ind;
        net.divideParam.testInd = test_Ind;

        % Choose a Performance Function
        net.performFcn = 'mse';  % Mean Squared Error

        % Train the Network
        rng("shuffle")     % randomize initialization in case train-test split used a fixed seed 
        [net,tr] = train(net,x,t);

        % Test the Network        
        y = net(x);

        % Train_Test split
        y_train = y(train_Ind);
        t_train = t(train_Ind);
        y_val = y(val_Ind);
        t_val = t(val_Ind);
        y_test = y(test_Ind);
        t_test = t(test_Ind);

        % Calculate train performance
        train_RMSE = sqrt(mean((t_train-y_train).^2));  % rmse
        train_R = corrcoef(t_train,y_train);            % Rsqaured
        train_R = train_R(1,2);                         % R

        % Calculate test performance
        test_RMSE = sqrt(mean((t_test-y_test).^2)); % rmse
        test_R = corrcoef(t_test,y_test);           % Rsquared
        test_R = test_R(1,2);                       % R

        Results.trainRMSE(k) = train_RMSE
        Results.testRMSE(k) = test_RMSE
    end; % for k

    % Calculate Average Results
    Average_train_rmse = mean(Results.trainRMSE);
    min_train_rmse = min(Results.trainRMSE);
    max_train_rmse = max(Results.trainRMSE);
    Average_test_rmse = mean(Results.testRMSE);
    min_test_rmse = min(Results.testRMSE);
    max_test_rmse = max(Results.testRMSE);
    
    results = [results; [i j [Average_train_rmse min_train_rmse max_train_rmse Average_test_rmse min_test_rmse max_test_rmse]*factor]]
    dlmwrite('scan_results.dat', results, 'delimiter','\t', 'precision', '%15.8f');

end; % if i*j
end; % for j
end; % for i