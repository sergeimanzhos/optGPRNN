% this is a wrapper code to optimize weights in redundant coordinates of  
% GPRNN with a MC algorithm 
% GPRNN engine is in NNviaHDMRGPR_prod.m 
clear all
global f

ifprod = 0; % if to use a dot product kernel, not used in this project

% select the data to fit
material = 'H2O'
factor = 1.0;
switch material
    case "H2O"
        D = 3
        rrmax = 20    % number of added redundant coordinates. rrmax+D is the total number of terms / neurons
        lp = 0.45
        np = 1e-13
    case "ZPE"
        D = 16
        lp = 0.5
        np = 1e-5
        factor = 1000   % factor applied to saved rmse for ease of viewing
end

% grand cycle
scan_results = [];
for rrmax = [0 10],   % scan the numner of redundant coordinates
    
perturb = zeros(D+rrmax, D);
perturb_best = perturb;
% first call the fit for Sobol weights
a = NNviaHDMRGPR_prod(perturb, material, rrmax, ifprod, lp, np);
    rmse_train_initial = a(1)
    rmse_test_initial = a(2)
rmse_train_best = a(1)
rmse_test_best = a(2)
rmse_train_best_history = rmse_train_best;
rmse_test_best_history = rmse_test_best;

MaxCycles = 1000 
for i=1:MaxCycles,
   rng("shuffle") 
   perturb = rand(D+rrmax, D)-0.5;  
   for j=1:(D+rrmax),
       perturb(j,:) = perturb(j,:)/norm(perturb(j,:));
   end;   
   perturb = perturb_best+rand*perturb/20;
   try close(f), end  % to prevent proliferation of figures during MC steps 
   a = NNviaHDMRGPR_prod(perturb, material, rrmax, ifprod, lp, np);
   if (a(1) < rmse_train_best),
       rmse_train_best = a(1)
       rmse_test_best = a(2)
       perturb_best = perturb;
   end;
   rmse_train_best_history = [rmse_train_best_history rmse_train_best];
   rmse_test_best_history = [rmse_test_best_history rmse_test_best];
end;

% rerun the best one for plotting 
a = NNviaHDMRGPR_prod(perturb_best, material, rrmax, ifprod, lp, np); 
plot([1:MaxCycles+1], rmse_train_best_history,'b',[1:MaxCycles+1], rmse_test_best_history,'r')
title('trainig (blue) and test (red) set errors')
xlabel('cycle') 
ylabel('rmse')
rmse_train_best
rmse_test_best
%perturb_best

scan_results = [scan_results; [rrmax [rmse_train_initial rmse_test_initial rmse_train_best rmse_test_best]*factor]]
dlmwrite(['scan_results_' material '_' num2str(lp) num2str(log10(np)) '.dat'], scan_results, 'delimiter','\t', 'precision', '%15.8f');
end; % for rrmax

