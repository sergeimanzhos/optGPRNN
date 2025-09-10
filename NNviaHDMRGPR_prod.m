% The code for the GPR-NN method was provided in J. Phys. Chem. A, 127, 7823–7835 (2023)
% This version is used for weights optimization with MC
% The optimization cycle is external
% This code was not produced as an end-user aimed distribution. No effort
% was made to optimize CPU or RAM performance. 
% Contact Sergei.Manzhos@gmail.com with questions about the code or the method
function a = NNviaHDMRGPR_prod(perturb, material, rrmax, ifprod, lp, np)
global f  % globalize the figure to be able to suppress multiple figures from the main cycle 

% This is the main function implementing the method
% output is a vector containing [trainRmse testRmse]
format compact
global polorder kernel_noise          % this version allows exploring noisy neurons, not used in this project
global D d Nparams Nsets coordsets widths
polorder = 3                                 % used when using a dot product based kernel 
neuron_noise = 0.0                           % noise addition to the neuron % noise addition to the kernel. NB: noise senstitity is high, start with small values
ifnormalize = 0                              % if 0 then rescaling to [0,1] if 1 then normalizing

% NB: length_parameter enters as as exp( .../(2*exp(l)^2) ) in the kernel function

% select the data to fit. material is passed from the expernal cycle
switch material
    case "H2O"
        length_parameter = log(0.45)       % water: about 0.45 is good. NB: this matters for the RBF kernel not for the product kernel mimicking transistor transfer functions
        SigmaF0 = 1e-13                    % this is to be understood as sigma^2 in GPR equations. Such low values are best for noise-free runs. With the noise it needs to be increased
        x = dlmread('h2o.dat');            % First D columns are features and the last column is the target in cm-1
        V = x(:,end);
        x = x(:,1:end-1);
    case "ZPE"
        data = readtable('ECM_QM9.csv');
        x = table2array(data(2:end,3:18)); % this selects 16 ECM features
        V = table2array(data(2:end,20));   % ZPE is in column 20, in Hartree
end;
% override hyperparameters by passed values if they are non-zero
if lp>0,
    length_parameter = log(lp)  % length parameter
end;
if np>0,
    SigmaF0 = np                % noise parameter
end;    

[Npts D] = size(x)
for i=1:D,
    x(:,i) = rescale(x(:,i));
end;
if ifnormalize, x=normalize(x); end;

rmse_train = [];
rmse_test = [];
DD = [];      
 
%adding redundant coordinates 
if rrmax>=0,                           % no. of reduntant coordinates 
    sobolpts = Sobol(D,rrmax+1);       % Sobol sequence. rrmax is the number of redundant coordinates (beyond D), it is passed from the outer cycle
    sobolpts = sobolpts (2:end,:);     % remove the first row of zeros
    sobolpts = [eye(D); sobolpts]+perturb;
else                                   % dimensionality reduction (negative rrmax)
    sobolpts = Sobol(D,D+rrmax+1);
    sobolpts = sobolpts(2:end,:)+perturb;  
end;
new = [];
for rr = 1:(D+rrmax),                  % number of rounds of adding redundant coordinates 
    new = [new x*sobolpts(rr,:)'];     % normalization below will take care of coefficients
end;
x = new; 
message = 'expanded dimensionality'
[Npts D] = size(x)                     % D from now on is the dimension of the redundant coordinates
for i=1:D,
    x(:,i) = rescale(x(:,i));
end;
if ifnormalize, x=normalize(x); end;

switch material
    case "H2O"
        Nfit = 1000                    % number of training points
        Ntest = 3000
        Ntest = min(Ntest,Npts-Nfit)   % or use all remaining points is less than Ntest remain
        rng(2,"twister")
    case "ZPE"
        Nfit = 3000
        Ntest = 10000
        rng(2,"twister")
end

d = 1                                  % order of HDMR - it is always 1 in this method

Nsets = nchoosek(D,d)
coordsets = nchoosek([1:D],d);

% inital hyperparameters
widths = length_parameter*ones(1,Nsets);
kparams0 = [widths ]; 
Nparams = max(size(kparams0));
order = randperm(max(size(V)));

widths = kparams0;
% model construction
if ifprod,
    K = kfcnAAfit_prod(x(order(1:Nfit),:),x(order(1:Nfit),:),kparams0);
else
    K = kfcnAAfit(x(order(1:Nfit),:),x(order(1:Nfit),:),kparams0);
end;
c = (K+eye(Nfit)*SigmaF0)\V(order(1:Nfit));

ybreakdown = 0;
for i=1:Nsets,
    if ifprod,
        Kstari = kfcnAAbreakdown_prod(x(order(1:Nfit),:),x(order(1:Nfit),:),kparams0,i);
    else
        Kstari = kfcnAAbreakdown(x(order(1:Nfit),:),x(order(1:Nfit),:),kparams0,i);
    end;
    fcomponent{i} = Kstari*c.*(1+neuron_noise*(rand(Nfit,1)-0.5));
    ybreakdown = ybreakdown+fcomponent{i};
    varfcomponent(i) = var(fcomponent{i});
    meani(i) = mean(fcomponent{i});
end;

% prediction
if ifprod,
    Kall = kfcnAAfit_prod(x(order(1:Nfit+Ntest),:),x(order(1:Nfit),:),kparams0);
else
    Kall = kfcnAAfit(x(order(1:Nfit+Ntest),:),x(order(1:Nfit),:),kparams0);
end;
y = Kall*c;
message = 'predicted train and test sets'
    %try to predict as some of component functions
    y = 0;
    for i=1:Nsets,
        if ifprod,
            Kstari = kfcnAAbreakdown_prod(x(order(1:Nfit+Ntest),:),x(order(1:Nfit),:),kparams0,i);
        else
            Kstari = kfcnAAbreakdown(x(order(1:Nfit+Ntest),:),x(order(1:Nfit),:),kparams0,i);
        end;
        y = y + Kstari*c.*(1+neuron_noise*(rand(Nfit+Ntest,1)-0.5));
    end;
    message = 'predicted train and test sets as a sum of component functions' 

try close(f), end
f=figure;
f.Position=[100 100 800 900];  
subplot(3,2,1)
plot(V(order(Nfit+1:Nfit+Ntest)),y(Nfit+1:Nfit+Ntest),'.r',V(order(1:Nfit)),y(1:Nfit),'.b')   

% NB y is already ordered same as x coming out of predict call so no need to use order for y here
Rtest = corrcoef(V(order(Nfit+1:Nfit+Ntest)),y(Nfit+1:Nfit+Ntest));
Rtest = Rtest(1,2)
Rtrain = corrcoef(V(order(1:Nfit)),y(1:Nfit));
Rtrain = Rtrain(1,2)
title(['  R_{train}=' num2str(Rtrain,'%.3f') ', R_{test}=' num2str(Rtest,'%.3f')])
xlabel('exact') 
ylabel(['model, N=' num2str(Nsets)])

rmse_train = [rmse_train; sqrt( mean((V(order(1:Nfit))-y(1:Nfit)).^2) )];
rmse_test = [rmse_test; sqrt( mean((V(order(Nfit+1:Nfit+Ntest))-y(Nfit+1:Nfit+Ntest)).^2) )];
DD = [DD; D];
D_trainRMSE_testRMSE = [DD rmse_train rmse_test]  

subplot(3,2,2)
bar(sqrt(varfcomponent))
title('importance of f_n(y_n)')
xlabel('function') 
ylabel('std')

[dummy orderoffi] = sort(varfcomponent, 'descend');
% plot largest component functions
if d==1,
for l=1:min(4,Nsets),
    subplot(3,2,2+l)
    xf = x(order(1:Nfit),orderoffi(l));
    yf = fcomponent{orderoffi(l)}-mean(fcomponent{orderoffi(l)});
    [dummy orderxf] = sort(xf, 'ascend');
    plot(xf(orderxf), yf(orderxf))
    xlabel(['y_n, n=' num2str(orderoffi(l))])
    ylabel('f_n(y_n)')
end;
end;
%saveas(f,'fig.png')

clear varfcomponent meani fcomponent
a = [rmse_train rmse_test]; % the return values of the main function 

