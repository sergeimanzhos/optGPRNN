function a = kfcnAAbreakdown(XN,XM,theta,i)
% this function is used to return component function components of the
% covariance matrix used to analyse individual component functions 
% (K* in (K*)c in the article)
% make sure the formula is the same as in kfcnAAfit which is the full
% additive kernel (summing over all coordinate sets)
global Nsets coordsets 
a = exp(-(pdist2(XN(:,coordsets(i,:)),XM(:,coordsets(i,:))).^2)/(2*exp(theta(i))^2))/Nsets;
