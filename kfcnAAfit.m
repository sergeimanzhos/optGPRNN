function a = kfcnAAfit(XN,XM,theta)
% HDMR-type kernel with squared exponential component functions
global Nsets coordsets 
a = exp(-(pdist2(XN(:,coordsets(1,:)),XM(:,coordsets(1,:))).^2)/(2*exp(theta(1))^2));
for i=2:Nsets,
    a = a+exp(-(pdist2(XN(:,coordsets(i,:)),XM(:,coordsets(i,:))).^2)/(2*exp(theta(i))^2));
end;
a = a/Nsets;