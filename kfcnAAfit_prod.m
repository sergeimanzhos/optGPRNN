function a = kfcnAAfit(XN,XM,theta)
% HDMR-type kernel with squared exponential component functions
global Nsets coordsets polorder 
%a = exp(-(pdist2(XN(:,coordsets(1,:)),XM(:,coordsets(1,:))).^2)/(2*exp(theta(1))^2));
h = XN(:,coordsets(1,:))*XM(:,coordsets(1,:))';
a = 1;  for kk=1:polorder, a = a+(h.^kk); end; 
for i=2:Nsets,
    ai = 1;
    %a = a+exp(-(pdist2(XN(:,coordsets(i,:)),XM(:,coordsets(i,:))).^2)/(2*exp(theta(i))^2));
    h = XN(:,coordsets(i,:))*XM(:,coordsets(i,:))';
    for kk=1:polorder, ai = ai+(h.^kk); end;
    a = a+ai;
end;
a = a/Nsets;
