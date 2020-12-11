function [B,rhov,nuv,mH,vH] = SVI(Adj,uO,rhov0,nuv0,typv,edgerow,edgecol,nedges,nnodes)

% stocahstic variational inference of ensemble of latent trees
% Code with ADAM step size automation
% Yu Hang, Sharon Huang, NTU, Feb. 2015.

%Input
% Adj -- adjacency matrix of the pyramidal graph describing the spatial dependence
% from which the spanning trees are drawn from

% uO -- CDF of the observed data in the bottom scale of the pyramidal model

% rhov0 -- initial values of the (first) parameters of the pairwise copulas

% nuv0 -- initial values of the (second) parameters of the pairwise copulas
% if applicable; specify nuv0 = [] if the pairwise copula are
% parameterized by one parameter

% typv -- type of the pairwise copulas (from 1 to 8)
%      1 -- Gaussian;
%      2 -- t;
%      Archimedemean
%      3 -- Gumbel;
%      4 -- Clayton;
%      5 -- Frank;
%      Extreme value
%      6 -- Galambos;
%      7 -- HusterReiss;
%      8 -- tev

% edgerow -- row indices of the non-zero entries in Adj
% edgecol -- column indices of the non-zero entreis in Adj
% nedges -- no. of edges in the pyramidal model
% nnodes -- no. of nodes in the pyramidal model

% Output
% B -- the beta matrix describing the weigthts of the edges in the
% pyramidal model

% rhov -- a vector the estimates of the first parameters of the pairwise
% copulas; the index of the edge corresponding to the pairwise copula is
% given [edgerow,edgecol]
% nuv -- a vector the estimates of the second parameters of the pairwise copulas

% mH -- the mean the of the Gaussian copula of the hidden variables in the
% coarser scales of the pyramidal model
% vH -- the variance of the Gaussian copula of the hidden variables 


% initialization
tic;
[pO,n] = size(uO);
nscale = length(nnodes);
pH = sum(nnodes(1:nscale-1));
p = pO+pH;

idH = edgerow <= pH | edgecol <= pH;
edgerowH = edgerow(idH);
edgecolH = edgecol(idH);

Adj = spdiags(sum(Adj,2),0,-Adj);
mH = -Adj(1:pH,1:pH)\Adj(1:pH,pH+1:p)*norminv(uO);
lvH = log(0.2*ones(pH,n)); % log of standard deviation




lBv = log(ones(nedges,1)); % vector consisting of log of all non-zero elements in the upper triangular of beta matrix
% a1 = 10*n;
% a2 = 1*sqrt(nedges);
rhov = rhov0;
nuv = nuv0;

% compute matrix E for the gradient of beta and c
E = sparse(nedges,p-1);
for i = 1:nedges
    if edgecol(i)<p
        E(i,edgerow(i)) = 1;
        E(i,edgecol(i)) = -1;
    else
        E(i,edgerow(i)) = 1;
    end
end

% compute matrix S for the gradient of m and v
S = sparse(pH,2*length(edgerowH));
eid = [edgerowH;edgecolH];
for i = 1:pH
    S(i,eid == i) = 1;
end


% initialize parameters for step size selection
g2B = 0;
g1B = 0;
g2r = 0;
g1r = 0;
g2m = zeros(pH,n);
g1m = zeros(pH,n);
g2v = zeros(pH,n);
g1v = zeros(pH,n);
tau1 = 0.9; %0.95;
tau2 = 0.999;
eps = 1e-6;
eta= 1e-2;
% initialization for low-rank approxiamtion method
c = graphcolor_irregular(abs(Adj(1:p-1,1:p-1))^4);

% quasi random index
ida= haltonset(1,'Skip',1e3,'Leap',1e2);
ida = scramble(ida,'RR2'); 
ida = qrandstream(ida);


mH0 = mH;
lvH0 = lvH;
Bv0 = exp(lBv);

EB = [];
rhov_array = [];
lBv_array = [];
rhoa = [];
lBa = [];

% stochastic variational inference
for nt = 1:1e8
    
    % pick one sample randomly
    nid = ceil(n*qrand(ida,1));
    
    % draw one sample from the variational Gaussian distribution
    z = randn(pH,1); 
    vH = exp(lvH(:,nid));
    yH = vH.*z+mH(:,nid); 
    
    uH = normcdf(yH);
    u = [uH;uO(:,nid)];
    
    Bv = exp(lBv);
    B = sparse(edgerow,edgecol,Bv,p,p);
    B = B+B.';
    QB = spdiags(sum(B,2),0,-B);
    QB = QB(1:p-1,1:p-1);
    
    Cv = allcopulapdf(typv,[u(edgerow),u(edgecol)],rhov,nuv);
    BC = sparse(edgerow,edgecol,Bv.*Cv,p,p);
    BC = BC+BC.';
    QBC = spdiags(sum(BC,2),0,-BC);
    QBC = QBC(1:p-1,1:p-1);

    
    % compuate low-rank matrix L
    L = sparse((1:p-1).',c,sign(randi([0,1],p-1,1)-0.5));
    
    R = QB\L;
    RC = QBC\L;
    EL = E*L;
    
    % compute gradient    
    pLpBC = sum((E*RC).*EL,2);
    pLpC = Bv.*pLpBC;
    
    
    ur1 = u(edgerowH)+1e-6;
    ur1(ur1>=1) = 1-1e-7;
    ur2 = u(edgerowH)-1e-6;
    ur2(ur2<=0) = 1e-7;
    uc1 = u(edgecolH)+1e-6;
    uc1(uc1>=1) = 1-1e-7;
    uc2 = u(edgecolH)-1e-6;
    uc2(uc2<=0) = 1e-7;
    
    pCpu = [(allcopulapdf(typv,[ur1,u(edgecolH)],rhov(idH),nuv(idH))-allcopulapdf(typv,[ur2,u(edgecolH)],rhov(idH),nuv(idH)))./(ur1-ur2);...
        (allcopulapdf(typv,[u(edgerowH),uc1],rhov(idH),nuv(idH))-allcopulapdf(typv,[u(edgerowH),uc2],rhov(idH),nuv(idH)))./(uc1-uc2)];
    pLpmv = S*([pLpC(idH);pLpC(idH)].*pCpu).*normpdf(yH);
    
    
    pCpr = (allcopulapdf(typv,[u(edgerow),u(edgecol)],rhov+1e-6,nuv)-allcopulapdf(typv,[u(edgerow),u(edgecol)],rhov-1e-6,nuv))/2e-6; % pay attention to the range of the parameter
    
    
    % smoothed gradient
    
    pLpB = ((Cv.*pLpBC - sum((E*R).*EL,2))*n-2*(Bv-1)).*Bv;   %-4*a1*(norm(Bv)^2-a2^2)*Bv
    pLpr = pLpC.*pCpr*n;
    pLpm = pLpmv - mH(:,nid);
    pLpv = (pLpmv.*z - vH + 1./vH).*vH;

    
        
    g1B = tau1*g1B+(1-tau1)*pLpB;
    g1r = tau1*g1r+(1-tau1)*pLpr;
    g1m(:,nid) = tau1*g1m(:,nid)+(1-tau1)*pLpm;
    g1v(:,nid) = tau1*g1v(:,nid)+(1-tau1)*pLpv;
    
    g2B = tau2*g2B+(1-tau2)*pLpB.^2;
    g2r = tau2*g2r+(1-tau2)*pLpr.^2;
    g2m(:,nid) = tau2*g2m(:,nid)+(1-tau2)*pLpm.^2;
    g2v(:,nid) = tau2*g2v(:,nid)+(1-tau2)*pLpv.^2;
    
    dB = g1B./(1-tau1)./(sqrt(g2B./(1-tau2))+eps);
    dr = g1r./(1-tau1)./(sqrt(g2r./(1-tau2))+eps);
    dm = g1m(:,nid)./(1-tau1)./(sqrt(g2m(:,nid)./(1-tau2))+eps);
    dv = g1v(:,nid)./(1-tau1)./(sqrt(g2v(:,nid)./(1-tau2))+eps);
    
    
    
    
    % update all parameters
    lBv = lBv+eta*dB; %dW;
    rhov = rhov+eta*dr;
    if any(rhov<1)
        rhov(rhov<1) = 1;
    end
    rhoa = cat(2,rhoa,rhov);
    lBa = cat(2,lBa,lBv);
    mH(:,nid) = mH(:,nid)+eta*dm;
    lvH(:,nid) = lvH(:,nid)+eta*dv;
    
    
    % check convergence
    
    
    
    if rem(nt,50*n) == 0
        mH1 = mH;
        lvH1 = lvH;
        Bv1 = exp(mean(lBa(:,randperm(50*n,50)),2));
        rhov1 = mean(rhoa(:,randperm(50*n,50)),2);
        MAE_B = mean(abs(Bv1-Bv0));
        MAE_rho = mean(abs(rhov1-rhov0));
        fprintf('# iterations = %d, MAE_B = %d, MAE_rho = %d\n',nt,MAE_B,MAE_rho);
        if MAE_B < 1e-3 && MAE_rho < 1e-3
            t = toc;
            Bv = Bv1; %exp(lBv);
%             Bv(abs(Bv)<1e-4) = 0;
%             id = find(Bv>0);
%             Bv(id) = Bv(id)/norm(Bv(id));
            B = sparse(edgerow,edgecol,Bv,p,p);
            B = B+B.';
            break;
        else
            Bv0 = Bv1;
            rhov0 = rhov1;
            eta=0.9*eta;
            rhov_array = cat(2,rhov_array,rhoa(:,1:n:end));
            lBv_array = cat(2,lBv_array,lBa(:,1:n:end));
            rhoa = [];
            lBa = [];
        end
    end
end