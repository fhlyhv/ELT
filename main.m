%clear
load MS8x8 Xtrain; %JPpeaks16x16_thr100 XDat;

XDat = Xtrain;
[n,~] = size(XDat);
pr = 8; % no. of rows at the bottom scale
pc = 8; % no. of columns at the bottom scale

%% marginal analysis: GEV fitting and smoothing
N=3000;   %no. of bootstrap subsets
Id=bootstrap(XDat,N);
[L0,G0,S0,L_Var,G_Var,S_Var] = GEVPrm_bootstrap (XDat,N,Id); 
Jp = thin_membrane(pc,pr);

[Lh,alpha_u]=EM_Smth(L0,L_Var,Jp);
[Gh,alpha_g]=EM_Smth(G0,G_Var,Jp);
[Sh,alpha_s]=EM_Smth(S0,S_Var,Jp);

% compute CDF
uO = gevcdf(XDat,repmat(Gh,n,1),repmat(Sh,n,1),repmat(Lh,n,1));


%% construct adjacency matrix of pyramidal model
pyr_size = [pr pc];
while(pr*pc > 1)
    pr = ceil(pr/2); pc = ceil(pc/2);
    pyr_size = [pr pc; pyr_size];
end
nnodes = prod(pyr_size,2);
nscale = size(nnodes,1);

% Initialize the tree

Adj = sparse(1);
for scale = 1:1:nscale-1
    ns = nnodes(scale+1);
    scr = pyr_size(scale+1,1);
    scc = pyr_size(scale+1,2);
    Dv = [1 1];
    Dv = kron(speye(floor(scr/2)), Dv);
    if(mod(scr,2) == 1)
        Dv = [Dv zeros(size(Dv,1),1); zeros(1, size(Dv, 2)) 1];
    end

    Dh = [1 1];
    Dh = kron(speye(floor(scc/2)), Dh);
    if(mod(scc,2) == 1)
        Dh = [Dh zeros(size(Dh,1),1); zeros(1, size(Dh, 2)) 1];
    end
    At = kron(Dh, Dv);
    interscale_A = [sparse(size(Adj,1)-size(At,1),size(At,2)) ;At];
%     if scale == nscale -1 
%         inscale_A = sparse(scr*scc,scr*scc);
%     else
        inscale_A = -triu(thin_membrane(scc,scr),1);
        inscale_A = inscale_A+inscale_A.';
%         inscale_A(interscale_A.'*interscale_A~=0) = 0;
%     end
    
    Adj = [Adj abs(sign(interscale_A)); abs(sign(interscale_A))' inscale_A];
    
end

[edgerow,edgecol] = find(triu(Adj,1));
nedges = length(edgerow);
rho0 = 10*ones(nedges,1);
nu0 = zeros(nedges,1);
typm = 3;

%% joint analysis: stochastic variatonal inference of ELT
[B,rhov,nuv,mh,vh] = SVI(Adj,uO.',rho0,nu0,typm,edgerow,edgecol,nedges,nnodes);

%% missing data imputation
load MS8x8 Xtest;

Grid = ones(8);
Grid(3:6,3:6) = NaN;
XMid = find(isnan(Grid(:)));
XO = Xtest(1,:);
XO(XMid) = NaN;

% Note that the hidden variables in the coarser scales don't have observations
XO = [NaN*ones(1,sum(nnodes(1:end-1))),XO];
XMid = [(1:sum(nnodes(1:end-1)))';sum(nnodes(1:end-1))+XMid]; 

[XO,lv] = SVI_ADAM(XMid,XO,B,rhom,num,typm*ones(nedges,1),Gh,Sh,Lh,sum(nnodes(1:end-1)));




