clear all
% load('pr6.mat');
load prJP16x16 lBv edgerow edgecol p pH mH1 lvH1 uO n typv rhov nuv
Ms=1e2;
lp=zeros(n,1);
Bv = exp(lBv);
B = sparse(edgerow,edgecol,Bv,p,p);
B = B+B.';
QB = spdiags(sum(B,2),0,-B);
QB = QB(1:p-1,1:p-1);
lQB = logdet(QB);

qrn = sobolset(pH,'Skip',1e3,'Leap',1e2);
qrn = scramble(qrn,'MatousekAffineOwen');
qrn = qrandstream(qrn);

for i = 301:n
    i
%     xHi = randn(pH,M);
%     xHi = sparse(vH(:,i),0,pH,pH)*xHi+repmat(mH(:,i),M,1);
    c = 4;
    flag = 0;
    urnd = qrand(qrn,Ms).'; %[qrand(qrn,Ms/2).',normcdf(randn(pH,Ms/2-1))];
    while 1
        lb = normcdf(mH1(:,i)-c*exp(lvH1(:,i)));
        ub = normcdf(mH1(:,i)+c*exp(lvH1(:,i)));
        
        if all(lb <0.1) && all(ub>0.9)
            flag = 1
            c=c/1.2
            break;
        end
        
        
    
        uHi= repmat(lb,1,Ms)+urnd.*repmat(ub-lb,1,Ms); %
%         uHi = [uHi,normcdf(mH1(:,i))];
        lp_uOi = zeros(Ms,1);
    
        
    
        parfor j=1:Ms
            u = [uHi(:,j);uO(:,i)];
            Cv = allcopulapdf(typv,[u(edgerow),u(edgecol)],rhov,nuv);
            BC = sparse(edgerow,edgecol,Bv.*Cv,p,p);
            BC = BC+BC.';
            QBC = spdiags(sum(BC,2),0,-BC);
            QBC = QBC(1:p-1,1:p-1); 
            lp_uOi(j) = logdet(QBC); %;
        end
        vmin = sum(exp(lp_uOi-lQB)<1e-3);
        if  vmin < 5
            c = c*1.2;
        elseif vmin > 20
            c = c/1.1;
        else
            c
            break;
        end
    end 
    
    nc = ceil(log(c/0.1)/log(1.1))
    lb0 = normcdf(mH1(:,i));
    ub0 = normcdf(mH1(:,i));
    a = 0;
    lpi = -Inf;
    
    
    for k = 0:nc
        lbh = normcdf(mH1(:,i)-0.1*1.2^k*exp(lvH1(:,i)));
        ubh = normcdf(mH1(:,i)+0.1*1.2^k*exp(lvH1(:,i)));
        M =5e3;
        urnd = qrand(qrn,M).'; %[qrand(qrn,M*0.8).',rand(pH,M/5)];
        uHi= repmat(lbh,1,M)+urnd.*repmat(ubh-lbh,1,M); 
        id = any(uHi<=repmat(lb0,1,M)|uHi>repmat(ub0,1,M));
        M = sum(id);
        if M ==0 
            continue;
        end
        uHi = uHi(:,id);
        lp_uOi = zeros(M,1);
        parfor j=1:M
            u = [uHi(:,j);uO(:,i)];
            Cv = allcopulapdf(typv,[u(edgerow),u(edgecol)],rhov,nuv);
            BC = sparse(edgerow,edgecol,Bv.*Cv,p,p);
            BC = BC+BC.';
            QBC = spdiags(sum(BC,2),0,-BC);
            QBC = QBC(1:p-1,1:p-1);         
            lp_uOi(j) = logdet(QBC); %;
        end
        v = lp_uOi-lQB+log(prod(ubh-lbh)-prod(ub0-lb0))-log(M);
        vmax = max(v);
        a = max(vmax - 500,a);
        lpi = log(exp(lpi-a)+sum(exp(v-a)))+a;
        ub0 = ubh;
        lb0 = lbh;
        if k == nc && flag == 1
            M =5e3;
            uHi = qrand(qrn,M).'; %[qrand(qrn,M*0.8).',rand(pH,M/5)];
            id = any(uHi<=repmat(lb0,1,M)|uHi>repmat(ub0,1,M));
            M = sum(id);
            if M ==0 
                continue;
            end
            uHi = uHi(:,id);
            lp_uOi = zeros(M,1);
            parfor j=1:M
                u = [uHi(:,j);uO(:,i)];
                Cv = allcopulapdf(typv,[u(edgerow),u(edgecol)],rhov,nuv);
                BC = sparse(edgerow,edgecol,Bv.*Cv,p,p);
                BC = BC+BC.';
                QBC = spdiags(sum(BC,2),0,-BC);
                QBC = QBC(1:p-1,1:p-1);         
                lp_uOi(j) = logdet(QBC); %;
            end
            v = lp_uOi-lQB+log(prod(ubh-lbh)-prod(ub0-lb0))-log(M);
            vmax = max(v);
            a = max(vmax - 500,a);
            lpi = log(exp(lpi-a)+sum(exp(v-a)))+a;
        end
    end
    lpi
    lp(i) = lpi;
    
    
    
    
end
KLV= mean(lp)