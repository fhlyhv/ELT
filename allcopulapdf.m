function PDF = allcopulapdf(typ,U,rho,nu)

% typ: 
%      elliptical
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

switch typ
    case 1 
        PDF = copulapdf('Gaussian',U,rho);
    case 2
        PDF = copulapdf('t',U,rho,nu);
    case 3
        PDF = Gumbelcopulapdf(U,rho);
    case 4
        PDF = copulapdf('Clayton',U,rho);
    case 5
        PDF = copulapdf('Frank',U,rho);
    case 6
        PDF = Galamboscopulapdf(U,rho);
    case 7 
        PDF = HusterReisscopulapdf(U,rho);
    case 8 
        PDF = tevcopulapdf(U,rho,nu);
end