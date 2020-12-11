function y = Gumbelcopulapdf(u,alpha)


v = -log(u); % u is strictly in (0,1) => v strictly in (0,Inf)
v = sort(v,2); vmin = v(:,1); vmax = v(:,2); % min/max, but avoid dropping NaNs
nlogC = vmax.*(1+(vmin./vmax).^alpha).^(1./alpha);
y = (alpha - 1 + nlogC) ...
    .* exp(-nlogC + sum(repmat((alpha-1),1,2).*log(v) + v, 2) + (1-2*alpha).*log(nlogC));
y(alpha == 1) = 1;