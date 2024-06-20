function y=fun_randomperm(x)
%
%
n=length(x);
seq=randperm(n);
y=x(seq);
end

