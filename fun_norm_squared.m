    function nrm=fun_norm_squared(B)
        C=B.*B;
        nrm=sum(sum(sum(C,1),2),3);
    end