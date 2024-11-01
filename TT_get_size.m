function sz = TT_get_size(tt)
    %TT_get_size (tt,n) 
    %   Algorithm that returns the size of a tt
    %      
    %INPUT:
    %   tt (cell array with N+1 cells):  tensor train (tt) decomposition of a 
    %                                   N-th order tensor, where the tt-cores 
    %                                   are stored in the first N cells and the 
    %                                   location of the norm-core is stored in the 
    %                                   N+1 cell. If the tt is not in site_n
    %                                   mixed canonical form, the N+1 cell
    %                                   contains a 0.
    %OUTPUT:
    %   sz (N x 3 double):              size of TT decomposition, see example
    %                                   figure 

    N = length(tt) - 1;
    sz = zeros(N, 3);
    
    for i = 1:N
        core_size = size(tt{i},[1,2,3]);
        sz(i, :) = core_size;
    end
end