module fun_myfun_ttl
include("./src/Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export myfun_ttl

using QuadGK

function myfun_ttl(x)
    # Calculate the system of equations in 2-term limit case
    # Global variables
    #=global beta p_d lambda ecost_d ecost_r p_all_r p_all_d sigma_pd
    global n_app pdf_d pdf_r a_grid 
    global fxr fxd
    global y_r y_d=#

    n = n_app
    
    v_d = x[1]
    v_r = x[2]
    
    u_d = x(3:3+n-1)
    l_d = x(3+n:3+2*n-1)
    u_r = x(3+2*n:3+3*n-1)
    l_r = x(3+3*n:3+4*n-1)
    
    v_d_grid = zeros(1,n)
    v_r_grid = zeros(1,n)
    
    f1_grid = zeros(1,n)
    f2_grid = zeros(1,n)
    f3_grid = zeros(1,n)
    f4_grid = zeros(1,n)
    
    p1r = zeros(n,1)
    p2r = zeros(n,1)
    p3r = zeros(n,1)
    p5r = zeros(n,1)
    p6r = zeros(n,1)
    
    g1r = zeros(n,1)
    g2r = zeros(n,1)
    g3r = zeros(n,1)
    g5r = zeros(n,1)
    g6r = zeros(n,1)
    
    p1d = zeros(n,1)
    p2d = zeros(n,1)
    p3d = zeros(n,1)
    p5d = zeros(n,1)
    p6d = zeros(n,1)
    
    g1d = zeros(n,1)
    g2d = zeros(n,1)
    g3d = zeros(n,1)
    g5d = zeros(n,1)
    g6d = zeros(n,1)
    
    e2r = zeros(n,1)
    e5r = zeros(n,1)
    
    e2d = zeros(n,1)
    e5d = zeros(n,1)
    
    p_d = exp(v_d/sigma_pd)/(exp(v_d/sigma_pd)+exp(v_r/sigma_pd))
    
    v_0 = v_d*p_d + (1-p_d)*v_r
    
    fun1 = y -> (fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2 .*exp(-(y-fxr[6]).^2/fxr[7]^2) 
    fun2 = y -> abs(y).*(fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2 .*exp(-(y-fxr[6]).^2/fxr[7]^2) 
    fun3 = y -> (fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2 .*exp(-(y-fxd[6]).^2/fxd[7]^2) 
    fun4 = y -> abs(y).*(fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2 .*exp(-(y-fxd[6]).^2/fxd[7]^2) 
    
    
    p_all_r = quadgk(fun2,-Inf,Inf)[1]
    p_all_d = quadgk(fun4,-Inf,Inf)[1]
    
    atol = 1d-7
    
    for i in 1:n
        if u_r[i] >= l_r[i]
            p1r[i] = quadgk(fun1,-Inf,l_r[i]-y_r[i])[1]
            p2r[i] = quadgk(fun1,l_r[i]-y_r[i],l_r[i])[1]
            p5r[i] = quadgk(fun1,u_r[i],u_r[i]+y_r[i])[1]
            p6r[i] = quadgk(fun1,u_r[i]+y_r[i],Inf)[1]
            p3r[i] = max(0,1 - p1r[i] - p2r[i] - p5r[i] - p6r[i])
            
            g1r[i] = quadgk(fun2,-Inf,l_r[i]-y_r[i])[1]
            g2r[i] = quadgk(fun2,l_r[i]-y_r[i],l_r[i])[1]
            g5r[i] = quadgk(fun2,u_r[i],u_r[i]+y_r[i])[1]
            g6r[i] = quadgk(fun2,u_r[i]+y_r[i],Inf)[1]
            g3r[i] = max(0,p_all_r - g1r[i] - g2r[i] - g5r[i] - g6r[i])
            
            e2r[i] = g2r[i]/max(p2r[i],atol)
            e5r[i] = g5r[i]/max(p5r[i],atol)
        
        end
    
        if u_d[i] >= l_d[i]
            p1d[i] = quadgk(fun3,-Inf,l_d[i]-y_d[i])[1]
            p2d[i] = quadgk(fun3,l_d[i]-y_d[i],l_d[i])[1]
            p5d[i] = quadgk(fun3,u_d[i],u_d[i]+y_d[i])[1]
            p6d[i] = quadgk(fun3,u_d[i]+y_d[i],Inf)[1]
            p3d[i] = max(0,1 - p1d[i] - p2d[i] - p5d[i] - p6d[i])
            
            g1d[i] = quadgk(fun4,-Inf,l_d[i]-y_d[i])[1]
            g2d[i] = quadgk(fun4,l_d[i]-y_d[i],l_d[i])[1]
            g5d[i] = quadgk(fun4,u_d[i],u_d[i]+y_d[i])[1]
            g6d[i] = quadgk(fun4,u_d[i]+y_d[i],Inf)[1]
            g3d[i] = max(0,p_all_d - g1d[i] - g2d[i] - g5d[i] - g6d[i])
            
            e2d[i] = g2d[i]/max(p2d[i],atol)
            e5d[i] = g5d[i]/max(p5d[i],atol)
        end
    
        if (lambda*a_grid[i]-ecost_r+ beta*v_0 - v_d >=0 && u_r[i]>=l_r[i])
            f1_grid[i] = -e2r[i] + lambda*a_grid[i]-ecost_r+ beta*v_0 - v_d
            f2_grid[i] = -e5r[i] + lambda*a_grid[i]-ecost_r+ beta*v_0 - v_d
        else  
            f1_grid[i] =  (l_r[i]-0.1)^2
            f2_grid[i] =  (u_r[i]+0.1)^2
        end
    
        if (lambda*a_grid[i]-ecost_d+ beta*v_0 - v_r>=0 && u_d[i]>=l_d[i])
            f3_grid[i] = -e2d[i] + lambda*a_grid[i]-ecost_d+ beta*v_0 - v_r
            f4_grid[i] = -e5d[i] + lambda*a_grid[i]-ecost_d+ beta*v_0 - v_r
        else  
            f3_grid[i] =  (l_d[i]-0.1)^2
            f4_grid[i] =  (u_d[i]+0.1)^2
        end
    end
    
    for i in 1:n
        if u_r[i] >= l_r[i]
            v_r_grid[i] =(p6r[i])*(lambda*a_grid[i]+beta*v_d)-g6r[i]+(p5r[i])*(-abs(u_r[i])+(1+beta)*lambda*a_grid[i]-beta*ecost_r+beta^2*v_0)-beta*g5r[i]+(p3r[i])*((1+beta)*lambda*a_grid[i]-beta*ecost_r+beta^2*v_0)-(1+beta)*g3r[i]+(p2r[i])*(-abs(l_r[i])+(1+beta)*lambda*a_grid[i]-beta*ecost_r+beta^2*v_0)-beta*g2r[i]+(p1r[i])*(lambda*a_grid[i]+beta*v_d)-g1r[i]
        else
            v_r_grid[i] = -p_all_r+lambda*a_grid[i]+beta*v_d
        end
        
        if u_d[i] >= l_d[i]
            v_d_grid[i] =(p6d[i])*(lambda*a_grid[i]+beta*v_r)-g6d[i]+(p5d[i])*(-abs(u_d[i])+(1+beta)*lambda*a_grid[i]-beta*ecost_d+beta^2*v_0) -beta*g5d[i]+(p3d[i])*((1+beta)*lambda*a_grid[i]-beta*ecost_d+beta^2*v_0)-(1+beta)*g3d[i].+(p2d[i])*(-abs(l_d[i])+(1+beta)*lambda*a_grid[i]-beta*ecost_d+beta^2*v_0)-beta*g2[i]+(p1d[i])*(lambda*a_grid[i]+beta*v_r)-g1d[i]
        else
            v_d_grid[i] = -p_all_d+lambda*a_grid[i]+beta*v_r
        end      
    end
    
    f[1] = sum(v_d_grid.*pdf_d) - v_d
    f[2] = sum(v_r_grid.*pdf_r) - v_r
    f[3:3+n-1]       = f1_grid
    f[3+n:3+2*n-1]   = f2_grid
    f[3+2*n:3+3*n-1] = f3_grid
    f[3+3*n:3+4*n-1] = f4_grid
    
    return f
end