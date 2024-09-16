module fun_smm12
include("./src/Replication_of_Sieg_Yoon_2017.jl")
using .Replication_of_Sieg_Yoon_2017
export smm12

using QuadGK

function smm12(x)
    # Global variables
    # preliminary
    global election_number weight
    global num_sim n_app a_grid p_d 
    global op_print_results op_thirdstage op_policyexp op_valuefunction
    # model parameters
    global y_d y_r ecost_d ecost_r lambda sigma_pd
    global mu_d mu_r mu2_d mu2_r   # factor loadings
    # density estimate
    global t3 # Grids of density estimate
    global fxr fxd                 # SNP Approximation of kxd kxr
    global pdf_d pdf_r cdf_d cdf_r # Discretization of kad kar
    global sigr sigd               # Normal Approximation of ker ked
    # model solution
    global u_d l_d u_r l_r
    # moments
    global std_1 std_2 std_last 
    global d_share r_share share_all op_model
    # Read Parameters
    if op_model == 1
        lambda   = x[3]
        ecost_d  = 0.0
        ecost_r  = 0.0
        sigma_pd = x[4]
        y_d      = (x[1] + lambda*a_grid) # beta*y_d
        y_r      = (x[2] + lambda*a_grid) # beta*y_r
    elseif op_model == 2 
        lambda   = 0
        ecost_d  = 0
        ecost_r  = 0
        sigma_pd = x[3]
        y_d      = (x[1] + lambda*a_grid) # beta*y_d
        y_r      = (x[2] + lambda*a_grid) # beta*y_r
    elseif op_model == 3 
        lambda   = x[3]
        ecost_d  = x[4]
        ecost_r  = x[5]
        sigma_pd = x[6]
        y_d      = (x[1] + lambda*a_grid) # beta*y_d
        y_r      = (x[2] + lambda*a_grid) # beta*y_r
    end
    # Set Random Number Seed
    rng(100)
    # Find election standards
    find_standard_ttl
    if op_valuefunction == 1
        display('draw value function')    
        draw_value_function
    end
    # Calculate Moments and Simulation of Policyes
    # variables
    a   = zeros(num_sim,1) # ability
    rho = zeros(num_sim,1) # true   ideology
    x   = zeros(num_sim,1) # chosen ideology position
    incumbent = zeros(num_sim+1,4)
    party = zeros(num_sim+1,4)
    p1 = zeros(num_sim,4) # observed policie1
    p2 = zeros(num_sim,4) # observed policie2
    p3 = zeros(num_sim,4) # observed policie1
    p4 = zeros(num_sim,4) # observed policie2
    p5 = zeros(num_sim,4) # observed policie1
    
    # initialize
    uniform_r = rand(num_sim,1)
    uniform_d = rand(num_sim,1)
    
    a_r = zeros(num_sim,1)
    a_d = zeros(num_sim,1)
    
    for i in 1:n_app
        a_d(uniform_d <= cdf_d(i+1) & uniform_d > cdf_d[i]) = i
        a_r(uniform_r <= cdf_r(i+1) & uniform_r > cdf_r[i]) = i
    end
    
    fun1 = y -> (fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
    fun3 = y -> (fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)
    
    rho_r = randpdf(fun1(t3),t3,[num_sim 1])
    rho_d = randpdf(fun3(t3),t3,[num_sim 1])
    
    p1_r = normrnd(0,sigr[1],[num_sim 4])
    p1_d = normrnd(0,sigd[1],[num_sim 4])
    p2_r = normrnd(0,sigr[2],[num_sim 4])
    p2_d = normrnd(0,sigd[2],[num_sim 4])
    p3_r = normrnd(0,sigr[3],[num_sim 4])
    p3_d = normrnd(0,sigd[3],[num_sim 4])
    p4_r = normrnd(0,sigr[4],[num_sim 4])
    p4_d = normrnd(0,sigd[4],[num_sim 4])
    p5_r = normrnd(0,sigr[5],[num_sim 4])
    p5_d = normrnd(0,sigd[5],[num_sim 4])
    
    ct1 = 1
    ct2 = 1
    ct3 = 1
    ct4 = 1

    incumbent[1, :] = 0
    party[1, :] = 1
    
    # Start generate model from period 1 to num_sim
    for i in 1:num_sim
        if incumbent[i, 1] == 0
           if party[i, :] == 1
               rho[i] = rho_d[ct1)
               a[i] = a_d[ct1)
               ct1 = ct1+1  
               if (l_d(a[i])>u_d(a[i])) || (rho[i]<l_d(a[i])-y_d(a[i])) || (rho[i]>u_d(a[i])+y_d(a[i]))
                   x[i] = rho[i]
                   incumbent[i+1, :] = 0
                   party[i+1, :] = 2
               elseif rho[i] < l_d(a[i])
                   x[i] = l_d(a[i])
                   incumbent[i+1, :] = 1
               elseif rho[i] > u_d(a[i])
                   x[i] = u_d(a[i])
                   incumbent[i+1, :] = 1           
               else
                   x[i] = rho[i]
                   incumbent[i+1, :] = 1
               end     
           elseif party[i, 1] == 2
               rho[i] = rho_r[ct2]
               a[i] = a_r[ct2]
               ct2 = ct2 + 1        
               if (l_r(a[i])>u_r(a[i])) || (rho[i]<l_r(a[i])-y_r(a[i])) || (rho[i]>u_r(a[i])+y_r(a[i])) # extremist
                   x[i] = rho[i]
                   incumbent[i+1, :] = 0
                   party[i+1, :] = 1
               elseif rho[i]<l_r(a[i])   # left moderate
                   x[i] = l_r(a[i])
                   incumbent[i+1, :] = 1               
               elseif rho[i] > u_r(a[i]) # right moderate
                   x[i] = u_r(a[i])
                   incumbent[i+1, :] = 1
               else                      # centrists
                   x[i] = rho[i]
                   incumbent[i+1, :] = 1                    
               end           
           end
        elseif incumbent[i, 1] == 1 
            rho[i] = rho[i-1]
            a[i] = a[i-1]
            party[i,:] = party[i-1,:]
            x[i] = rho[i]
            incumbent[i+1,:] = 0
            party[i+1,:] = randi[2]
        end  
        if party[i, 1] == 1
            p1[i,:] = x[i]         + p1_d[ct3,:]
            p2[i,:] = mu_d[2]*x[i] + p2_d[ct3,:]
            p3[i,:] = mu_d[3]*x[i] + a_grid(a[i])           + p3_d[ct3,:]
            p4[i,:] = mu_d[4]*x[i] + mu2_d[4]*a_grid(a[i])  + p4_d[ct3,:]
            p5[i,:] = mu_d[5]*x[i] + mu2_d[5]*a_grid(a[i])  + p5_d[ct3,:]
            ct3 = ct3+1
        elseif party[i, 1] == 2
            p1[i,:] = x[i]         + p1_r[ct4,:]
            p2[i,:] = mu_r[2]*x[i] + p2_r[ct4,:]
            p3[i,:] = mu_r[3]*x[i] + a_grid(a[i])           + p3_r[ct4,:]
            p4[i,:] = mu_r[4]*x[i] + mu2_r[4]*a_grid(a[i])  + p4_r[ct4,:]
            p5[i,:] = mu_r[5]*x[i] + mu2_r[5]*a_grid(a[i])  + p5_r[ct4,:]       
            ct4 = ct4+1
        end     
    end

    # Calculate Moments Using Simulated Policies
    mean_2_sim= zeros(5,2)
    mean_all_sim= zeros(5,2)
    std_1_sim= zeros(5,2)
    std_2_sim= zeros(5,2)
    
    p1v = reshape(p1,[4*num_sim,1])
    p2v = reshape(p2,[4*num_sim,1])
    p3v = reshape(p3,[4*num_sim,1])
    p4v = reshape(p4,[4*num_sim,1])
    p5v = reshape(p5,[4*num_sim,1])
    
    partyv = reshape(party[1:num_sim,:],[4*num_sim,1])
    election_number1 = 1*(incumbent[1:num_sim,:]==0 & incumbent[2:num_sim+1,:]==1 ) + 2*(incumbent[1:num_sim,:]==1 )+3*(incumbent[1:num_sim,:]==0 & incumbent[2:num_sim+1,:]==0)
    election_number = reshape(election_number1,[4*num_sim,1])
    
    datav = [p1v p2v p3v p4v p5v]

    for i in 1:5
        ps = datav[:,i]
        for j in 1:2
            std_1_sim[i,j] = std(ps(partyv[1:num_sim*4]==j   & election_number==1 ))        
            mean_2_sim[i,j] = mean(ps(partyv[1:num_sim*4]==j & election_number==2 ))
            std_2_sim[i,j] = std(ps(partyv[1:num_sim*4]==j   & election_number==2 ))       
            mean_all_sim[i,j] = mean(ps(partyv[1:num_sim*4]==j  ))
        end
    end
    
    share_all_sim = zeros(3,4)

    for i in 1:3    
        ps = datav[:,i+2]
        std4 = std(ps)
    
        id_last_1 = sum(election_number!=2 & ps<-1*std4)
        id_last_2 = sum(election_number!=2 & ps>=-1*std4 & ps< 0)
        id_last_3 = sum(election_number!=2 & ps>= 0 & ps< 1*std4)
        id_last_4 = sum(election_number!=2 & ps(:,1)>= 1*std4 )
    
        id_ext_1 = sum(election_number==3 & ps<-1*std4)
        id_ext_2 = sum(election_number==3 & ps>=-1*std4 & ps< 0)
        id_ext_3 = sum(election_number==3 & ps>= 0 & ps< 1*std4)
        id_ext_4 = sum(election_number==3 & ps>= 1*std4 )
    
        share_all_sim[i,1] = id_ext_1/id_last_1
        share_all_sim[i,2] = id_ext_2/id_last_2
        share_all_sim[i,3] = id_ext_3/id_last_3
        share_all_sim[i,4] = id_ext_4/id_last_4
    end    
        
    # Moments
    g = zeros(9,2)
    
    for j in 1:2
        id_last_all = sum(partyv==j & election_number!=2)
        id_ext_all = sum(partyv==j & election_number==3)
        g[j,2] = id_ext_all/id_last_all
    end
    
    g[1,1] =  d_share 
    g[2,1] =  r_share 
    
    g[3,1] = 0.5041322
    g[3,2] = p_d
    
    # Moments to identify the benefit of holding office
    ct = 3
    g[ct+1,1] = std_1[1,1]/std_2[1,1]
    g[ct+2,1] = std_1[1,2]/std_2[1,2]
    
    g[ct+1,2] = std_1_sim[1,1]/std_2_sim[1,1]
    g[ct+2,2] = std_1_sim[1,2]/std_2_sim[1,2]
    
    # Identification of lambda, mu25, mu26
    
    ct = 5
    for i = 1
        g[ct+1+4*(i-1),1] = share_all[i,1]
        g[ct+2+4*(i-1),1] = share_all[i,2]
        g[ct+3+4*(i-1),1] = share_all[i,3]
        g[ct+4+4*(i-1),1] = share_all[i,4]
        g[ct+1+4*(i-1),2] = share_all_sim[i,1]
        g[ct+2+4*(i-1),2] = share_all_sim[i,2]
        g[ct+3+4*(i-1),2] = share_all_sim[i,3]
        g[ct+4+4*(i-1),2] = share_all_sim[i,4]
    end
    
    
    g_diff = (g[:,1] - g[:,2])
    f = g_diff'*weight*g_diff

    #Print results
    if op_print_results == 1  
    format short
    
    display('model fits')
    display(g[1:2,:])
    display(g[6:9,:])
    display(g[4:5,:])
    display(g[3,:])
    
    pr1_r = transpose(collect(LinRange(0,1,n_app)))
    pr1_d = transpose(collect(LinRange(0,1,n_app)))
    
    pr2_r = transpose(collect(LinRange(0,1,n_app)))
    pr2_d = transpose(collect(LinRange(0,1,n_app)))
    
    
    for i in 1:n_app
        pr1_r[i] = quadgk(fun1,l_r[i],u_r[i])[1]
        pr1_d[i] = quadgk(fun3,l_d[i],u_d[i])[1]
        pr2_r[i] = quadgk(fun1,l_r[i]-y_r[i],u_r[i]+y_r[i])-pr1_r[i][1]
        pr2_d[i] = quadgk(fun3,l_d[i]-y_d[i],u_d[i]+y_d[i])-pr1_d[i][1]
    end
        
    display('centrist D and R')
    display(sum(pr1_d.*pdf_d) )
    display(sum(pr1_r.*pdf_r) )
    display('Moderates D and R')
    display(sum(pr2_d.*pdf_d) )
    display(sum(pr2_r.*pdf_r) )
    display('Extremist D and R')
    display(sum((1-pr1_d-pr2_d).*pdf_d) )
    display(sum((1-pr1_r-pr2_r).*pdf_r) )
        
    figure    
    plot(a_grid,u_d,'Color','b','LineStyle','--','LineWidth',2)
    hold on
    plot(a_grid,u_r,'Color','r','LineStyle','--','LineWidth',2)
    hold on
    plot(a_grid,l_d,'Color','b','LineStyle','-','LineWidth',2)
    hold on
    plot(a_grid,l_r,'Color','r','LineStyle','-','LineWidth',2)
    hold on
    plot(a_grid,u_d+y_d,'Color','b','LineStyle','-.','LineWidth',2)
    hold on
    plot(a_grid,u_r+y_r,'Color','r','LineStyle','-.','LineWidth',2)
    hold on
    plot(a_grid,l_d-y_d,'Color','b','LineStyle',':','LineWidth',2)
    hold on
    plot(a_grid,l_r-y_r,'Color','r','LineStyle',':','LineWidth',2)
    hold on
    hleg = legend('$\bar{s}_D$','$\bar{s}_R$','$\underline{s}_D$','$\underline{s}_R$'...
        ,'$\bar{\rho}_D$', '$\bar{\rho}_R$'...
        ,'$\underline{\rho}_D$','$\underline{\rho}_R$')
    set(hleg, 'Box','off','Location','eastoutside','Interpreter','LaTex')
    %title('Election Stadard')
    xlabel('competence')
    ylabel('ideology')
    axis([-1.0 1.0 -2 2])
    
    p1_v = p1[:]*std_last[1]
    p2_v = p2[:]*std_last[2]
    p3_v = p3[:]*std_last[3]*100
    p4_v = p4[:]*std_last[4]*100
    p5_v = p5[:]*std_last[5]
    a_v = a_grid(a)
    x_v = x
    display('with term limit')
    mean_v = zeros(7,1)
    mean_v[1] = mean(p1_v)
    mean_v[2] = mean(p2_v)
    mean_v[3] = mean(p3_v)
    mean_v[4] = mean(p4_v)
    mean_v[5] = mean(p5_v)
    mean_v[6] = mean(x_v)
    mean_v[7] = mean(a_v)
    display('mean ability')
    display(mean_v[7])
    display('mean policy')
    display(mean_v[1:5])
    display('std')
    std_v = zeros(5,1)
    std_v[1] = std(p1_v)
    std_v[2] = std(p2_v)
    std_v[3] = std(p3_v)
    std_v[4] = std(p4_v)
    std_v[5] = std(p5_v)
    display(std_v)
    
    end
    # Option to run third stage estimation
    if op_policyexp == 1     
        display('solve model without term limit')
        ntl_snp2
    end

    if op_thirdstage == 1
        display('estimate third stage')    
        third_stage_snp2
    end
    
    end
    
    return f    

end