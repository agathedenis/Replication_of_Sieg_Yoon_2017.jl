module code_ntl_snp2

using QuadGK

global y_d y_r 
global t3 
global n_app cdf_d cdf_r a_grid
global l_d2 l_r2 u_d2 u_r2
global num_sim
global fxr fxd sigr sigd std_last
global pdf_d pdf_r

rng(100)

# Solve the model for the election standards

find_standard_ntl

# Simulation of Policies

# variables
a   = zeros(num_sim, 1) # ability
rho = zeros(num_sim, 1) # true   ideology
x   = zeros(num_sim, 1) # chosen ideology position
incumbent = zeros(num_sim+1, 4)
party = zeros(num_sim+1, 4)
governor = zeros(num_sim, 4)

p1 = zeros(num_sim, 4) # observed policie1
p2 = zeros(num_sim, 4) # observed policie2
p3 = zeros(num_sim, 4) # observed policie1
p4 = zeros(num_sim, 4) # observed policie2
p5 = zeros(num_sim, 4) # observed policie1

# initialize
uniform_r = rand(num_sim, 1)
uniform_d = rand(num_sim, 1)

a_r = zeros(num_sim, 1)
a_d = zeros(num_sim, 1)

for i in 1:n_app
    a_d(uniform_d <= cdf_d[i+1] & uniform_d > cdf_d[i]) = i
    a_r(uniform_r <= cdf_r[i+1] & uniform_r > cdf_r[i]) = i
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

incumbent[1,:] = 0
party[1,:] = 1
governor[1,:] = 1

# start generate model from period 1 to num_sim
for i in 1:num_sim
    if incumbent[i,1] == 0
       if party[i,:] == 1
           rho[i] = rho_d[ct1)
           a[i] = a_d[ct1)
           ct1 = ct1+1
           if (l_d2(a[i])>u_d2(a[i])) || (rho[i]<l_d2(a[i])-y_d(a[i])) || (rho[i]>u_d2(a[i])+y_d(a[i]))
               x[i] = rho[i]
               incumbent[i+1,:] = 0
               party[i+1,:] = 2
           elseif rho[i] < l_d2(a[i])
               x[i] = l_d2(a[i])
               incumbent[i+1,:] = 1
           elseif rho[i] > u_d2(a[i])
               x[i] = u_d2(a[i])
               incumbent[i+1,:] = 1       
           else
               x[i] = rho[i]
               incumbent[i+1,:] = 1
           end
       elseif party[i, 1]==2
           rho[i] = rho_r[ct2]
           a[i] = a_r[ct2]
           ct2 = ct2+1          
           if (l_r2(a[i])>u_r2(a[i])) || (rho[i]<l_r2(a[i])-y_r(a[i])) || (rho[i]>u_r2(a[i])+y_r(a[i])) # extremist
               x[i] = rho[i]
               incumbent[i+1, :] = 0
               party[i+1, :] = 1
           elseif rho[i]<l_r2(a[i])   # left moderate
               x[i] = l_r2(a[i])
               incumbent[i+1, :] = 1                 
           elseif rho[i] > u_r2(a[i]) # right moderate
               x[i] = u_r2(a[i])
               incumbent[i+1, :] = 1
           else                      # centerists
               x[i] = rho[i]
               incumbent[i+1, :] = 1                      
           end           
       end
       
sim_max = 5

    elseif incumbent[i, 1] < sim_max
        rho[i] = rho[i-1]
        a[i] = a[i-1]
        party[i,:] = party[i-1, :]
        x[i] = x[i-1]
        incumbent[i+1, :] = incumbent[i, 1]+1
    elseif incumbent[i, 1] == sim_max
        rho[i] = rho[i-1]
        a[i] = a[i-1]
        party[i,:] = party[i-1, :]
        x[i] = x[i-1]
        incumbent[i+1, :] = 0
        party[i+1,:] = randi[2]
    end
    
    if party[i,1] == 1
    p1[i,:] =        x[i]        + p1_d[ct3,:]
    p2[i,:] =        mu_d[2]*x[i] + p2_d[ct3,:]
    p3[i,:] =        mu_d[3]*x[i] + a_grid(a[i])           + p3_d[ct3,:] 
    p4[i,:] =        mu_d[4]*x[i] + mu2_d[4]*a_grid(a[i])  + p4_d[ct3,:] 
    p5[i,:] =        mu_d[5]*x[i] + mu2_d[5]*a_grid(a[i])  + p5_d[ct3,:] 
    ct3 = ct3+1
    
    elseif party[i,1] == 2
    p1[i,:] =        x[i]         + p1_r[ct4,:]
    p2[i,:] =        mu_r[2]*x[i] + p2_r[ct4,:]
    p3[i,:] =        mu_r[3]*x[i] + a_grid(a[i])           + p3_r[ct4,:] 
    p4[i,:] =        mu_r[4]*x[i] + mu2_r[4]*a_grid(a[i])  + p4_r[ct4,:] 
    p5[i,:] =        mu_r[5]*x[i] + mu2_r[5]*a_grid(a[i])  + p5_r[ct4,:]    
        
    ct4 = ct4+1
    end    
end

pr1_r = transpose(collect(LinRange(0, 1, n_app)))
pr1_d = transpose(collect(LinRange(0, 1, n_app)))
pr2_r = transpose(collect(LinRange(0, 1, n_app)))
pr2_d = transpose(collect(LinRange(0, 1, n_app)))

for i in 1:n_app
    pr1_r[i] = quadgk(fun1,l_r2[i],u_r2[i])[1]
    pr1_d[i] = quadgk(fun3,l_d2[i],u_d2[i])[1]
    pr2_r[i] = quadgk(fun1,l_r2[i]-y_r[i],u_r2[i]+y_r[i])-pr1_r[i][1]
    pr2_d[i] = quadgk(fun3,l_d2[i]-y_d[i],u_d2[i]+y_d[i])-pr1_d[i][1]
end
    
display('centrist D and R')
display(sum(pr1_d.*pdf_d) )
display(sum(pr1_r.*pdf_r) )
display('Moderates D and R')
display(sum(pr2_d.*pdf_d) )
display(sum(pr2_r.*pdf_r) )
display('Extremist D and R')
display(sum((1-pr1_d-pr2_d).*pdf_d))
display(sum((1-pr1_r-pr2_r).*pdf_r))


format short
p1_v = p1[:]*std_last[1]
p2_v = p2[:]*std_last[2]
p3_v = p3[:]*std_last[3]*100
p4_v = p4[:]*std_last[4]*100
p5_v = p5[:]*std_last[5]
x_v = x
a_v = a_grid(a)

display('without term limit')

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