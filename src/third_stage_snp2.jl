module code_third_stage_snp2
include("./src/fun_nlls_snp.jl")
using .fun_nlls_snp

using Optim
using QuadGK
using Distributions

global party3 p_1 p_2 p_d vote op_welfare
global cut_d_grid cut_r_grid xr_grid xd_grid 
global n_grid m_grid x_grid
global theta vvr vvd 
global u_d l_d u_r l_r xt_1 at_1 u_d2 l_d2 u_r2 l_r2
global pud pld pur plr
global n1 prob p_grid
global n_app a_grid
global sigr sigd
global mu_d mu_r y_d y_r
global pdf_d pdf_r pd_pos pr_pos ecost_d ecost_r
global fxr fxd datafile3

party3 = datafile3[:,1]
vote = datafile3[:,2]
p_1 = datafile3[:,3]
p_2 = datafile3[:,4]

n_grid = 10
m_grid = 10
x_grid = transpose(collect(LinRange(-10,10,m_grid)))
vvd = transpose(collect(LinRange(-10,10,m_grid)))
vvr = transpose(collect(LinRange(-10,10,m_grid)))
xr_grid = zeros(n_app,n_grid)
xd_grid = zeros(n_app,n_grid)
cut_d_grid = zeros(n_app,n_grid)
cut_r_grid = zeros(n_app,n_grid)

pd_pos = sum(pdf_d.*(u_d > l_d))
pr_pos = sum(pdf_r.*(u_r > l_r))


for k in 1:n_app
    xr_grid[k,:] = transpose(collect(LinRange(l_r[k],u_r[k],n_grid)))
    xd_grid[k,:] = transpose(collect(LinRange(l_d[k],u_d[k],n_grid)))
end

options = optimset('Display','off')
x0 = [-2;-2]

for i in 1:m_grid
    theta = x_grid[i]
    [x,~]= fsolve(@v_func_snp,x0,options)
    vvd[i] = x[1]
    vvr[i] = x[2]
end

x0 = 0
for k in 1:n_app
    if u_d[k] > l_d[k]
        for j in 1:n_grid
            xt_1 = xd_grid[k,j]
            at_1 = a_grid[k]
            [xx,~] = fsolve(@v_diff_d,x0,options)
            cut_d_grid[k, j] = xx
        end
    end
end

x0 = 0
for k in 1:n_app
    if u_r[k] > l_r[k]
        for j in 1:n_grid
            xt_1 = xr_grid[k, j]
            at_1 = a_grid[k]
            [xx,~] = fsolve(@v_diff_r,x0,options)
            cut_r_grid[k, j] = xx
        end
    end
end

fr = y -> (fxr[1] + fxr[2]*(y-fxr[6]) + fxr[3]*(y-fxr[6]).^2 + fxr[4]*(y-fxr[6]).^3 +fxr[5]*(y-fxr[6]).^4).^2.*exp(-(y-fxr[6]).^2/fxr[7]^2)
fd = y -> (fxd[1] + fxd[2]*(y-fxd[6]) + fxd[3]*(y-fxd[6]).^2 + fxd[4]*(y-fxd[6]).^3 +fxd[5]*(y-fxd[6]).^4).^2.*exp(-(y-fxd[6]).^2/fxd[7]^2)

for k in 1:n_app
    pud[k] = quadgk(fd,u_d[k], u_d[k]+y_d[k])[1]
    pld[k] = quadgk(fd,l_d[k]-y_d[k], l_d[k])[1]
    pur[k] = quadgk(fr,u_r[k], u_r[k]+y_r[k])[1]
    plr[k] = quadgk(fr,l_r[k]-y_r[k], l_r[k])[1]
end

n1 = size(party3, 1)
p_grid =zeros(n1,n_app,n_grid)

for i in 1:n1
    if party3[i] == 1 
        for k in 1:n_app
            if u_d[k] > l_d[k]         
                for j in 1:n_grid
                    xt_1 = xd_grid[k,j]
                    p_grid[i,k,j] = fd(xt_1)*pdf(Normal(0,sigd[1]), p_1[i]-xt_1)*pdf(Normal(0,sigd[2]),p_2[i]-mu_d[2]*xt_1)

                end
                    prob[i,k] = (pud[k]*pdf(Normal(0,sigd[1]), p_1[i]-u_d[k])*pdf(Normal(0,sigd[2]), p_2[i]-mu_d[2]*u_d[k])+pld[k]*pdf(Normal(0,sigd[1]), p_1[i]-l_d[k])*pdf(Normal(0,sigd[2]), p_2[i]-mu_d[2]*l_d[k])+trapz(xd_grid[k,:],p_grid[i,k,:]))    
            end
        end
    else
        for k in 1:n_app
            if u_r[k] > l_r[k]
                for j in 1:n_grid
                    xt_1 = xr_grid[k,j]
                    p_grid[i,k,j]=fr(xt_1)*pdf(Normal(0,sigr[1]), p_1[i]-xt_1)*pdf(Normal(0,sigr[2]), p_2[i]-mu_r[2]*xt_1)
                end
                    prob[i,k] = (pur[k]*pdf(Normal(0,sigr[1]), p_1[i]-u_r[k])*pdf(Normal(0,sigr[2]), p_2[i]-mu_r[2]*u_r[k])+plr[k]*pdf(Normal(0,sigr[1]), p_1[i]-l_r[k])*pdf(Normal(0,sigr[2]), p_2[i]-mu_r[2]*l_r[k])+trapz(xr_grid[k,:],p_grid[i,k,:]))
            end
        end
    end
end
      
# Estimate distribution of voters ideology   
x0=2.5
xx = optimize(nlls_snp,x0).minimizer
display(xx)

# plot
t9 = transpose(collect(LinRange(-8,8,100)))
plot2  = fr(t9)
plot1  = fd(t9)
plot3 = pdf(Normal(0,xx), t9)
plot4 = plot1*p_d + plot2*(1-p_d)

figure
plot(t9,plot3,'b-','LineWidth',2)
hold on
plot(t9,plot4,'r--','LineWidth',2)
hleg = legend('Voter','Candidate')
set(hleg, 'Box','off','Location','NorthEast')
xlabel('ideology')
axis([-8 8 0 0.4])


if op_welfare == 1 # Calculate welfare under 2-term limit and no term limit
display('welfare analysis')
grid = transpose(collect(LinRange(-6, 6, 101)))
vv = zeros(2,101)
vv_ntl = zeros(2,101)
v = zeros(2,101)
w = zeros(2,101)
# cost  x 2
pr3_r = transpose(collect(LinRange(0, 1, n_app)))
pr3_d = transpose(collect(LinRange(0, 1, n_app)))

num_ecost = 5
ecost_grid_d = transpose(collect(LinRange(0, 0.5, num_ecost)))
ecost_grid_r = transpose(collect(LinRange(0, 0.5, num_ecost)))
xx_grid = ecost_grid_d
welfare0 = zeros(1,num_ecost)
welfare1 = welfare0
welfare2 = welfare0

frac1_d = welfare0
frac1_r = welfare0
frac2_d = welfare0
frac2_r = welfare0

cost1_d = welfare0
cost1_r = welfare0
cost2_d = welfare0
cost2_r = welfare0

for jj in 1:num_ecost
    ecost_d = ecost_grid_d[jj]
    ecost_r = ecost_grid_r[jj]

    find_standard_ttl

    for i in 1:n_app
        if l_r[i] <= u_r[i]
            pr3_r[i] = quadgk(fr,l_r[i]-y_r[i], u_r[i]+y_r[i])[1]
        else
            pr3_r[i] = 0
        end  
        if l_d[i] <= u_d[i]
            pr3_d[i] = quadgk(fd,l_d[i]-y_d[i], u_d[i]+y_d[i])[1]
        else
            pr3_d[i] = 0
        end   
    end

    frac1_d[jj] = sum(pr3_d.*pdf_d)
    frac1_r[jj] = sum(pr3_r.*pdf_r)
    beta = 0.8

    cost1_d[jj] =-beta*ecost_d*(frac1_d[jj]*(1-frac1_r[jj]*beta^2*(1-p_d)) + frac1_r[jj]*(frac1_d[jj]*beta^2*(1-p_d)+(1-frac1_d[jj])*beta))/((1-frac1_d[jj]*beta^2*p_d)*(1-frac1_r[jj]*beta^2*(1-p_d))-(frac1_d[jj]*beta^2*(1-p_d)+(1-frac1_d[jj])*beta)*(frac1_r[jj]*beta^2*p_d+(1-frac1_r[jj])*beta))
    cost1_r[jj] = (-beta*ecost_d*frac1_r[jj]+cost1_d[jj]*(frac1_r[jj]*beta^2*p_d+(1-frac1_r[jj])*beta))/(1-frac1_r[jj]*beta^2*(1-p_d))

    options = optimset('Display','off')
    x0 = [-2; -2]
    for i in 1:101
        theta = grid[i]
        [x,fval] = fsolve(@v_func_snp,x0,options)
        vv[1,i] = x[1]
        vv[2,i] = x[2]
        x0[1] = x[1]
        x0[2] = x[2]
    end

    find_standard_ntl

    for i in 1:n_app
        if l_r2[i] <= u_r2[i]
            pr3_r[i] = quadgk(fr,l_r2[i]-y_r[i], u_r2[i]+y_r[i])[1]
        else
            pr3_r[i] = 0
        end  

        if (l_d2[i] <= u_d2[i])
            pr3_d[i] = quadgk(fd,l_d2[i]-y_d[i], u_d2[i]+y_d[i])[1]
        else
            pr3_d[i] = 0
        end   
    end

    frac2_d[jj] = sum(pr3_d.*pdf_d)
    frac2_r[jj] = sum(pr3_r.*pdf_r)
    beta = 0.8

    cost2_d[jj] = (-beta*ecost_d/(1-beta))*(frac2_d[jj]+beta*(1-frac2_d[jj])*frac2_r[jj])/(1-beta^2*(1-frac2_d[jj])*(1-frac2_r[jj]))
    cost2_r[jj] = (-beta*ecost_d/(1-beta))*frac2_r[jj] + (1-frac2_r[jj])*beta*cost2_d[jj]

    options = optimset('Display','off')
    x0 = [-2; -2]
    for i in 1:101
        theta = grid[i]
        [x,fval]= fsolve(@v_func_snp_ntl,x0,options)
        vv_ntl[1,i] = x[1]
        vv_ntl[2,i] = x[2]
        x0[1]=x[1]
        x0[2]=x[2]
    end

    y1 =(p_d*vv[1, :]+(1-p_d)*vv(2,:]).*pdf(Normal(0,xx),grid)
    y2 =(p_d*vv_ntl[1, :]+(1-p_d)*vv_ntl(2,:]).*pdf(Normal(0,xx),grid)
    welfare1[jj] = trapz(grid,y1)
    welfare2[jj] = trapz(grid,y2)
end

welfare0[:] =  welfare1[1]

frac1 = p_d*frac1_d + (1-p_d)*frac1_r
frac2 = p_d*frac2_d + (1-p_d)*frac2_r

cost1 = p_d*cost1_d + (1-p_d)*cost1_r
cost2 = p_d*cost2_d + (1-p_d)*cost2_r

figure
plot(xx_grid,frac1,'k--','LineWidth',2)
hold on
plot(xx_grid,frac2,'r:','LineWidth',2)
hleg = legend('2 term limit','no term limit')
set(hleg, 'Box','off','Location','NorthEast')
xlabel ('negative tenure effect')
ylabel ('reelection probability')
axis([min(xx_grid) max(xx_grid) 0 1])

figure
plot(xx_grid,cost1,'k--','LineWidth',2)
hold on
plot(xx_grid,cost2,'r:','LineWidth',2)
hleg = legend('2 term limit','no term limit')
set(hleg, 'Box','off','Location','NorthEast')
xlabel ('negative tenure effect')
ylabel ('lifetime costs of voter fatigue')

figure
plot(xx_grid,welfare1,'k--','LineWidth',2)
hold on
plot(xx_grid,welfare2,'r:','LineWidth',2)
hleg = legend('2 term limit','no term limit')
set(hleg, 'Box','off','Location','NorthEast')
xlabel ('negative tenure effect')
ylabel ('welfare')

end

end