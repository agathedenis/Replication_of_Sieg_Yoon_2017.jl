module Replication_of_Sieg_Yoon_2017
# Replication code for Sieg and Yoon (2017)

using DataFrames
using CSV
using Statistics
using Plots
using GLM
using Optim
using Trapz
using LinearAlgebra
using JuMP
using Distributions
using QuadGK
using NLopt

# Options TO BE MADE INTERACTIVE
op_model = 1  # 1 : Baseline model specification / 2 : lambda = 0 / 3 : extended
op_estimation = 0 # 0: simulation of 2nd stage, 1-2 : esimation of 2nd stage (takes long time)

# Set the value of parameters
num_sim = 10000 # Number of simulations
beta = 0.8      # Fixed discount factor
n_app = 5       # Number of ability grids
a_max = 1.2     # Max ability grids
a_grid = transpose(collect(LinRange(-a_max,a_max,n_app))) # Ability grids
num_eval = 5000 

# Read data
datafile = CSV.read("./src/datafile6.csv", DataFrame; header=false)
party2 = datafile[:,1]
election_number = datafile[:,7]
election_code = datafile[:,8]
vote_share = datafile[:,9]
state = datafile[:,10]
# Normalize data
std_last = zeros(5, 1)
for i in 1:5     
    ps = datafile[:,1+i]        
    std_last[i] = std(ps)
    datafile[:,1+i] = ps/std_last[i]
end

# initial guess
mu_d = zeros(1,5)
mu2_d = zeros(1,5)
mu_r = zeros(1,5)
mu2_r = zeros(1,5)

global itr = 1
dist = 1
while dist > 1e-10 
    loading_old = sum([mu_d[2], mu_d[3], mu_d[4], mu_d[5], mu2_d[4], mu2_d[5]])
    # factor loading
    # state fixed effects
    y1 = datafile[:,2]
    y2 = datafile[:,3]
    y3 = datafile[:,4]
    y4 = datafile[:,5]
    y5 = datafile[:,6]
    n_all = size(y1)[1]
    # Estimation of idology using y1 and y2
    Y1 = datafile[election_number.>1, 2]
    Y2 = datafile[election_number.>1, 3]
    n = size(Y1)[1]
    x = zeros(n,24)
    for i in 1:24
        x[:,i] = state[election_number.>1].==i # state dummy
    end
    X1 = zeros(n,24)
    X2 = zeros(n, 24)
    for i in 1:n
        X1[i, :] = x[i,:]
        X2[i, :] = x[i,:]*mu_d[2]
    end
    beta1 = 0.5*(coef(lm(X1+X2,Y1))+coef(lm(X1+X2,Y2)))
    ideology = zeros(n_all, 1)
    for i in 1:n_all
        for j in 1:24
            if state[i] == j
                ideology[i] = beta1[j]
            end
        end
    end
    residual = zeros(n_all, 5)
    residual[:,1] = y1 - ideology
    residual[:,2] = y2 - mu_d[2]*ideology
    y3 = y3 - mu_d[3]*ideology
    y4 = y4 - mu_d[4]*ideology
    y5 = y5 - mu_d[5]*ideology
    Y3 = datafile[election_number.>1, 4]
    Y4 = datafile[election_number.>1, 5]
    Y5 = datafile[election_number.>1, 6]
    X3 = zeros(n,24)
    X4 = zeros(n,24)
    X5 = zeros(n,24)
    for i in 1:n
        X3[i, :] = x[i,:]
        X4[i, :] = x[i,:]*mu2_d[2]
        X5[i, :] = x[i,:]*mu2_d[5]
    end
    beta2 = 1/3*(coef(lm(X3+X4+X5,Y3))+coef(lm(X3+X4+X5,Y4))+coef(lm(X3+X4+X5,Y5)))
    ability = zeros(n_all, 1)
    for i in 1:n_all
        for j in 1:24
            if state[i] == j
                ability[i] = beta2[j]
            end
        end
    end
    residual[:,3] = y3 - ability
    residual[:,4] = y4 - mu2_d[4]*ability
    residual[:,5] = y5 - mu2_d[5]*ability
    covariance = cov(residual)
    mu_d[2] = covariance[2,3]/covariance[1,3]
    sigma_rho = covariance[1,2]/mu_d[2]
    mu_d[3] = covariance[1,3]/sigma_rho
    mu_d[4] = covariance[1,4]/sigma_rho
    mu_d[5] = covariance[1,5]/sigma_rho
    mu2_d[4] = (covariance[4,5]-mu_d[4]*mu_d[5]*sigma_rho)/(covariance[3,5]-mu_d[3]*mu_d[5]*sigma_rho)
    sigma_a = (covariance[3,4] -mu_d[3]*mu_d[4]*sigma_rho)/mu2_d[4]
    mu2_d[5] = (covariance[3,5]-mu_d[3]*mu_d[5]*sigma_rho)/sigma_a
    mu_r = mu_d
    mu2_r = mu2_d
    loading_new = sum([mu_d[2], mu_d[3], mu_d[4], mu_d[5], mu2_d[4], mu2_d[5]])
    dist = max(abs(loading_old - loading_new))
    global itr = itr + 1
end

# State names: those limited to 2 consecutive terms (NM and OR changed it)
statename = ["AL"; "AZ"; "CO"; "FL"; "GA"; "IN"; "KS"; "KY"; "LA"; "ME"; "MD"; "NE"; "NJ"; "NM"; "NC"; "OH"; "OK"; "OR"; "PA"; "RI"; "SC"; "SD"; "TN"; "WV"]

fig1a = plot(beta1,beta2,seriestype= :scatter, legend= false, label = statename,
        title = "State Fixed Effects (simulated)", ylabel="Ideology", xlabel="Competence",
        series_annotations=(text.(statename)), markersize=10, color = :yellow)
display(fig1a)
png(fig1a,"./docs/fig1a")

# True values
ability = CSV.read("./src/ability.csv", DataFrame; header=false)
ideology = CSV.read("./src/ideology.csv", DataFrame; header=false)
residual = CSV.read("./src/residual.csv", DataFrame; header=false)
sigma_rho = 0.576662899359854
beta1 = [-0.222113983732354	-0.848058595993730	-1.99882870162214	-0.0325236132515880	-0.0888557616323525	0.0525026613939720	-0.395819828791645	0.816175217364974	-0.658549920369977	0.117148315308067	-0.133681685282318	-0.0194687341022869	0.193427672460232	0.119830294335563	0.327433891903263	0.122428491896759	-0.381153544843729	-0.130921290143661	0.0889288412940485	0.808516709556322	-0.0111320989187031	-0.263294213185911	-1.05100390838971	0.288237736174178]'
beta2 = [-0.0381512360423356	-0.345442620547150	-0.736791775204788	0.104761592164850	-0.0625471608375573	0.0626941828624524	0.0642681091470936	-0.0217397688601464	-0.209242151310862	-0.0670046223146057	-0.0171982527493006	-0.233696739708889	0.0605499987561971	0.133720107032712	0.446437701737977	-0.211412067173662	0.152583936834852	0.000793584402733982	-0.291672689334009	0.184879987237118	0.554769458212349	0.338473089552399	-0.439400740106697	0.0424734726214156]'
mu2_r = [0	0	0	-0.400068404146344	-0.0595399668386188]
mu2_d = mu2_r
mu_d = [0	0.703295074242790	-0.146931937949867	0.110838880139945	-0.0181959564331241]
mu_r = mu_d
se_d = [0, 0.026, 0.006, 0.005, 0.006]
se2_d = [0, 0.094, 0.029]

fig1b = plot(beta1,beta2,seriestype= :scatter, legend= false, label = statename,
        title = "State Fixed Effects (real)", ylabel="Ideology", xlabel="Competence",
        series_annotations=(text.(statename)), markersize=10, color = :yellow)
display(fig1b)
png(fig1b,"./docs/fig1b")
# Data moments

d_share = sum(min.(party2 .==1, election_number .==3))/sum(min.(party2 .==1, election_number .>= 2))
r_share = sum(min.(party2 .==2, election_number .==3))/sum(min.(party2 .==2, election_number .>= 2))

share_all = zeros(3,4)

for i = 1
        ps = residual[:, 3]
        std4 = std(ps)
        share_all[i,1] = (sum(min.(election_number .==3, ps .< -1*std4)))/(sum(min.(election_number .!=2, ps .< -1*std4)))
        share_all[i,2] = (sum(min.(election_number .==3, ps .>= -1*std4, ps .< 0)))/(sum(min.(election_number .!=2, ps .>= -1*std4, ps .< 0)))
        share_all[i,3] = (sum(min.(election_number .==3, ps .< 1*std4, ps .>= 0)))/(sum(min.(election_number .!=2, ps .< 1*std4, ps .>= 0)))
        share_all[i,4] = (sum(min.(election_number .==3, ps .>= 1*std4)))/(sum(min.(election_number .!=2, ps .>= 1*std4)))
end

mean_1 = zeros(5,2)
mean_2 = zeros(5,2)
mean_3 = zeros(5,2)
mean_all = zeros(5,2)
std_1 = zeros(5,2)
std_2 = zeros(5,2)
std_3 = zeros(5,2)
std_all = zeros(5,2)

for i in 1:5
    for j in 1:2       
        ps = residual[:,i]        
        mean_1[i,j] = mean(ps[election_number.==1 .& party2.==j])
        mean_2[i,j] = mean(ps[election_number.==2 .& party2.==j])
        mean_3[i,j] = mean(ps[election_number.==3 .& party2.==j])        
        mean_all[i,j] = mean(ps[party2.==j])
        std_1[i,j] = std(ps[election_number.==1 .& party2.==j])
        std_2[i,j] = std(ps[election_number.==2 .& party2.==j])
        std_3[i,j] = std(ps[election_number.==3 .& party2.==j])        
        std_all[i,j] = std(ps[party2.==j])
    end
end

# First Stage Estimation (Kotlarski)
demo = CSV.read("./src/demo.csv", DataFrame; header=false)
repub = residual[election_number.>=2 .& party2.==2, :]
datafile3 = hcat(party2[election_number.==1], vote_share[election_number.==1], residual[election_number.==1,1],residual[election_number.==1,2])
m = 10*(floor(10-(-10))+1)       
t3 = collect(LinRange(-10,10,m))
t4 = collect(LinRange(-30,30,m))
ted =[t3 t3 t3 t3 t4]
ter =[t3 t3 t3 t3 t4]
kad = CSV.read("./src/kad.csv", DataFrame; header=false)
kar = CSV.read("./src/kar.csv", DataFrame; header=false)
kxd = CSV.read("./src/kxd.csv", DataFrame; header=false)
kxr = CSV.read("./src/kxr.csv", DataFrame; header=false)
ked = CSV.read("./src/ked.csv", DataFrame; header=false)
ker = CSV.read("./src/ker.csv", DataFrame; header=false)

# Approximate distibution of ideology and competence using SNP density
# Function that fits the SNP estimator, proposed by Gallant and Nychka (1987),
# a nonparametric method used here to approximate distribution of ideology and competence
# Objective function
function snp_fit(x,grad)
    x0 = x[1]
    x1 = x[2]
    x2 = x[3]
    x3 = x[4]
    x4 = x[5]
    mu = x[6]
    sig = x[7]
    
    f1  = (x0 .+ x1.*(t3.-mu) .+ x2.*(t3.-mu).^2 .+ x3.*(t3.-mu).^3 .+ x4.*(t3.-mu).^4).^2 .*exp.(-(t3.-mu).^2/sig.^2)

    if mode == 1
        f2 = Array(kxr)
    elseif mode == 2
        f2 = Array(kxd)
    elseif mode == 3
        f2 = Array(kar)
    elseif mode == 4
        f2 = Array(kad)
    end
    
    f = sum((f1-f2).^2)

    return f
end

function mycon(x, grad)
    x0 = x[1]
    x1 = x[2]
    x2 = x[3]
    x3 = x[4]
    x4 = x[5]
    mu = x[6]
    sig = x[7]
    
    ceq = x0^2*sqrt(pi)*sig + (x1^2+2*x0*x2)*sqrt(pi)*sig^3/2 + (x2^2 + 2*x0*x4 + 2*x1*x3)*sqrt(pi)*sig^5*3/4 + (x3^2 + 2*x2*x4)*sqrt(pi)*sig^7*15/8 + (x4^2)*sqrt(pi)*sig^9*105/16 - 1

    return ceq
end

xx_vector = zeros(4,7)
x0=[0.5880    0.0087   -0.0617   -0.0004    0.0048   -0.2016    2.0044
    0.6078   -0.0429   -0.0723    0.0026    0.0077    0.2638    1.7891
   -0.7009    0.0033    0.1973   -0.0001   -0.0083   -0.0504    2.1520
    0.7060    0.0070   -0.2045   -0.0003    0.0089   -0.0088    2.1206]
for mode in 1:4  
    opt1 = Opt(:LN_COBYLA, 7)

    min_objective!(opt1, snp_fit)
    equality_constraint!(opt1, mycon)

    xx_vector[mode,:] = NLopt.optimize(opt1, x0[mode,:])[2]
end
fxr = xx_vector[1, :]
fxd = xx_vector[2, :]
far = xx_vector[3, :]
fad = xx_vector[4, :]

# Approximate distibution of error terms using Normal Density
x0v = [1 1 1 1 15]

sigr = zeros(5,1)
sigd = zeros(5,1)

function normal_fit_d(x,mode)
    mu   =  0
    sig  = x
    for i in 1:size(ted[:,mode])[1]
        f1[i] = (1/sqrt(2*pi*sig^2)).*exp.(-(ted[:,mode][i].-mu).^2/sig.^2)
    end
    f2  = ked[:, mode]
    f = sum((f1-f2).^2)

    return f
end

function normal_fit_r(x,mode)
    mu   =  0
    sig  = x
    for i in 1:size(ter[:,mode])[1]
        f1[i] = (1/sqrt(2*pi*sig^2)).*exp.(-(ter[:,mode][i].-mu).^2/sig.^2)
    end
    f2  = ker[:, mode]
    f = sum((f1-f2).^2)

    return f
end

for i in 1:5
    mode = round(i)
    x0 = x0v[i]
    #sigr[i] = Optim.optimize(normal_fit_r, [x0,mode])
    # sigr[i] = Optim.optimize(normal_fit_r, [x0,mode])
    # Doesn't work
end

sigd = [0.849609375000000, 1.24169921875000, 0.930957031250000, 0.859667968750000, 7.24859619140625]
sigr = [0.735937500000000, 1.30947265625000, 0.939550781250000, 0.806542968750000, 8.99615478515625]

sigd[2] = sigd[2]*mu_d[2]
sigd[3] = sqrt(sigd[3]^2-(mu_d[3]*sigd[1])^2)
sigd[4] = sqrt((mu2_d[4]*sigd[4])^2-(mu_d[4]*sigd[2]/mu_d[2])^2)
sigd[5] = sqrt((mu2_d[5]*sigd[5])^2-(mu_d[5]*sigd[1])^2)
sigr[2] = sigr[2]*mu_r[2]
sigr[3] = sqrt(sigr[3]^2-(mu_r[3]*sigr[1])^2)
sigr[4] = sqrt((mu2_r[4]*sigr[4])^2-(mu_r[4]*sigr[2]/mu_r[2])^2)
sigr[5] = sqrt((mu2_r[5]*sigr[5])^2-(mu_r[5]*sigr[1])^2)

# Approximate distribution of competence using discrete grids
f15  = y -> (far[1] .+ far[2]*(y.-far[6]) .+ far[3]*(y.-far[6]).^2 .+ far[4]*(y.-far[6]).^3 .+far[5]*(y.-far[6]).^4).^2 .*exp.(-(y.-far[6]).^2/far[7]^2)
f16  = y -> (fad[1] .+ fad[2]*(y.-fad[6]) .+ fad[3]*(y.-fad[6]).^2 .+ fad[4]*(y.-fad[6]).^3 .+fad[5]*(y.-fad[6]).^4).^2 .*exp.(-(y.-fad[6]).^2/fad[7]^2)

cdf_r = transpose(collect(LinRange(0, 1, n_app+1)))
cdf_d = transpose(collect(LinRange(0, 1, n_app+1)))
a_dist = a_grid[2] - a_grid[1]
for i in 2:n_app
    cdf_r[i] = quadgk(f15,-Inf,a_grid[i-1]+a_dist*0.5)[1]
    cdf_d[i] = quadgk(f16,-Inf,a_grid[i-1]+a_dist*0.5)[1]
end
pdf_r = cdf_r[2:n_app+1] - cdf_r[1:n_app]
pdf_d = cdf_d[2:n_app+1] - cdf_d[1:n_app]

# Plots distribution of ideology and competence by party
fun1 = y -> (fxr[1] .+ fxr[2]*(y.-fxr[6]) .+ fxr[3]*(y.-fxr[6]).^2 .+ fxr[4]*(y.-fxr[6]).^3 .+fxr[5]*(y.-fxr[6]).^4).^2 .*exp.(-(y.-fxr[6]).^2/fxr[7]^2)
fun3 = y -> (fxd[1] .+ fxd[2]*(y.-fxd[6]) .+ fxd[3]*(y.-fxd[6]).^2 .+ fxd[4]*(y.-fxd[6]).^3 .+fxd[5]*(y.-fxd[6]).^4).^2 .*exp.(-(y.-fxd[6]).^2/fxd[7]^2)

plot(t3,fun1(t3), xlabel="Ideology", xlims=(-4,4), ylims=(0,0.6), title="Distribution of ideology by party",color="red", label="Republican")
fig2 = plot!(t3,fun3(t3), line= :dash, label="Democrat", color="blue")
display(fig2)
png(fig2, "./docs/fig2")
plot(t3,f15(t3), xlims=(-4,4), ylims=(0,0.6), ylabel="Competence", title="Distribution of competence by party",color="red", label="Republican")
fig3 = plot!(t3,f16(t3), line= :dash, label="Democrat", color="blue")
png(fig3, "./docs/fig3")

# 2nd stage using SMM

w_diag = [0.0280439; 0.0302329; 0.0473946; 0.1031027; 0.0809237; 0.0704786; 0.0344337; 0.0301466; 0.0650172]
weight = inv(diagm(w_diag.^2))

# Load testing values
if op_model == 1
    xx = [0.613830043757023
    0.837176997391922
    0.218441841751121
    0.272728922517451
    0.0363090286173462
    1.55729424043509]
elseif op_model == 2
    xx = [0.328229621688830
    0.624053970167049
    1.30293931253859]
else
    xx = [0.752711727106632
    0.791278559891922
    0.202450630813621
    0.227274897248897
    0.0363090286173462
    1.55730568452688]
end

x0=xx
display("First-stage estimates")
display("Factor loadings: ideology")
display("Expenditure:")
display(1)
display("Taxes:")
display(mu_d[2])
display("Economic growth:")
display(mu_d[3])
display("Debt costs:")
display(mu_d[4])
display("Workers' compensation:")
display(mu_d[5])
display("Factor loadings: competence")
display("Economic growth:")
display(1)
display("Debt costs:")
display(mu2_d[4])
display("Workers' compensation:")
display(mu2_d[5])

op_policyexp        = 0
op_thirdstage       = 0 
op_valuefunction    = 0 
op_welfare          = 0 
    
if op_estimation==0 && op_model==1
    op_policyexp        = 1 
    op_thirdstage       = 1 
    op_valuefunction    = 1 
    op_welfare          = 1 
end

#=if op_estimation == 1
    op_print_results    = 0
    options = optimset('Display','iter','MaxFunEvals',num_eval)
    [xx,fval] = fminsearch(@smm12,x0,options)
    op_print_results = 1
    smm12(xx)
elseif op_estimation==2
    op_print_results    = 0
    options = psoptimset('Display','iter','MaxFunEvals',num_eval,'TolX',1d-6)
    A =[]    b = []    Aeq = []    beq = []
    LB = x0 - abs(x0)*0.2
    UB = x0 + abs(x0)*0.2
    xx = patternsearch(@smm12,x0,A,b,Aeq,beq,LB,UB,options)
    op_print_results = 1
    smm12(xx)
else
    display('2nd stage estimate')
    display(xx)
    op_print_results = 1
    smm12(x0)
end=#

end