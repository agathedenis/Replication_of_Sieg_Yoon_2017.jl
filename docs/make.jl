push!(LOAD_PATH,"../src/")
using Documenter, Replication_of_Sieg_Yoon_2017

makedocs(modules = [Replication_of_Sieg_Yoon_2017], sitename = "Replication of Sieg and Yoon (2017)")

deploydocs(repo = "github.com/vho97/Replication_of_Sieg_Yoon_2017.jl.git", devbranch = "main")
