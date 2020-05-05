# Bibliotéques utilisées
using LinearAlgebra
using Plots
using Statistics
using Random
using Distributions
using Pkg
using PyCall
using Distributed
# Bibliotéques Python
np = pyimport("numpy")
plt = pyimport("matplotlib.pyplot")


#Most efficient way to compute Gershgorin circles in Julia ?
function cerclesG(A, eps)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    return Disk : Array{Any,1} array of centers Complex{Float64} and rays Complex{Float64}
    =#
    #@time begin
    Disk = []
    n = size(A,1)
    k = sqrt(n)*eps
    # We iterate over every element
    for i = 1:n
        summ = 0
        for j = 1:n
            if i !=j
                summ += abs(A[i,j])

            end
        end
        # Adding in order to the array
        push!(Disk, [A[i,i] summ+k])
    end
    #end
    return Disk
end


# Circle figure used to plot
function circle(center, radius)
    #=
    center :  Complex{Float64}
    radius : Complex{Float64}
    =#
    theta = LinRange(0, 2*π, 500)
    real(center) .+ real(radius)*sin.(theta), imag(center) .+ real(radius)*cos.(theta)
end

# This function is used to plot the circles with the eigen values
function trace_figure(A, eps)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    return Disk : Array{Any,1} array of centers Complex{Float64} and rays Complex{Float64}
    =#
    eig = eigvals(A)
    circles = cerclesG(A, eps)
    p = plot([real(e) for e in eig], [imag(e) for e in eig], seriestype=:scatter,title="GC epsilon : $(eps)")

    for c_ in circles
        p = plot!(circle(c_[1], c_[2]), seriestype = [:shape], lw = 0.5, c = :blue, linecolor = :black, legend = false, fillalpha = 0.2, aspect_ratio = 1)

    end
    # If removed then the function doesn't plot anything
    #display(p)
    return circles, p
end


# On va tracer tracer la fenetre
function build_contour(dots, p; plot_ = true)
    #=
    dots :  Array{Any,1} array of centers Complex{Float64} and rays Complex{Float64}
    p : plot
    plot_ = true : ploting option if false then we only return the window coordinates
    =#

    # We get the centers and rays drom the dots array
    # Maybe something more efficient ?
    centres = [dots[i][1] for i in 1:size(dots,1)]
    rayons = [dots[i][2] for i in 1:size(dots,1)]

    # Computation of the starting point of the cuboide (down-left)
    x = minimum(real(centres)-real(rayons))
    y = minimum(imag(centres)-real(rayons))

    # We compute the coordinates of the upper right point
    k = maximum(real(centres)+real(rayons))
    l = maximum(imag(centres)+real(rayons))

    # Definition of a rectangular shape
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

    # Lenght and height definined such as absolute value of the last points computed in x and y axis
    w = abs(l-y)
    h = abs(k - x)
    if plot_ == true
        p = plot!(rectangle(h,w,x,y), seriestype = [:shape], lw = 0.5, linecolor = :red, legend = false, fillalpha = 0, aspect_ratio = 1)
        # If removed then the function doesn't plot anything
        #display(p)
        return [x y w h], p
    else
        return [x y w h]
    end
end

# Function used to plot circles of gershgorin and the cuboide
function plot_pseudospectre(A, eps; plot_ = true)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    p : plot
    plot_ = true : ploting option if false then we only return the window coordinates
    =#
    @time begin
    circles, p = trace_figure(A, eps)
    contour_, p = build_contour(circles, p)
    if plot_ == true
        return contour_, p
    else
        return contour_
    end
    end
end

# Pseudo meshgrid not used
function make_grid(contour_, nb_points)
    x_ = range(contour_[1],stop = contour_[1] + contour_[3],length = nb_points)
    y_ = range(contour_[2],stop = contour_[2] + contour_[4],length = nb_points)
    return x_, y_
end


# Algo GRID
function grid_sequential(A, eps, nb_points)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    nb_points : Int64 grain of GRID
    =#
    @time begin
    eig = eigvals(A)
    contour_, p = plot_pseudospectre(A, eps)

    # If we don't want to use python fuctions use this instead
    #x, y = make_grid(contour_, nb_points)
    #Y = repeat(y, 1, length(x))
    #X = repeat(reshape(x, 1, :), length(y), 1)

    x = np.linspace(contour_[1],contour_[1] + contour_[3],nb_points)
    y = np.linspace(contour_[2],contour_[2] + contour_[4],nb_points)
    (X,Y) = np.meshgrid(x,y)

    # Main part of the algorithm
    func(x, y) = begin
        C_ = diagm(0=>fill(x + y *1im, size(A,1)))
        minimum(svdvals(C_ .- A))
    end

    Z = map(func, X, Y)

    p_ = plt.contour(X, Y, Z, levels=[0,eps], cmap = "RdYlBu")
    p_ = plt.scatter([real(e) for e in eig], [imag(e) for e in eig])
    plt.show()
    end
end

function grid_par_composante(A, eps, nb_points)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    nb_points : Int64 grain of GRID
    =#
    @time begin
    eig = eigvals(A)
    contour_, p = plot_pseudospectre(A, eps)

    # If we don't want to use python fuctions use this instead
    #x, y = make_grid(contour_, nb_points)
    #Y = repeat(y, 1, length(x))
    #X = repeat(reshape(x, 1, :), length(y), 1)

    x = np.linspace(contour_[1],contour_[1] + contour_[3],nb_points)
    y = np.linspace(contour_[2],contour_[2] + contour_[4],nb_points)
    (X,Y) = np.meshgrid(x,y)

    E = ones(size(A));

    # Main part of the algo
    func(x, y) = begin
        C_ = diagm(0=>fill(x + y *1im, size(A,1)))
        M = abs.(inv(C_ .- A))
        N = M*E
        maximum(abs.(eigvals(N)))
    end

    Z = map(func, X, Y)

    p_ = plt.contour(X, Y, Z, levels=[0,eps], cmap = "RdYlBu")
    p_ = plt.scatter([real(e) for e in eig], [imag(e) for e in eig])
    plt.show()
    end
end

function grid_parallel(A, eps, nb_points, nb_proc)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    nb_points : Int64 grain of GRID
    nb_proc : Int64 number of processes
    =#
    @time begin
    # We add new processes
    addprocs(nb_proc)
    N = div(nb_points, size(workers(), 1))
    sz = size(workers(), 1)

    eig = eigvals(A)
    contour_, p = plot_pseudospectre(A, eps)

    # If we don't want to use python fuctions use this instead
    #x, y = make_grid(contour_, nb_points)
    #Y = repeat(y, 1, length(x))
    #X = repeat(reshape(x, 1, :), length(y), 1)

    x = np.linspace(contour_[1],contour_[1] + contour_[3],nb_points)
    y = np.linspace(contour_[2],contour_[2] + contour_[4],nb_points)
    (X,Y) = np.meshgrid(x,y)

    # Function used by pmap to share work
    func(x, y) = begin
        # Main part of the algorithm
        funz(x,y) = begin
            C_ = diagm(0=>fill(x + y *1im, size(A,1)))
            minimum(svdvals(C_ .- A))
        end

        map(funz, x, y)
    end
    X_ = vcat([X[(i-1)*N+1:N*i,:] for i in 1:sz-1], [X[(sz-1)*N+1:end,:]])
    Y_ = vcat([Y[(i-1)*N+1:N*i,:] for i in 1:sz-1], [Y[(sz-1)*N+1:end,:]])
    Z = pmap((a1,a2)->func(a1,a2), X_, Y_)

    p_ = plt.contour(X, Y, vcat(Z...),  levels=[0, eps], cmap = "RdYlBu")
    p_ = plt.scatter([real(e) for e in eig], [imag(e) for e in eig])
    plt.show()
    end
end

function grid_par_comp_parallel(A, eps, nb_points, nb_proc)
    #=
    A : Array{Complex{Float64},n} matrix of size n x n
    eps : Float64 limit
    nb_points : Int64 grain of GRID
    nb_proc : Int64 number of processes
    =#
    @time begin
    # We add new processes
    addprocs(nb_proc)
    N = div(nb_points, size(workers(), 1))
    sz = size(workers(), 1)

    eig = eigvals(A)
    contour_, p = plot_pseudospectre(A, eps)

    # If we don't want to use python fuctions use this instead
    #x, y = make_grid(contour_, nb_points)
    #Y = repeat(y, 1, length(x))
    #X = repeat(reshape(x, 1, :), length(y), 1)

    x = np.linspace(contour_[1],contour_[1] + contour_[3],nb_points)
    y = np.linspace(contour_[2],contour_[2] + contour_[4],nb_points)
    (X,Y) = np.meshgrid(x,y)
    E = ones(size(A));

    # Function used by pmap to share work
    func(x, y) = begin
        # Main part of the algorithm
        funz(x,y) = begin
            C_ = diagm(0=>fill(x + y *1im, size(A,1)))
            M = abs.(inv(C_ .- A))
            N = M*E
            maximum(abs.(eigvals(N)))
        end

        map(funz, x, y)
    end

    X_ = vcat([X[(i-1)*N+1:N*i,:] for i in 1:sz-1], [X[(sz-1)*N+1:end,:]])
    Y_ = vcat([Y[(i-1)*N+1:N*i,:] for i in 1:sz-1], [Y[(sz-1)*N+1:end,:]])
    Z = pmap((a1,a2)->func(a1,a2), X_, Y_)

    p_ = plt.contour(X, Y, vcat(Z...), levels=[0, eps], cmap = "RdYlBu")
    p_ = plt.scatter([real(e) for e in eig], [imag(e) for e in eig])
    plt.show()
    end
end

# TEST matrix
Test = diagm(0=>[12.137922407708743 + 9.143400412264079im,
  5.597913829753046 + 26.21904779969848im,
 3.6449816521072282 + 9.045310291419105im,
 28.049294614948565 + 14.584550334349101im,
  18.16586787961614 + 20.169126480346918im,
   22.1853137368798 + 37.41137568231745im ,
 10.620933677185871 + 36.014346841676726im,
 22.545898561246915 + 18.3334794370776im  ,
 14.668415979402148 + 21.444454918103688im,
 22.284207345201782 + 22.237880312943666im,
  6.766447450587974 + 30.526724258401792im,
  18.14639906461857 + 3.572167317379977im,
 28.230241830644133 + 26.871540131828716im,
 28.145120689676965 + 46.978904788051274im,
  4.117576436642369 - 0.6792162803388444im]);


Big = diagm(0=>rand(-20:20,100).+rand(-20:20,100).*1im)
