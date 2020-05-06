#Bibliotéques utilisées
using LinearAlgebra
using Plots
using Distributions
import Base.Threads


function triplet(val,A)
    C=diagm(0=>fill(val, size(A,1)))
    U, S, V =svd(C-A)
    #les valeurs singulieres sont trié par ordre decroissant dans S, on choisis la plus petite
    smin=S[size(A,1)]
    umin=U[:,size(A,1)]
    vmin=V[:,size(A,1)]
    return [smin,umin,vmin]
end
function prediction_correction(A,eps=1,nbpoints=10000,d=1im,tol=1,steplength=0.1)
    @time begin


    #step:0
    #transconjugué
    eigval=eigvals(A)
    p = plot([real(eig) for eig in eigval], [imag(eig) for eig in eigval], seriestype=:scatter,c=:red)
    n=size(eigval,1)
    for i=1: n

        eig=eigval[i]
        teta=eps
        z_new=eig+teta*d
        t=[]
        t=triplet(z_new,A)
        while abs(triplet(z_new,A)[1]-eps)>tol*eps
            z_old=z_new
            t=triplet(z_old,A)
            z_new=(t[1]-eps)/real(-1im*(conj(transpose(t[3]))*t[2]))*d
        end

        z=z_new
        pseudospectre=[]

        for k=1:nbpoints
            #step:1
            r=1im*((conj(transpose(t[3]))*t[2])/abs((conj(transpose(t[3]))*t[2])))
            z=z+steplength*r
            #step:2
            t=triplet(z,A)
            z=z-(t[1]-eps)/(conj(transpose(t[2]))*t[3])
            push!(pseudospectre,z)
        end
        p = plot!([real(e) for e in pseudospectre], [imag(e) for e in pseudospectre], seriestype=:scatter,title="pseudospectre",c=:blue)

        p = plot!([real(eig)], [imag(eig)], seriestype=:scatter,c=:red)
    end
    end
    display(p)
    return 0


end

function generate_random_matrix(a=-1,b=1; n=10, complex= true)
    #R = rand(a:b,n, n)
    R = rand(Uniform(a,b),n,n)
    if complex == true
        #I = rand(a:b,n, n)
        I = rand(Uniform(a,b),n,n)
        return R + I*im
    end
    return R
end
#A = [1+2im 3+4im 10; 4 + 3im 1 20; 3 + 4im 30 1]
#C = [ 3+1im  -1.5  0  1.5im; 0.5  4  1im  0.5im; sqrt(2) sqrt(2)*1im  2+3im 0; 1im  1  1im 4im]
#D = [3.5im  1  0.5; 1  2  1; 1 1  1]



function prediction_correction2(A,eps=1,nbpoints=10000,d=1im,tol=1,steplength=0.1)
    @time begin
    #step:0
    #transconjugué
    eigval=eigvals(A)
    p = plot([real(eig) for eig in eigval], [imag(eig) for eig in eigval], seriestype=:scatter,c=:red)
    n=size(eigval,1)
    Threads.@threads for i=1: n
        println("i = $i on thread $(Threads.threadid())")
        eig=eigval[i]
        teta=eps
        z_new=eig+teta*d
        t=[]
        t=triplet(z_new,A)
        while abs(triplet(z_new,A)[1]-eps)>tol*eps
            z_old=z_new
            t=triplet(z_old,A)
            z_new=(t[1]-eps)/real(-1im*(conj(transpose(t[3]))*t[2]))*d
        end

        z=z_new
        pseudospectre=[]

        for k=1:nbpoints
            #step:1
            r=1im*((conj(transpose(t[3]))*t[2])/abs((conj(transpose(t[3]))*t[2])))
            z=z+steplength*r
            #step:2
            t=triplet(z,A)
            z=z-(t[1]-eps)/(conj(transpose(t[2]))*t[3])
            push!(pseudospectre,z)
        end
        p = plot!([real(e) for e in pseudospectre], [imag(e) for e in pseudospectre], seriestype=:scatter,title="pseudospectre",c=:blue)

        p = plot!([real(eig)], [imag(eig)], seriestype=:scatter,c=:red)
    end
    end
    display(p)
    return 0


end



