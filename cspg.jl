module CSPG

export cspg

using Printf, CUTEst, NLPModels

const libcspg = abspath(joinpath(@__DIR__, "libcspg.so"))

struct Box{T,N,L<:AbstractArray{T,N},U<:AbstractArray{T,N}}
    lower::L
    upper::U
end

function _cspg(func, func_data, grad, grad_data, proj, proj_data, obsv, obsv_data,
               m, n, x, epsopt, maxit, maxfc, verb,
               fx, gpsupn, iter, fcnt, gcnt, pcnt, status, inform)
    @ccall libcspg.cspg(func::Ptr{Cvoid}, func_data::Ptr{Cvoid},
                        grad::Ptr{Cvoid}, grad_data::Ptr{Cvoid},
                        proj::Ptr{Cvoid}, proj_data::Ptr{Cvoid},
                        obsv::Ptr{Cvoid}, obsv_data::Ptr{Cvoid},
                        m::Clong,
                        n::Clong,
                        x::Ptr{Cdouble},
                        epsopt::Cdouble,
                        maxit::Clong,
                        maxfc::Clong,
                        verb::Cint,
                        fx::Ptr{Cdouble},
                        gpsupn::Ptr{Cdouble},
                        iter::Ptr{Clong},
                        fcnt::Ptr{Clong},
                        gcnt::Ptr{Clong},
                        pcnt::Ptr{Clong},
                        status::Ptr{Cint},
                        inform::Ptr{Cint})::Cvoid
end

function _func(n::Clong, x_ptr::Ptr{Cdouble}, fx::Ptr{Cdouble}, inform::Ptr{Cint}, func_ptr::Ptr{Cvoid})
    func = unsafe_pointer_to_objref(func_ptr)[]
    x = unsafe_wrap(Array, x_ptr, n; own=false)
    unsafe_store!(fx, func(x))
    unsafe_store!(inform, 0)
    return nothing
end

function _grad(n::Clong, x_ptr::Ptr{Cdouble}, g_ptr::Ptr{Cdouble}, inform::Ptr{Cint}, grad_ptr::Ptr{Cvoid})
    grad! = unsafe_pointer_to_objref(grad_ptr)[]
    x = unsafe_wrap(Array, x_ptr, n; own=false)
    g = unsafe_wrap(Array, g_ptr, n; own=false)
    grad!(g, x)
    unsafe_store!(inform, 0)
    return nothing
end

function _proj(n::Clong, x_ptr::Ptr{Cdouble}, inform::Ptr{Cint}, proj_ptr::Ptr{Cvoid})
    #o = unsafe_pointer_to_objref(proj_ptr)
    #println(stderr, "proj: obj is ", typeof(o))
    proj! = unsafe_pointer_to_objref(proj_ptr)[]
    x = unsafe_wrap(Array, x_ptr, n; own=false)
    proj!(x)
    unsafe_store!(inform, 0)
    return nothing
end

struct Result{T<:AbstractFloat,Xa<:AbstractArray}
    x_best::Xa
    f_best::T
    gpsupn::T
    iter::Int
    fcnt::Int
    gcnt::Int
    pcnt::Int
    status::Cint
    inform::Cint
end

function cspg(func, grad!, proj!, x0::AbstractArray;
              m::Integer = 100,
              epsopt::Real = 1e-5, maxit::Integer = 1000, maxfc::Integer = 2000,
              verb::Integer = 0)
    fx = Ref{Cdouble}(NaN)
    gpsupn = Ref{Cdouble}(NaN)
    iter = Ref{Clong}(0)
    fcnt = Ref{Clong}(0)
    gcnt = Ref{Clong}(0)
    pcnt = Ref{Clong}(0)
    status = Ref{Cint}(0)
    inform = Ref{Cint}(0)
    x = Array{Cdouble}(undef, size(x0))
    copy!(x, x0)
    n = length(x)
    GC.@preserve func grad! proj! x begin
        _cspg(
            @cfunction(_func, Cvoid, (Clong, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid})),
            Ref(func),
            @cfunction(_grad, Cvoid, (Clong, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid})),
            Ref(grad!),
            @cfunction(_proj, Cvoid, (Clong, Ptr{Cdouble}, Ptr{Cint}, Ptr{Cvoid})),
            Ref(proj!),
            C_NULL, C_NULL,
            m, n, x, epsopt, maxit, maxfc, verb, fx, gpsupn, iter, fcnt, gcnt, pcnt,
            status, inform)
    end
    return Result(x, fx[], gpsupn[], iter[], fcnt[], gcnt[], pcnt[], status[], inform[])
end

function cspg(nlp::CUTEstModel; kwds...)
    return cspg(x -> obj(nlp, x),
                (g, x) -> copy!(g, grad(nlp, x)),
                Box(nlp.meta.lvar, nlp.meta.uvar), nlp.meta.x0; kwds...)
end

function cspg(name::AbstractString, args::AbstractString...; kwds...)
    nlp = CUTEstModel(name, args...)
    try
        return cspg(nlp; kwds...)
    finally
        finalize(nlp)
    end
end

function cspg(::Type{T}, name::AbstractString, args::AbstractString...; kwds...) where {T<:AbstractFloat}
    nlp = CUTEstModel{T}(name, args...)
    try
        return cspg(nlp; kwds...)
    finally
        finalize(nlp)
    end
end

(box::Box)(x::AbstractArray) = boxproj!(x, box.lower, box.upper)

function boxproj!(x::AbstractArray, l::AbstractArray, u::AbstractArray)
    axes(x) == axes(l) == axes(u) || throw(DimensionMismatch("arguments must have the same axes"))
    @inbounds @simd for i in eachindex(x, l, u)
        x[i] = clamp(x[i], l[i], u[i])
    end
    return x
end

function runtests1()
    println("Problem                 n     iter      fcnt      gcnt      time (s)    f(x)       ‖gp(x)‖ₒₒ")
    println("-------------------- ------ --------- --------- --------- ---------- ----------- -----------")
    for (name, n) in ("BDEXP" => 5000,
                      "EXPLIN" => 120,
                      "EXPLIN2" => 120,
                      "EXPQUAD" => 120,
                      "MCCORMCK" => 10000,
                      "PROBPENL" => 500,
                      "QRTQUAD" => 120,
                      "S368" => 100,
                      "HADAMALS" => 1024,
                      "CHEBYQAD" => 50,
                      "HS110" => 50,
                      "LINVERSE" => 1999,
                      "NONSCOMP" => 10000,
                      "DECONVB" => 61,
                      "QR3DLS" => 610,
                      "SCOND1LS" => 1002,
                      )
        t0 = time()
        r = CSPG.cspg(Cdouble, name, "-param", "N=$n"; verb=0, maxit=10_000_000, maxfc=15_000_000)
        t = time() - t0
        #length(x) == n || @warn "expecting $n variables, got $(length(x))"
        @printf("%-20s %6d %9d %9d %9d %10.3f %11.3e %11.3e\n", name, length(r.x_best),
                r.iter, r.fcnt, r.gcnt, t, r.f_best, r.gpsupn)
    end
end

function runtests(; m::Integer=100, maxit::Integer=2_000_000, maxfc::Integer=4maxit,
                  orig::Bool=false)
    if orig
        println("===============================================================================")
        println(" Problem       n    iter    fcnt         f         gpsupn      SC  Time (secs.)")
        println("===============================================================================")
    else
        println("# julia> CSPG.runtests(; m=$m, maxit=$maxit, maxfc=$maxfc)")
        println()
        println("Problem         n      iter      fcnt      gcnt   time (s)     f(x)      ‖gp(x)‖ₒₒ status")
        println("---------- ------ --------- --------- --------- ---------- ----------- ----------- ------")
    end
    for (name, (n,iter,fcnt,f,gpsupn,sc,tm)) in (
        # Problem          n     iter     fcnt         f           gpsupn  SC  Time (secs.)
        "BDEXP"    => ( 5000,      15,      16,   3.08289524e-04, 9.4e-07, 0,    0.11),
        "EXPLIN"   => ( 1200,     787,     803,  -7.19250129e+07, 4.5e-07, 0,    0.08),
        "EXPLIN2"  => ( 1200,     463,     480,  -7.19988337e+07, 4.6e-07, 0,    0.06),
        "EXPQUAD"  => ( 1200,    4140,    5055,  -3.68494055e+09, 4.3e-07, 0,    0.53),
        "MCCORMCK" => ( 5000,      17,      18,  -4.56658055e+03, 6.1e-07, 0,    0.16),
        "PROBPENL" => (  500,       2,       6,   3.99198393e-07, 1.0e-07, 0,    0.01),
        "QRTQUAD"  => ( 5000,   50000,  244663,  -2.62588259e+11, 1.9e+05, 1,   39.45),
        "S368"     => (    8,       8,      10,  -1.00000000e+00, 2.7e-07, 0,    0.00),
        "HADAMALS" => (  400,     462,     588,   7.16546110e+03, 8.5e-07, 0,    0.18),
        "CHEBYQAD" => (  100,   24195,   29254,   9.46974634e-03, 1.0e-06, 0,  121.36),
        "HS110"    => (   10,       7,       9,  -4.57784755e+01, 1.6e-07, 0,    0.00),
        "LINVERSE" => ( 1999,    7981,    9356,   6.81000000e+02, 1.0e-06, 0,    5.58),
        "NONSCOMP" => ( 5000,      46,      47,   1.11602424e-12, 3.8e-07, 0,    0.12),
        "QR3DLS"   => (  610,   50000,   72851,   8.07575338e-06, 7.6e-06, 1,   24.54),
        "SCOND1LS" => ( 5002,   50000,   62593,   7.95401974e+00, 1.5e-02, 1,   99.51),
        "DECONVB"  => (   61,    4440,    4817,   1.27974471e-08, 1.0e-06, 0,    0.22),
        "BIGGSB1"  => ( 5000,   19688,   23091,   1.60018657e-02, 1.0e-06, 0,    8.89),
        "BQPGABIM" => (   50,      37,      50,  -3.79034323e-05, 3.1e-07, 0,    0.00),
        "BQPGASIM" => (   50,      40,      53,  -5.51981402e-05, 9.5e-07, 0,    0.00),
        "BQPGAUSS" => ( 2003,   50000,   61749,  -3.62540671e-01, 3.9e-03, 1,   20.97),
        "CHENHARK" => ( 5000,   15213,   18220,  -1.99998251e+00, 9.8e-07, 0,    7.92),
        "CVXBQP1"  => (10000,       1,       2,   2.25022500e+06, 2.8e-17, 0,    0.16),
        "JNLBRNG1" => (10000,    1281,    1344,  -1.80573195e-01, 8.5e-07, 0,    4.82),
        "JNLBRNG2" => (10000,    1117,    1162,  -4.14865279e+00, 8.3e-07, 0,    4.14),
        "JNLBRNGA" => (10000,     958,    1018,  -2.71101666e-01, 8.3e-07, 0,    3.24),
        "JNLBRNGB" => (10000,   13304,   15689,  -6.30068672e+00, 8.9e-07, 0,   42.76),
        "NCVXBQP1" => (10000,       1,       2,  -1.98554385e+10, 2.8e-17, 0,    0.12),
        "NCVXBQP2" => (10000,      72,      73,  -1.33402261e+10, 9.1e-07, 0,    0.19),
        "NCVXBQP3" => (10000,     125,     126,  -6.55826637e+09, 1.4e-07, 0,    0.25),
        "NOBNDTOR" => ( 5476,     365,     386,  -4.49933170e-01, 9.4e-07, 0,    0.74),
        "OBSTCLAE" => (10000,     512,     523,   1.88646121e+00, 9.2e-07, 0,    1.86),
        "OBSTCLAL" => (10000,     311,     318,   1.88646121e+00, 6.4e-07, 0,    1.22),
        "OBSTCLBL" => (10000,     331,     337,   7.27215590e+00, 6.9e-07, 0,    1.27),
        "OBSTCLBM" => (10000,     239,     244,   7.27215591e+00, 6.8e-07, 0,    0.97),
        "OBSTCLBU" => (10000,     315,     320,   7.27215591e+00, 6.1e-07, 0,    1.24),
        "PENTDI"   => ( 5000,       1,       3,  -7.50000000e-01, 0.0e+00, 0,    0.06),
        "TORSION1" => ( 5476,     442,     462,  -4.30275769e-01, 9.2e-07, 0,    0.85),
        "TORSION2" => ( 5476,     415,     430,  -4.30275798e-01, 2.7e-07, 0,    0.80),
        "TORSION3" => ( 5476,     128,     139,  -1.21695608e+00, 7.3e-07, 0,    0.35),
        "TORSION4" => ( 5476,     151,     156,  -1.21695608e+00, 9.7e-07, 0,    0.37),
        "TORSION5" => ( 5476,      67,      74,  -2.86337797e+00, 1.1e-07, 0,    0.24),
        "TORSION6" => ( 5476,      59,      63,  -2.86337797e+00, 8.2e-07, 0,    0.22),
        "TORSIONA" => ( 5476,     480,     502,  -4.18296119e-01, 9.3e-07, 0,    1.06),
        "TORSIONB" => ( 5476,     378,     388,  -4.18296121e-01, 9.2e-07, 0,    0.87),
        "TORSIONC" => ( 5476,     153,     164,  -1.20420894e+00, 8.4e-07, 0,    0.45),
        "TORSIOND" => ( 5476,     137,     142,  -1.20420895e+00, 9.5e-07, 0,    0.44),
        "TORSIONE" => ( 5476,      54,      61,  -2.85024786e+00, 8.7e-07, 0,    0.28),
        "TORSIONF" => ( 5476,      60,      64,  -2.85024786e+00, 7.6e-07, 0,    0.26),
        "ODNAMUR"  => (11130,   50000,   58358,   9.26480840e+03, 4.0e+00, 1,   85.73))
        t0 = time()
        r = CSPG.cspg(Cdouble, name, "-param", "N=$n"; m=m, verb=0, maxit=maxit, maxfc=maxfc)
        t = time() - t0
        #length(x) == n || @warn "expecting $n variables, got $(length(x))"
        if orig
            str = @sprintf("%8s %7d %7d %7d %16.8E %7.1E %7d %7.2f", name, length(r.x_best),
                           r.iter, r.fcnt, r.f_best, r.gpsupn, r.status - 1, t)
            println(replace(str, r"([0-9])E([-+][0-9])" => s"\1D\2"))
        else
            @printf("%-10s %6d %9d %9d %9d %10.3f %11.3e %11.3e ", name, length(r.x_best),
                    r.iter, r.fcnt, r.gcnt, t, r.f_best, r.gpsupn)
            printstyled(lpad(r.status - 1, 6); color=(r.status == 1 ? :green : :red))
            println()
        end
    end
end

end # module
