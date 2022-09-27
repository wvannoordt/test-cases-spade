#include <chrono>
#include "spade.h"
#include "c2p.h"
#include "preconditioner.h"
#include "PTL.h"

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    std::vector<std::string> args;
    for (auto i: range(0, argc)) args.push_back(std::string(argv[i]));
    if (args.size() < 2)
    {
        if (group.isroot()) print("Please provide an input file name!");
        return 1;
    }
    std::string input_filename = args[1];
    PTL::PropertyTree input;
    input.Read(input_filename);
    
    //============================================================+
    //Cadieux, F., Barad, M., & Kiris, C. (2018). 
    //A High-Order Kinetic Energy Conserving Scheme for
    //Compressible Large-Eddy Simulation.
    //https://doi.org/10.13140/RG.2.2.12165.86241</div>
    //
    //Secion 4.3
    //============================================================+
    
    const real_t targ_cfl         = input["Config"]["cfl"];
    const real_t inner_cfl        = input["Config"]["icfl"];
    const real_t error_tol        = input["Config"]["tol"];
    const real_t beta             = input["Config"]["beta"];
    const int    nt_max           = input["Config"]["nt_max"];
    const int    nt_skip          = input["Config"]["nt_skip"];
    const int    checkpoint_skip  = input["Config"]["ck_skip"];
    const int    nx               = input["Config"]["nx_cell"];
    const int    ny               = input["Config"]["ny_cell"];
    const int    nz               = input["Config"]["nz_cell"];
    const int    nxb              = input["Config"]["nx_blck"];
    const int    nyb              = input["Config"]["ny_blck"];
    const int    nzb              = input["Config"]["nz_blck"];
    const int    nguard           = input["Config"]["nguard"];
    const real_t xmin             = input["Config"]["xmin"];
    const real_t xmax             = input["Config"]["xmax"];
    const real_t ymin             = input["Config"]["ymin"];
    const real_t ymax             = input["Config"]["ymax"];
    const real_t zmin             = input["Config"]["zmin"];
    const real_t zmax             = input["Config"]["zmax"];
    const std::string init_file   = input["Config"]["init_file"];
    const real_t mach             = input["Fluid"]["mach"];
    const real_t reynolds         = input["Fluid"]["reynolds"];
    const real_t T_ref            = input["Fluid"]["T_ref"];
    const real_t gamma            = input["Fluid"]["gamma"];
    const real_t cp               = input["Fluid"]["cp"];
    const real_t diss_coeff       = input["Fluid"]["diss_coeff"];
    
    spade::fluid_state::perfect_gas_t<real_t> air;
    air.gamma = gamma;
    air.R = (1.0-(1.0/gamma))*cp;
    
    const real_t delta = 1.0;
    const real_t aref  = std::sqrt(gamma*air.R*T_ref);
    const real_t u0    = aref*mach;
    const real_t p0    = 30.0;
    const real_t rho0  = 1.0;
    
    const real_t mu = rho0*u0*delta/reynolds;
    
    spade::viscous_laws::constant_viscosity_t<real_t> visc_law(mu);
    visc_law.prandtl = 0.72;
    
    
    spade::ctrs::array<int, 3> num_blocks(nxb, nyb, nzb);
    spade::ctrs::array<int, 3> cells_in_block(nx, ny, nz);
    spade::ctrs::array<int, 3> exchange_cells(nguard, nguard, nguard);
    spade::bound_box_t<real_t, 3> bounds;
    bounds.min(0) =  xmin;
    bounds.max(0) =  xmax;
    bounds.min(1) =  ymin;
    bounds.max(1) =  ymax;
    bounds.min(2) =  zmin;
    bounds.max(2) =  zmax;
    
    spade::coords::identity<real_t> coords;
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    
    spade::grid::cartesian_grid_t grid(num_blocks, cells_in_block, exchange_cells, bounds, coords, group);
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::grid_array prim (grid, fill1);
    spade::grid::grid_array rhs  (grid, fill2);
    
    auto ini = [&](const spade::ctrs::array<real_t, 3> x) -> prim_t
    {
        prim_t output;
        output.p() =  p0 + (1.0/16.0)*rho0*u0*u0*(std::cos(2.0*x[0]/delta)+std::cos(2.0*x[1]/delta))*(std::cos(2.0*x[2]/delta) + 2.0);
        output.T() =  T_ref;
        output.u() =  u0*std::sin(x[0]/delta)*std::cos(x[1]/delta)*std::cos(x[2]/delta);
        output.v() = -u0*std::cos(x[0]/delta)*std::sin(x[1]/delta)*std::cos(x[2]/delta);
        output.w() =  0.0;
        return output;
    };
    
    spade::algs::fill_array(prim, ini);
    grid.exchange_array(prim);
    
    if (init_file != "none")
    {
        if (group.isroot()) print("reading...");
        spade::io::binary_read(init_file, prim);
        if (group.isroot()) print("Init done.");
        grid.exchange_array(prim);
    }
    
    struct p2c_t
    {
        const spade::fluid_state::perfect_gas_t<real_t>* gas;
        typedef prim_t arg_type;
        p2c_t(const spade::fluid_state::perfect_gas_t<real_t>& gas_in) {gas = &gas_in;}
        cons_t operator () (const prim_t& q) const
        {
            cons_t w;
            spade::fluid_state::convert_state(q, w, *gas);
            return w;
        }
    } p2c(air);
    
    struct c2p_t
    {
        const spade::fluid_state::perfect_gas_t<real_t>* gas;
        typedef cons_t arg_type;
        c2p_t(const spade::fluid_state::perfect_gas_t<real_t>& gas_in) {gas = &gas_in;}
        prim_t operator () (const cons_t& w) const
        {
            prim_t q;
            spade::fluid_state::convert_state(w, q, *gas);
            return q;
        }
    } c2p(air);
    
    struct get_u_t
    {
        const spade::fluid_state::perfect_gas_t<real_t>* gas;
        typedef prim_t arg_type;
        get_u_t(const spade::fluid_state::perfect_gas_t<real_t>& gas_in) {gas = &gas_in;}
        real_t operator () (const prim_t& q) const
        {
            return sqrt(gas->gamma*gas->R*q.T()) + sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w());
        }
    } get_u(air);
    
    spade::reduce_ops::reduce_max<real_t> max_op;
    spade::reduce_ops::reduce_max<real_t> sum_op;
    real_t time0 = 0.0;
    
    
    
    const real_t dx = spade::utils::min(grid.get_dx(0), grid.get_dx(1), grid.get_dx(2));
    const real_t umax_ini = spade::algs::transform_reduce(prim, get_u, max_op);
    const real_t dt     = targ_cfl*dx/umax_ini;

    spade::convective::pressure_diss_lr diss_scheme(air, diss_coeff);
    spade::convective::totani_lr        cent_scheme(air);
    spade::viscous::visc_lr             visc_scheme(visc_law);
    
    trans_t trans(air, prim);
    auto calc_rhs = [&](auto& rhs, auto& q, const auto& t) -> void
    {
        rhs = 0.0;
        grid.exchange_array(q);
	spade::pde_algs::flux_div(q, rhs, cent_scheme, visc_scheme, diss_scheme);
	//spade::pde_algs::flux_div(q, rhs, cent_scheme, visc_scheme);
    };
    
    
    int max_its = 5000000;
    spade::static_math::int_const_t<2> bdf_order;
    const int ndof = grid.grid_size();
    auto error_norm = [&](const auto& r) -> real_t
    {
        auto output = spade::algs::transform_reduce(
            r, 
            [](const flux_t& f) -> real_t 
            {
                return f[0]*f[0] + f[1]*f[1] + f[2]*f[2] + f[3]*f[3] + f[4]*f[4];
            },
            max_op);
        if (group.isroot()) print("Residual:", output);
        return output;
    };
    
    preconditioner_t preconditioner(air, prim, beta);
    
    spade::algs::iterative_control convergence_crit(rhs, error_norm, error_tol, max_its);
    // spade::time_integration::dual_time_t time_int(prim, rhs, time0, dt, dt*(inner_cfl/targ_cfl), calc_rhs, convergence_crit, bdf_order, trans, preconditioner);
    
    spade::time_integration::rk2 time_int(prim, rhs, time0, dt, calc_rhs, trans);
    
    std::ofstream myfile("hist.dat");
    real_t running_t_avg = 0.0;
    int num_avg_iter = 0;
    for (auto nt: range(0, nt_max+1))
    {
        const real_t umax   = spade::algs::transform_reduce(prim, get_u, max_op);        
        if (group.isroot())
        {
            const real_t cfl = umax*dt/dx;
            const int pn = 10;
            print(
                "nt: ",  spade::utils::pad_str(nt, pn),
                "cfl:",  spade::utils::pad_str(cfl, pn),
                "umax:", spade::utils::pad_str(umax, pn),
                "dx: ",  spade::utils::pad_str(dx, pn),
                "dt: ",  spade::utils::pad_str(dt, pn)
            );
            myfile << nt << " " << cfl << " " << umax << " " << dx << " " << dt << std::endl;
            myfile.flush();
        }
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solution...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "prims"+nstr;
            spade::io::output_vtk("output", filename, prim);
            if (group.isroot()) print("Done.");
        }
        if (nt%checkpoint_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "check"+nstr;
            filename = "checkpoint/"+filename+".bin";
            spade::io::binary_write(filename, prim);
            if (group.isroot()) print("Done.");
        }
    	auto start = std::chrono::steady_clock::now();
        time_int.advance();
    	auto end = std::chrono::steady_clock::now();
        real_t alpha = ((real_t)num_avg_iter)/(num_avg_iter+1);
        real_t beta  = 1.0 - alpha;
        real_t millis = (real_t)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        running_t_avg = beta*millis+alpha*running_t_avg;
        num_avg_iter++;
        // print(alpha, beta);
    	if (group.isroot()) print("Elapsed:", millis, "ms, avg.", running_t_avg, "ms");
        if (std::isnan(umax))
        {
            if (group.isroot())
            {
                print("A tragedy has occurred!");
            }
            group.sync();
            return 155;
        }
    }
    
    
    return 0;
}
