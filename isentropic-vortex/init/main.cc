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
    
    //==========================================================================
    //Kuya, Y., & Kawai, S. (2020). 
    //A stable and non-dissipative kinetic energy and entropy preserving (KEEP)
    //scheme for non-conforming block boundaries on Cartesian grids.
    //Computers and Fluids, 200. https://doi.org/10.1016/j.compfluid.2020.104427
    //
    // Equations 50, 52
    //
    //==========================================================================
    const real_t targ_cfl         = input["Config"]["cfl"];
    const real_t inner_cfl        = input["Config"]["icfl"];
    const real_t error_tol        = input["Config"]["tol"];
    const real_t beta             = input["Config"]["beta"];
    const int    nt_max           = input["Config"]["nt_max"];
    const int    nt_skip          = input["Config"]["nt_skip"];
    const int    checkpoint_skip  = input["Config"]["ck_skip"];
    const int    nx               = input["Config"]["nx_cell"];
    const int    ny               = input["Config"]["ny_cell"];
    const int    nxb              = input["Config"]["nx_blck"];
    const int    nyb              = input["Config"]["ny_blck"];
    const int    nguard           = input["Config"]["nguard"];
    const real_t xmin             = input["Config"]["xmin"];
    const real_t xmax             = input["Config"]["xmax"];
    const real_t ymin             = input["Config"]["ymin"];
    const real_t ymax             = input["Config"]["ymax"];
    const std::string init_file   = input["Config"]["init_file"];
    const real_t u0               = input["Fluid"]["u0"];
    const real_t deltau           = input["Fluid"]["deltau"];
    const real_t gamma            = input["Fluid"]["gamma"];
    const real_t b                = input["Fluid"]["b"];
    const real_t cp               = input["Fluid"]["cp"];
    const real_t theta_d          = input["Fluid"]["theta_d"];
    
    spade::fluid_state::perfect_gas_t<real_t> air;
    air.gamma = gamma;
    air.R = (1.0-(1.0/gamma))*cp;
    
    const real_t xc     = 0.5*(xmin+xmax);
    const real_t yc     = 0.5*(ymin+ymax);
    
    spade::ctrs::array<int, 2> num_blocks(nxb, nyb);
    spade::ctrs::array<int, 2> cells_in_block(nx, ny);
    spade::ctrs::array<int, 2> exchange_cells(nguard, nguard);
    spade::bound_box_t<real_t, 2> bounds;
    bounds.min(0) =  xmin;
    bounds.max(0) =  xmax;
    bounds.min(1) =  ymin;
    bounds.max(1) =  ymax;
    
    spade::coords::identity<real_t> coords;
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    
    spade::grid::cartesian_grid_t grid(num_blocks, cells_in_block, exchange_cells, bounds, coords, group);
    
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::grid_array prim (grid, fill1);
    spade::grid::grid_array rhs (grid, fill2);
    
    
    const real_t sintheta = std::sin(theta_d*spade::consts::pi/180.0);
    const real_t costheta = std::cos(theta_d*spade::consts::pi/180.0);
    const real_t u_theta  = u0*costheta;
    const real_t v_theta  = u0*sintheta;
    
    auto ini = [&](const spade::ctrs::array<real_t, 3> x) -> prim_t
    {
        prim_t output;
        const real_t r      = std::sqrt((x[0]-xc)*(x[0]-xc) + (x[1]-yc)*(x[1]-yc));
        const real_t upmax  = deltau*u0;
        const real_t expfac = std::exp(0.5*(1.0-((r*r)/(b*b))));
        const real_t ur     = (1.0/b)*deltau*u0*r*expfac;
        const real_t rhor   = std::pow(1.0 - 0.5*(air.gamma-1.0)*deltau*u0*deltau*u0*expfac, 1.0/(air.gamma - 1.0));
        const real_t pr     = std::pow(rhor, air.gamma)/air.gamma;
        const real_t theta_loc = std::atan2(x[1], x[0]);
        output.p() = pr;
        output.T() = pr/(rhor*air.R);
        output.u() = u_theta - ur*std::sin(theta_loc);
        output.v() = v_theta + ur*std::cos(theta_loc);
        output.w() = 0.0;
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
    
    spade::convective::totani_lr        tscheme(air);
    spade::convective::pressure_diss_lr dscheme(air, 0.1);
    
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
    
    trans_t trans(air, prim);
    auto calc_rhs = [&](auto& rhs, auto& q, const auto& t) -> void
    {
        rhs = 0.0;
        grid.exchange_array(q);
        spade::pde_algs::flux_div(q, rhs, tscheme);
        // spade::pde_algs::flux_div(q, rhs, dscheme);
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
    spade::time_integration::dual_time_t time_int(prim, rhs, time0, dt, dt*(inner_cfl/targ_cfl), calc_rhs, convergence_crit, bdf_order, trans, preconditioner);
    
    // spade::time_integration::rk2 time_int(prim, rhs, time0, dt, calc_rhs, trans);
    
    std::ofstream myfile("hist.dat");
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
            spade::io::output_vtk("output", filename, grid, prim);
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
    	if (group.isroot()) print("Elapsed:", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(), "ms");
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
