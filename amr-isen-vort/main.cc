#include <stdio.h>
#include <vector>

#include "scidf.h"
#include "spade.h"

using real_t = double;
using flux_t = spade::fluid_state::flux_t<real_t>;
using prim_t = spade::fluid_state::prim_t<real_t>;
using cons_t = spade::fluid_state::cons_t<real_t>;

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    std::vector<std::string> args;
    for (auto i: range(0, argc)) args.push_back(std::string(argv[i]));
    std::string input_filename = "input.sdf";
    if (args.size() < 2)
    {
        if (group.isroot()) print("Warning: defaulting to", input_filename);
    }
    else
    {
        input_filename = args[1];
    }
    
    scidf::node_t input;
    scidf::clargs_t clargs(argc, argv);
    scidf::read(input_filename, input, clargs);
    
    //==========================================================================
    //Kuya, Y., & Kawai, S. (2020). 
    //A stable and non-dissipative kinetic energy and entropy preserving (KEEP)
    //scheme for non-conforming block boundaries on Cartesian grids.
    //Computers and Fluids, 200. https://doi.org/10.1016/j.compfluid.2020.104427
    //
    // Equations 50, 52
    //
    //==========================================================================
    const real_t targ_cfl         = scidf::required<real_t>      (input["Config"]["cfl"])       >> scidf::greater_than(0.0);
    const int nt_max              = scidf::required<int>         (input["Config"]["nt_max"])    >> scidf::greater_than(0);
    const int nt_skip             = scidf::required<int>         (input["Config"]["nt_skip"])   >> scidf::greater_than(0);
    const int checkpoint_skip     = scidf::required<int>         (input["Config"]["ck_skip"])   >> scidf::greater_than(0);
    const int nx                  = scidf::required<int>         (input["Config"]["nx_cell"])   >> (scidf::greater_than(4) && scidf::even);
    const int ny                  = scidf::required<int>         (input["Config"]["ny_cell"])   >> (scidf::greater_than(4) && scidf::even);
    const int nxb                 = scidf::required<int>         (input["Config"]["nx_blck"])   >> (scidf::greater_than(0));
    const int nyb                 = scidf::required<int>         (input["Config"]["ny_blck"])   >> (scidf::greater_than(0));
    const int nguard              = scidf::required<int>         (input["Config"]["nguard"])    >> scidf::greater_than(0);
    const real_t xmin             = scidf::required<real_t>      (input["Config"]["xmin"])      ;
    const real_t xmax             = scidf::required<real_t>      (input["Config"]["xmax"])      >> scidf::greater_than(xmin);
    const real_t ymin             = scidf::required<real_t>      (input["Config"]["ymin"])      ;
    const real_t ymax             = scidf::required<real_t>      (input["Config"]["ymax"])      >> scidf::greater_than(ymin);
    const bool do_output          = scidf::required<bool>        (input["Config"]["output"])    ;
    const std::string init_file   = scidf::required<std::string> (input["Config"]["init_file"]) >> (scidf::is_file || scidf::equals("none"));
    const bool   do_refine        = scidf::required<bool>        (input["Config"]["do_refine"]) ;
    const bool resid_exch         = scidf::required<bool>        (input["Config"]["resid_exch"]);
    const bool manual_exchange    = scidf::required<bool>        (input["Config"]["manual_exchange"]);
    const real_t u0               = scidf::required<real_t>      (input["Fluid"]["u0"])         ;
    const real_t deltau           = scidf::required<real_t>      (input["Fluid"]["deltau"])     >> scidf::greater_than(0.0);
    const real_t gamma            = scidf::required<real_t>      (input["Fluid"]["gamma"])      >> scidf::greater_than(0.0);
    const real_t b                = scidf::required<real_t>      (input["Fluid"]["b"])          >> scidf::greater_than(0.0);
    const real_t cp               = scidf::required<real_t>      (input["Fluid"]["cp"])         >> scidf::greater_than(0.0);
    const real_t theta_d          = scidf::required<real_t>      (input["Fluid"]["theta_d"])    ;
    
    
    spade::fluid_state::ideal_gas_t<real_t> air;
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
    spade::amr::amr_blocks_t blocks(num_blocks, bounds);
    using refine_t = typename decltype(blocks)::refine_type;
    spade::ctrs::array<bool, 2> periodic = true;
    refine_t ref0  = {true,  true};
    refine_t refx  = {true,  false};
    refine_t refy  = {false, true};
    
    const auto near = [](const auto x, const auto y) { return spade::utils::abs(y-x) < 1e-2; };
    
    auto c0 = blocks.select([&](const auto& node)
    {
        const auto bbx = blocks.get_block_box(node.tag);
        bool y0 = near(bbx.min(0),  0.0 + xc) || near(bbx.max(0), 0.0 + xc);
        bool y1 = near(bbx.min(1), -1.0 + yc) || near(bbx.max(1), 1.0 + yc);
        return y0 && !y1;
    });
    if (do_refine) blocks.refine(c0, periodic, refx, spade::amr::constraints::factor2);
    
    auto c1 = blocks.select([&](const auto& node)
    {
        const auto bbx = blocks.get_block_box(node.tag);
        bool y0 = near(bbx.min(1),  0.0 + yc) || near(bbx.max(1), 0.0 + yc);
        bool y1 = near(bbx.min(0), -1.0 + xc) || near(bbx.max(0), 1.0 + xc);
        return y0 && !y1;
    });
    if (do_refine) blocks.refine(c1, periodic, refy, spade::amr::constraints::factor2);
    
    spade::grid::cartesian_grid_t grid(cells_in_block, exchange_cells, blocks, coords, group);
    auto handle = spade::grid::create_exchange(grid, group, periodic);
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;

    spade::grid::grid_array prim (grid, fill1, spade::device::best);
    spade::grid::grid_array rhs  (grid, fill2, spade::device::best);
    
    
    
    const real_t sintheta = std::sin(theta_d*spade::consts::pi/180.0);
    const real_t costheta = std::cos(theta_d*spade::consts::pi/180.0);
    const real_t u_theta  = u0*costheta;
    const real_t v_theta  = u0*sintheta;

    auto ini = _sp_lambda (const spade::coords::point_t<real_t>& x)
    {
        prim_t output;
        const real_t r         = sqrt((x[0] - xc)*(x[0] - xc) + (x[1] - yc)*(x[1] - yc));
        const real_t upmax     = deltau*u0;
        const real_t expfac    = exp(0.5*(1.0-((r*r)/(b*b))));
        const real_t ur        = (1.0/b)*deltau*u0*r*expfac;
        const real_t rhor      = pow(1.0 - 0.5*(air.gamma-1.0)*deltau*u0*deltau*u0*expfac, 1.0/(air.gamma - 1.0));
        const real_t pr        = pow(rhor, air.gamma)/air.gamma;
        const real_t theta_loc = atan2(x[1], x[0]);
        output.p() = pr;
        output.T() = pr/(rhor*air.R);
        output.u() = u_theta - ur*sin(theta_loc);
        output.v() = v_theta + ur*cos(theta_loc);
        output.w() = 0.0;
        
        return output;
    };

    spade::algs::fill_array(prim, ini);    
    
    if (init_file != "none")
    {
        if (group.isroot()) print("reading...");
        spade::io::binary_read(init_file, prim);
        if (group.isroot()) print("Init done.");
        handle.exchange(prim);
    }
    
    // const auto s0 = spade::convective::cent_keep<2>(air);
    // spade::convective::rusanov_t       flx    (air);
    // spade::convective::weno_t          s1     (flx);
    // spade::state_sensor::ducros_t      ducr   (1.0e-3);
    // spade::convective::hybrid_scheme_t tscheme(s0, s1, ducr);
    
    const auto tscheme = spade::convective::cent_keep<2>(air);
    
    const auto get_sig = [&](const prim_t& q) { return std::sqrt(air.gamma*air.R*q.T()) + std::sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w()); };
    
    spade::reduce_ops::reduce_max<real_t> max_op;
    real_t time0 = 0.0;
    
    const real_t dx       = spade::utils::min(grid.get_dx(0, 0), grid.get_dx(1, 0), grid.get_dx(2, 0));
    // const real_t umax_ini = spade::algs::transform_reduce(prim, get_sig, max_op);
    const real_t umax_ini = 1.0;
    
    const real_t dt       = targ_cfl*dx/umax_ini;
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air, spade::grid::include_exchanges);
    
    auto calc_rhs = [&](auto& resid, const auto& sol, const auto& t)
    {
        resid = 0.0;
        spade::pde_algs::flux_div(sol, resid, tscheme);
        spade::io::output_vtk("output", "rhs", resid);
        std::cin.get();
        // if (resid_exch) handle.exchange(resid);
    };
    
    auto boundary_cond = [&](auto& sol, const auto& t)
    {
        // if (!resid_exch) handle.exchange(sol);
    };
    
    spade::time_integration::time_axis_t       axis(time0, dt);
    spade::time_integration::ssprk34_t         alg;
    spade::time_integration::integrator_data_t q(prim, rhs, alg);
    spade::time_integration::integrator_t      time_int(axis, alg, q, calc_rhs, boundary_cond, trans);
    
    spade::timing::mtimer_t tmr("advance");
    std::ofstream myfile("hist.dat");
    for (auto nt: range(0, nt_max+1))
    {
        // const real_t umax   = spade::algs::transform_reduce(time_int.solution(), get_sig, max_op);
        const real_t umax = 1.0;
    
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
            if (do_output) spade::io::output_vtk("output", filename, time_int.solution());
            if (group.isroot()) print("Done.");
        }
        if (nt%checkpoint_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "check"+nstr;
            filename = "checkpoint/"+filename+".bin";
            // if (do_output) spade::io::binary_write(filename, time_int.solution());
            if (group.isroot()) print("Done.");
        }
    
    	tmr.start("advance");
        time_int.advance();
        tmr.stop("advance");
        if (group.isroot()) print(tmr);
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