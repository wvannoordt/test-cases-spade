#include <stdio.h>
#include <vector>

#include "scidf.h"
#include "spade.h"

#include "sample_cloud.h"
#include "fill_ghosts.h"

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
    
    spade::fluid_state::ideal_gas_t<real_t> air;
    air.gamma = input["Fluid"]["gamma"];
    air.R     = input["Fluid"]["Rgas"];
    
    const real_t mach = input["Fluid"]["mach"];
    const real_t aoa  = input["Fluid"]["aoa"];
    
    
    const std::string init_file = input["Config"]["init_file"];
    const real_t      targ_cfl  = input["Config"]["cfl"];
    const int         nt_max    = input["Config"]["nt_max"];
    const int         nt_skip   = input["Config"]["nt_skip"];
    
    spade::ctrs::array<int,    3> num_blocks     = input["Grid"]["nblck"];
    spade::ctrs::array<int,    3> num_cells      = input["Grid"]["ncell"];
    spade::ctrs::array<int,    3> exchange_cells = input["Grid"]["nexch"];
    spade::ctrs::array<real_t, 6> bnd            = input["Grid"]["bounds"];
    spade::bound_box_t<real_t, 3> bounds;
    bounds.bnds = bnd;
    const std::string geom_fname = input["Grid"]["geom"];
    const int         maxlevel   = input["Grid"]["maxlevel"];
    const real_t      sampl_dist = input["Grid"]["sampl_dist"];

    spade::coords::identity<real_t> coords;
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    spade::ctrs::array<bool, 3> periodic = true;
    spade::amr::amr_blocks_t blocks(num_blocks, bounds);
    
    if (group.isroot()) print("Create grid");
    spade::grid::cartesian_grid_t grid(num_cells, exchange_cells, blocks, coords, group);
    if (group.isroot()) print("Done");
    
    if (group.isroot()) print("Read", geom_fname);
    spade::geom::vtk_geom_t<3> geom;
    spade::geom::read_vtk_geom(geom_fname, geom);
    if (group.isroot()) print("Done");
    
    using refine_t = typename decltype(blocks)::refine_type;
    refine_t ref0  = {true, true, true};
    
    if (group.isroot()) print("Begin refine");
    int iter = 0;
    while (true)
    {
        if (group.isroot()) print(" -- iteration", iter++);
        if (group.isroot()) print(" ---- points:", grid.get_grid_size());
        const auto bndy_intersect = [&](const auto& lb)
        {
            if (grid.get_blocks().get_amr_node(lb).level[0] >= maxlevel) return false;
            const auto bnd = grid.get_bounding_box(lb);
            return geom.box_contains_boundary(bnd);
        };
        auto rblks = grid.select_blocks(bndy_intersect, spade::partition::global);
        if (rblks.size() == 0) break;
        grid.refine_blocks(rblks);
    }
    if (group.isroot()) print("Done");
    if (group.isroot()) print("Num. points:", grid.get_grid_size());
    
    if (group.isroot()) print("Compute ghosts");
    const auto ghosts = spade::ibm::compute_ghosts(grid, geom);
    if (group.isroot()) print("Done");
    
    if (group.isroot()) print("Compute ips");
    const real_t dx       = spade::utils::min(grid.get_dx(0, 0), grid.get_dx(1, 0), grid.get_dx(2, 0));
    auto ips = local::compute_ghost_sample_points(ghosts, grid, sampl_dist*dx);
    spade::io::output_vtk("debug/ips.vtk", ips);
    if (group.isroot()) print("Done");
    
    auto handle = spade::grid::create_exchange(grid, group, periodic);
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;

    spade::grid::grid_array prim (grid, fill1, spade::device::best);
    spade::grid::grid_array rhs  (grid, fill2, spade::device::best);
    
    auto interp = spade::grid::create_interpolation(prim, ips);

    const real_t Tinf  = 75.0;
    const real_t Pinf  = 5000.0;
    const real_t Uinf  = mach*sqrt(air.gamma*air.R*Tinf);
    const real_t theta = spade::consts::pi*aoa/180.0;
    
    const real_t uu = Uinf*cos(theta);
    const real_t vv = Uinf*sin(theta);
    const real_t ww = 0.0;
    
    auto ini = _sp_lambda (const spade::coords::point_t<real_t>& x)
    {
        // real_t dfac = (sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) - 0.08)/0.08;
        // if (dfac < 0.001) dfac = 0.0;
        // if (dfac > 1.0) dfac = 1.0;
        real_t dfac = 1.0;
        prim_t output;
        output.p() = Pinf;
        output.T() = Tinf;
        output.u() = dfac*uu;
        output.v() = dfac*vv;
        output.w() = dfac*ww;
        return output;
    };

    spade::algs::fill_array(prim, ini);
    
    auto sampldata = spade::grid::sample_array(prim, interp);
    
    
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
    
    // const auto tscheme = spade::convective::cent_keep<2>(air);
    
    spade::convective::first_order_t tscheme(air);
    
    const auto get_sig = [&](const prim_t& q) { return std::sqrt(air.gamma*air.R*q.T()) + std::sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w()); };
    
    spade::reduce_ops::reduce_max<real_t> max_op;
    real_t time0 = 0.0;
    
    const real_t umax_ini = Uinf + sqrt(air.R*air.gamma*Tinf);
    const real_t dt       = targ_cfl*dx/umax_ini;
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air, spade::grid::include_exchanges);
    
    
    auto zr = _sp_lambda (const spade::coords::point_t<real_t>& x, const flux_t& f)
    {
        real_t dfac = (sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]))/0.1;
        if (dfac < -0.001)
        {
            flux_t output = 0.0;
            return output;
        }
        return f;
    };
    
    auto calc_rhs = [&](auto& resid, const auto& sol, const auto& t)
    {
        resid = 0.0;
        spade::pde_algs::flux_div(sol, resid, tscheme);
        local::zero_ghost_rhs(resid, ghosts);
        spade::algs::fill_array(resid, zr);
    };
    
    auto boundary_cond = [&](auto& sol, const auto& t)
    {
        spade::grid::sample_array(sampldata, sol, interp);
        local::fill_ghost_vals(sol, ghosts, ips, sampldata);
        handle.exchange(sol);
    };
    
    boundary_cond(prim, time0);
    
    spade::time_integration::time_axis_t       axis(time0, dt);
    spade::time_integration::rk2_t             alg;
    spade::time_integration::integrator_data_t q(prim, rhs, alg);
    spade::time_integration::integrator_t      time_int(axis, alg, q, calc_rhs, boundary_cond, trans);
    
    spade::timing::mtimer_t tmr("advance");
    for (auto nt: range(0, nt_max+1))
    {
        const real_t umax = umax_ini;
    
        if (group.isroot())
        {
            const real_t cfl = umax*dt/dx;
            const int pn = 10;
            print("nt: ",  spade::utils::pad_str(nt, pn));
        }
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solution...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "prims"+nstr;
            spade::io::output_vtk("output", filename, time_int.solution());
            if (group.isroot()) print("Done.");
        }
    
    	tmr.start("advance");
        time_int.advance();
        tmr.stop("advance");
        if (group.isroot()) print(tmr);
    }
    return 0;
}