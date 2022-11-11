#include <chrono>
#include "spade.h"
#include "proto/hywall_interop.h"

#include "typedef.h"
#include "calc_u_bulk.h"
#include "calc_boundary_flux.h"

#include "PTL.h"

static inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void set_channel_slip(auto& prims)
{
    const auto& grid = prims.get_grid();
    for (auto lb: range(0, grid.get_num_local_blocks()))
    {
        const auto& lb_glob = grid.get_partition().get_global_block(lb);
        int idc = 0;
        for (int dir = 2; dir <= 3; ++dir)
        {
            const auto& idomain = grid.is_domain_boundary(lb_glob);
            if (idomain(dir/2, dir%2))
            {
                const auto lb_idx = spade::ctrs::expand_index(lb_glob, grid.get_num_blocks());
                const auto nvec_out = v3i(0,2*idc-1,0);
                const spade::grid::cell_t<int> j = idc*(grid.get_num_cells(1)-1);
                auto r1 = range(-grid.get_num_exchange(0), grid.get_num_cells(0) + grid.get_num_exchange(0));
                auto r2 = range(-grid.get_num_exchange(2), grid.get_num_cells(2) + grid.get_num_exchange(2));
                for (auto ii: r1*r2)
                {
                    for (int nnn = 0; nnn < 2; ++nnn)
                    {
                        spade::grid::cell_idx_t i_d(ii[0], j-(nnn+0)*nvec_out[1], ii[1], lb);
                        spade::grid::cell_idx_t i_g(ii[0], j+(nnn+1)*nvec_out[1], ii[1], lb);
                        prim_t q_d, q_g;
                        for (auto n: range(0,5)) q_d[n] = prims(n, i_d[0], i_d[1], i_d[2], i_d[3]);
                        const auto x_g = grid.get_comp_coords(i_g);
                        const auto x_d = grid.get_comp_coords(i_d);
                        const auto n_g = calc_normal_vector(grid.coord_sys(), x_g, i_g, 1);
                        const auto n_d = calc_normal_vector(grid.coord_sys(), x_d, i_d, 1);
                        q_g.p() =  q_d.p();
                        q_g.u() =  q_d.u();
                        q_g.v() = -q_d.v()*n_d[1]/n_g[1];
                        q_g.w() =  q_d.w();
                        q_g.T() =  q_d.T();
                        for (auto n: range(0,5)) prims(n, i_g[0], i_g[1], i_g[2], i_g[3]) = q_g[n];
                    }
                }
            }
            ++idc;
        }
    }
}

template <typename gas_t> struct cns_force_t
{
    using input_type = spade::fetch::cell_fetch_t
    <
        spade::fetch::cell_mono
        <
            spade::fetch::cell_info
            <
                spade::fetch::cell_state<prim_t>
            >
        >
    >;
    using output_type = spade::fluid_state::flux_t<real_t>;
    real_t source_term;
    const gas_t* gas;
    cns_force_t(const real_t& fin, const gas_t& gas_in) {source_term = fin; gas = &gas_in;}
    output_type operator() (const input_type& input) const
    {
        const auto& q = std::get<0>(input.cell_data.mono_data.elements).data;
        cons_t w;
        spade::fluid_state::convert_state(q, w, *gas);
        const real_t rho = w.rho();
        output_type output;
        output.continuity() = 0.0;
        output.energy()     = 0.0;
        output.x_momentum() = rho*source_term;
        output.y_momentum() = 0.0;
        output.z_momentum() = 0.0;
        return output;
    }
    
};

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    
    const std::size_t dim = 3;
    
    bool init_from_file = false;
    std::string init_filename = "";
    spade::cli_args::shortname_args_t args(argc, argv);
    if (args.has_arg("-init"))
    {
        init_filename = args["-init"];
        if (group.isroot()) print("Initialize from", init_filename);
        init_from_file = true;
        if (!std::filesystem::exists(init_filename))
        {
            if (group.isroot()) print("Cannot find ini file", init_filename);
            abort();
        }
    }
    std::string input_filename = "none";
    for (auto i: range(0, argc))
    {
        std::string line(argv[i]);
        if (ends_with(line, ".ptl"))
        {
            input_filename = line;
            if (group.isroot()) print("Reading", input_filename);
        }
    }
    if (input_filename == "none")
    {
        if (group.isroot()) print("E: No input file name provided!");
        return 1;
    }
    
    //notes for subsonic channel
    //Re_bulk = 125,000
    //rho = 1.176 (T = 300.0, P = 101325)
    //M = 0.2
    //U_infty = 69.4377
    //tau_wall = 9.815
    //u_tau = 2.889
	// Lx, Ly, Lz = 4pi, 2delta, 4pi/3, delta = 1
	// dx, dy, dz = 0.6666666666666643E-01, 0.1562500015624990E-01, 0.5999999999999961E-01
    PTL::PropertyTree input;
    input.Read(input_filename);
    
    std::vector<int>    nblk     = input["Grid"]["num_blocks"];
    std::vector<int>    ncell    = input["Grid"]["num_cells"];
    std::vector<int>    nexg     = input["Grid"]["num_exchg"];
    std::vector<real_t> bbox     = input["Grid"]["dims"];
    
    real_t              targ_cfl = input["Time"]["cfl"];
    int                 nt_max   = input["Time"]["nt_max"];
    int                 nt_skip  = input["Time"]["nt_skip"];
    int         checkpoint_skip  = input["Time"]["ck_skip"];
    bool          output_timing  = input["Time"]["output_timing"];
    
    real_t                 Twall = input["Fluid"]["Twall"];
    real_t                  Tref = input["Fluid"]["Tref"];
    real_t                mu_ref = input["Fluid"]["mu_ref"];
    real_t                    p0 = input["Fluid"]["p0"];
    real_t                    u0 = input["Fluid"]["u0"];
    real_t               prandtl = input["Fluid"]["prandtl"];
    real_t            wall_shear = input["Fluid"]["wall_shear"];
    real_t                 rho_b = input["Fluid"]["rho_b"];
    
    real_t                eps_p  = input["Num"]["eps_p"];
    real_t                eps_T  = input["Num"]["eps_T"];
    bool              wm_enable  = input["Num"]["wm_enable"];
    
    spade::coords::identity<real_t> coords;
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    spade::ctrs::array<int, dim> num_blocks;
    spade::ctrs::array<int, dim> cells_in_block;
    spade::ctrs::array<int, dim> exchange_cells;
    spade::bound_box_t<real_t, dim> bounds;
    
    for (auto i: range(0, dim))
    {
        num_blocks[i]     = nblk[i];
        cells_in_block[i] = ncell[i];
        exchange_cells[i] = nexg[i];
        bounds.min(i)     = bbox[2*i];
        bounds.max(i)     = bbox[2*i + 1];
    }
    
    spade::grid::cartesian_grid_t grid(num_blocks, cells_in_block, exchange_cells, bounds, coords, group);
    
    real_t delta = 0.5*(bounds.size(1));
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::grid_array prim (grid, fill1);
    spade::grid::grid_array rhs (grid, fill2);
    
    spade::viscous_laws::constant_viscosity_t<real_t> visc_law(mu_ref, prandtl);
    
    spade::fluid_state::perfect_gas_t<real_t> air;
    air.R = 287.15;
    air.gamma = 1.4;
    
    const real_t Lx = bounds.size(0);
    const real_t Lz = bounds.size(2);
    auto ini = [&](const spade::ctrs::array<real_t, 3>& x) -> prim_t
    {
        const real_t yh = x[1]/delta;
        prim_t output;
        output.p() = p0;
        output.T() = Tref - (Tref - Twall)*yh*yh;
        output.u() = u0*(1.0-yh*yh);
        output.v() = output.u()*0.1*std::sin(20.0*spade::consts::pi*x[0]/Lx);
        output.w() = output.u()*0.1*std::sin(20.0*spade::consts::pi*x[2]/Lz);
        
        return output;
    };
    
    spade::algs::fill_array(prim, ini);
    
    if (init_from_file)
    {
        if (group.isroot()) print("reading...");
        spade::io::binary_read(init_filename, prim);
        if (group.isroot()) print("Init done.");
        grid.exchange_array(prim);
        set_channel_slip(prim);
    }
    
    spade::convective::totani_lr        tscheme(air);
    spade::convective::pressure_diss_lr dscheme(air, eps_p, eps_T);
    spade::viscous::visc_lr  visc_scheme(visc_law);
    
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
    real_t time0 = 0.0;
    
    
    
    const real_t dx = spade::utils::min(grid.get_dx(0), grid.get_dx(1), grid.get_dx(2));
    const real_t umax_ini = spade::algs::transform_reduce(prim, get_u, max_op);
    const real_t dt     = targ_cfl*dx/umax_ini;
    
    
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air);
    
    const real_t force_term = wall_shear/(delta*rho_b);
    cns_force_t source(force_term, air);
    
    spade::bound_box_t<bool, grid.dim()> boundary = true;
    boundary.min(1) = false;
    boundary.max(1) = false;
    
    spade::proto::hywall_binding_t wall_model(prim, rhs, air);
    wall_model.read(input["WallModel"]);
    if (wm_enable)
    {
        for (auto& b: boundary) b = !b;
        wall_model.init(prim, boundary);
        for (auto& b: boundary) b = !b;
        wall_model.set_dt(dt);
    }
    spade::utils::mtimer_t tmr("exchg", "bc", "fdiv", "wm", "source", "advance");
    auto calc_rhs = [&](auto& rhs, auto& q, const auto& t) -> void
    {
        rhs = 0.0;
        tmr.start("exchg");
        grid.exchange_array(q);
        tmr.stop("exchg");
        
        tmr.start("bc");
        set_channel_slip(q);
        tmr.stop("bc");
        
        tmr.start("fdiv");
        spade::pde_algs::flux_div(q, rhs, tscheme, dscheme);
        if (!wm_enable)
        {
            spade::pde_algs::flux_div(q, rhs, visc_scheme);
        }
        else
        {
            auto policy = spade::pde_algs::block_flux_all;
            spade::pde_algs::flux_div(q, rhs, policy, boundary, visc_scheme);
        }
        tmr.stop("fdiv");
        
        tmr.start("wm");
        if (wm_enable)
        {
            wall_model.sample(q, visc_law);
            wall_model.solve();
            wall_model.apply_flux(rhs);
        }
        tmr.stop("wm");
        
        tmr.start("source");
        spade::pde_algs::source_term(q, rhs, source);
        tmr.stop("source");
        
        
    };
    
    spade::time_integration::rk2 time_int(prim, rhs, time0, dt, calc_rhs, trans);
    
    std::ofstream myfile("hist.dat");
    for (auto nt: range(0, nt_max+1))
    {
        const real_t umax   = spade::algs::transform_reduce(prim, get_u, max_op);
        real_t ub, rhob;
        calc_u_bulk(prim, air, ub, rhob);
        
        if (group.isroot())
        {
            const real_t cfl = umax*dt/dx;
            const int pn = 10;
            print(
                "nt: ", spade::utils::pad_str(nt,   pn),
                "cfl:", spade::utils::pad_str(cfl,  pn),
                "u+a:", spade::utils::pad_str(umax, pn),
                "ub: ", spade::utils::pad_str(ub,   pn),
                "rb: ", spade::utils::pad_str(rhob, pn),
                "dx: ", spade::utils::pad_str(dx,   pn),
                "dt: ", spade::utils::pad_str(dt,   pn)
            );
            myfile << nt << " " << cfl << " " << umax << " " << ub << " " << rhob << " " << dx << " " << dt << std::endl;
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
        tmr.start("advance");
        time_int.advance();
        tmr.stop("advance");
        if (group.isroot() && output_timing) print(tmr);
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
