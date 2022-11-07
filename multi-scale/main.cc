#include <chrono>
#include <typeinfo>
#include "spade.h"

typedef double real_t;
typedef spade::ctrs::array<real_t, 3> v3d;
typedef spade::ctrs::array<real_t, 5> v5d;
typedef spade::ctrs::array<int,    3> v3i;
typedef spade::ctrs::array<int,    4> v4i;
typedef spade::ctrs::array<spade::grid::cell_t<int>, 4> v4c;
typedef spade::fluid_state::prim_t<real_t> prim_t;
typedef spade::fluid_state::flux_t<real_t> flux_t;
typedef spade::fluid_state::cons_t<real_t> cons_t;

#include "calc_u_bulk.h"
#include "calc_boundary_flux.h"

void set_channel_noslip(auto& prims)
{
    const real_t t_wall = 0.1;
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
                        q_g.u() = -q_d.u();
                        q_g.v() = -q_d.v()*n_d[1]/n_g[1];
                        q_g.w() = -q_d.w();
                        q_g.T() =  2.0*t_wall - q_d.T();
                        for (auto n: range(0,5)) prims(n, i_g[0], i_g[1], i_g[2], i_g[3]) = q_g[n];
                    }
                }
            }
            ++idc;
        }
    }
}

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    
    spade::bound_box_t<real_t, 3> bounds;
    const real_t re_tau = 180.0;
    const real_t delta = 1.0;
    bounds.min(0) =  0.0;
    bounds.max(0) =  4.0*spade::consts::pi*delta;
    bounds.min(1) = -delta;
    bounds.max(1) =  delta;
    bounds.min(2) =  0.0;
    bounds.max(2) =  2*spade::consts::pi*delta;
    
    const real_t targ_cfl = 0.25;
    const int    nt_max   = 300001;
    const int    nt_skip  = 5000;
    const int    checkpoint_skip  = 2500;
    
    spade::coords::identity<real_t> coords_les;
    spade::coords::identity_1D<real_t> xc;
    spade::coords::integrated_tanh_1D<real_t> yc(bounds.min(1), bounds.max(1), 0.1, 1.3);
    spade::coords::identity_1D<real_t> zc;
    spade::coords::diagonal_coords coords_dns(xc, yc, zc);
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    
    spade::ctrs::array<int, 3> num_blocks_les(4, 4, 4);
    spade::ctrs::array<int, 3> cells_in_block_les(16, 16, 16);
    spade::ctrs::array<int, 3> exchange_cells_les(2, 2, 2);
    spade::grid::cartesian_grid_t grid_les(num_blocks_les, cells_in_block_les, exchange_cells_les, bounds, coords_les, group);
    
    
    spade::ctrs::array<int, 3> num_blocks_dns(4, 4, 4);
    spade::ctrs::array<int, 3> cells_in_block_dns(16, 16, 16);
    spade::ctrs::array<int, 3> exchange_cells_dns(2, 2, 2);
    spade::grid::cartesian_grid_t grid_dns(num_blocks_dns, cells_in_block_dns, exchange_cells_dns, bounds, coords_dns, group);
    
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::grid_array prim_les   (grid_les, fill1);
    spade::grid::face_array fluxes_les (grid_les, fill2);
    spade::grid::grid_array rhs_les    (grid_dns, fill2);
    
    spade::grid::grid_array prim_dns   (grid_dns, fill1);
    spade::grid::grid_array rhs_dns    (grid_dns, fill2);
    
    spade::viscous_laws::constant_viscosity_t<real_t> visc_law(1.85e-4);
    visc_law.prandtl = 0.72;
    
    spade::fluid_state::perfect_gas_t<real_t> air;
    air.R = 287.15;
    air.gamma = 1.4;
    
    const real_t p0 = 30.0;
    const real_t t0 = 0.1;
    const real_t u0 = 69.54;
    const real_t mu = visc_law.get_visc();
    const real_t rho = p0/(air.R*t0);
    const real_t u_tau = re_tau*mu/(rho*delta);
    const real_t force_term = rho*u_tau*u_tau/delta;
    
    auto ini = [&](const spade::ctrs::array<real_t, 3>& x) -> prim_t
    {
        const real_t shape = 1.0 - pow(x[1]/delta, 4);
        prim_t output;
        output.p() = p0;
        output.T() = t0;
        output.u() = (20.0*u_tau)*shape;
        output.v() = (0.0       )*shape;
        output.w() = (0.0       )*shape;
        return output;
    };
    
    spade::algs::fill_array(prim_les, ini);
    spade::algs::fill_array(prim_dns, ini);


    spade::convective::pressure_diss_lr dscheme(air, 0.025, 0.025);
    spade::convective::totani_lr tscheme(air);
    spade::convective::weno_3    wscheme(air);
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
    
    
    spade::ctrs::array<real_t, 3> dx_comp(grid_dns.get_dx(0), grid_dns.get_dx(1), grid_dns.get_dx(2));
    spade::ctrs::array<real_t, 3> dx_phys
    (
        xc.map(bounds.min(0)+dx_comp[0]) - xc.map(bounds.min(0)),
        yc.map(bounds.min(1)+dx_comp[1]) - yc.map(bounds.min(1)),
        zc.map(bounds.min(2)+dx_comp[2]) - zc.map(bounds.min(2))
    );
    const real_t dx       = spade::utils::min(dx_phys[0], dx_phys[1], dx_phys[2]);
    const real_t umax_ini = spade::algs::transform_reduce(prim_dns, get_u, max_op);
    const real_t dt     = targ_cfl*dx/umax_ini;
    
    
    auto calc_rhs_dns = [&](auto& rhs, auto& q, const auto& t) -> void
    {
        rhs = 0.0;
        grid_dns.exchange_array(q);
        set_channel_noslip(q);
        spade::pde_algs::flux_div(q, rhs, tscheme, visc_scheme, dscheme);
        spade::pde_algs::source_term(q, rhs, [&]()->v5d{return v5d(0,0,force_term,0,0);});
    };
    
    auto calc_rhs_les = [&](auto& rhs, auto& q, const auto& t) -> void
    {
        rhs = 0.0;
        // interp_flux(prim_dns, fluxes_les, tscheme);
        
        //todo: write this as a transform() kernel
        // calc_div(rhs, fluxes_les);
    };
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air);
    
    spade::time_integration::rk2 time_int_dns(prim_dns, rhs_dns, time0, dt, calc_rhs_dns, trans);
    spade::time_integration::rk2 time_int_les(prim_les, rhs_les, time0, dt, calc_rhs_les, trans);
    
    
    std::ofstream myfile("hist.dat");
    for (auto nti: range(0, nt_max))
    {
        int nt = nti;
        const real_t umax   = spade::algs::transform_reduce(prim_dns, get_u, max_op);
        real_t ub, rhob;
        calc_u_bulk(prim_dns, air, ub, rhob);
        const real_t area = bounds.size(0)*bounds.size(2);
        auto conv2 = proto::get_domain_boundary_flux(prim_dns, visc_scheme, 2);
        auto conv3 = proto::get_domain_boundary_flux(prim_dns, visc_scheme, 3);
        conv2 /= area;
        conv3 /= area;
        const real_t tau = 0.5*(spade::utils::abs(conv2.x_momentum()) + spade::utils::abs(conv3.x_momentum()));
        
        if (group.isroot())
        {
            const real_t cfl = umax*dt/dx;
            const int pn = 10;
            print(
                "nt: ", spade::utils::pad_str(nt, pn),
                "cfl:", spade::utils::pad_str(cfl, pn),
                "u+a:", spade::utils::pad_str(umax, pn),
                "ub: ", spade::utils::pad_str(ub, pn),
                "rb: ", spade::utils::pad_str(rhob, pn),
                "tau:", spade::utils::pad_str(tau, pn),
                "dx: ", spade::utils::pad_str(dx, pn),
                "dt: ", spade::utils::pad_str(dt, pn),
                "ftt:", spade::utils::pad_str(20.0*u_tau*time_int_dns.time()/delta, pn)
            );
            myfile << nt << " " << cfl << " " << umax << " " << ub << " " << rhob << " " << tau << " " << dx << " " << dt << std::endl;
            myfile.flush();
        }
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solution...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename_dns = "prims_dns"+nstr;
            std::string filename_les = "prims_les"+nstr;
            spade::io::output_vtk("output", filename_dns, prim_dns);
            spade::io::output_vtk("output", filename_les, prim_les);
            if (group.isroot()) print("Done.");
        }
        if (nt%checkpoint_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename_les = "check_les"+nstr;
            filename_les = "checkpoint/"+filename_les+".bin";
            std::string filename_dns = "check_les"+nstr;
            filename_dns = "checkpoint/"+filename_dns+".bin";
            spade::io::binary_write(filename_dns, prim_dns);
            spade::io::binary_write(filename_les, prim_les);
            if (group.isroot()) print("Done.");
        }
        
        time_int_dns.advance();
        time_int_les.advance();
        
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
