#include <chrono>
#include "spade.h"
#include "PTL.h"

typedef double real_t;
typedef spade::ctrs::array<real_t, 3> v3d;
typedef spade::ctrs::array<int,    3> v3i;
typedef spade::ctrs::array<int,    4> v4i;
typedef spade::fluid_state::prim_t<real_t> prim_t;
typedef spade::fluid_state::flux_t<real_t> flux_t;
typedef spade::fluid_state::cons_t<real_t> cons_t;

#include "calc_u_bulk.h"
#include "calc_boundary_flux.h"

#include "io_control.h"
#include "filter_array.h"
#include "filter_flux.h"

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
                const int j = idc*(grid.get_num_cells(1)-1);
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

template <typename state_trans_t> struct tuple_trans_t
{
    const state_trans_t& tr;
    tuple_trans_t(const state_trans_t& tr_in) :tr{tr_in}{}
    void transform_forward(auto& q_in) const
    {
        tr.transform_forward(spade::ctrs::get<0>(q_in));
        tr.transform_forward(spade::ctrs::get<1>(q_in));
    }
    void transform_inverse(auto& q_in) const
    {
        tr.transform_inverse(spade::ctrs::get<0>(q_in));
        tr.transform_inverse(spade::ctrs::get<1>(q_in));
    }
};

int main(int argc, char** argv)
{
    spade::parallel::mpi_t group(&argc, &argv);
    
    local::io_control_t control(argc, argv);
    control.create_dirs();
    PTL::PropertyTree input;
    input.Read(control.get_input_file_name());
    
    const int blocks_x = input["Config"]["blocks_x"];
    const int blocks_y = input["Config"]["blocks_y"];
    const int blocks_z = input["Config"]["blocks_z"];
    
    const int cells_x  = input["Config"]["cells_x"];
    const int cells_y  = input["Config"]["cells_y"];
    const int cells_z  = input["Config"]["cells_z"];
    
    const int fsize_x  = input["Config"]["fsize_x"];
    const int fsize_y  = input["Config"]["fsize_y"];
    const int fsize_z  = input["Config"]["fsize_z"];
    
    const int nexch    = input["Config"]["nexch"];
    
    const real_t cfl_in  = input["Config"]["cfl"];
    const int nt_max     = input["Config"]["nt_max"];
    const int nt_skip    = input["Config"]["nt_skip"];
    const int ck_skip    = input["Config"]["ck_skip"];
    
    const real_t targ_cfl = cfl_in;
    
    spade::ctrs::array<int, 3> num_blocks(blocks_x, blocks_y, blocks_z);
    spade::ctrs::array<int, 3> cells_in_block_dns(cells_x, cells_y, cells_z);
    spade::ctrs::array<int, 3> cells_in_block_les(cells_x/fsize_x, cells_y/fsize_y, cells_z/fsize_z);
    spade::ctrs::array<int, 3> exchange_cells(nexch, nexch, nexch);
    //spade::ctrs::array<int, 3> exchange_cells(8, 8, 8);
    spade::bound_box_t<real_t, 3> bounds;
    const real_t re_tau = 180.0;
    const real_t delta = 1.0;
    bounds.min(0) =  0.0;
    bounds.max(0) =  4.0*spade::consts::pi*delta;
    bounds.min(1) = -delta;
    bounds.max(1) =  delta;
    bounds.min(2) =  0.0;
    bounds.max(2) =  2*spade::consts::pi*delta;
    
    spade::coords::identity<real_t> coords;
    spade::grid::cartesian_grid_t grid_dns(num_blocks, cells_in_block_dns, exchange_cells, bounds, coords, group);
    spade::grid::cartesian_grid_t grid_les(num_blocks, cells_in_block_les, exchange_cells, bounds, coords, group);
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::grid_array prim_dns (grid_dns, fill1);
    spade::grid::grid_array rhs_dns  (grid_dns, fill2);
    
    spade::grid::grid_array prim_les (grid_les, fill1);
    spade::grid::grid_array rhs_les  (grid_les, fill2);
    
    spade::viscous_laws::constant_viscosity_t<real_t> visc_law(1.85e-4, 0.72);
    
    spade::fluid_state::ideal_gas_t<real_t> air;
    air.R = 287.15;
    air.gamma = 1.4;
    
    const real_t p0 = 30.0;
    const real_t t0 = 0.1;
    const real_t u0 = 69.54;
    const real_t mu = visc_law.get_visc();
    const real_t rho = p0/(air.R*t0);
    const real_t u_tau = re_tau*mu/(rho*delta);
    const real_t force_term = rho*u_tau*u_tau/delta;
    const real_t du = 3.0;
    
    const int nidx = 8;
    std::vector<real_t> r_amp_1(cells_in_block_dns[0]/nidx);
    std::vector<real_t> r_amp_2(cells_in_block_dns[1]/nidx);
    std::vector<real_t> r_amp_3(cells_in_block_dns[2]/nidx);
    std::vector<real_t> r_amp_4(grid_dns.get_partition().get_num_local_blocks());
    
    for (auto& p: r_amp_1) p = 1.0 - 2.0*spade::utils::unitary_random();
    for (auto& p: r_amp_2) p = 1.0 - 2.0*spade::utils::unitary_random();
    for (auto& p: r_amp_3) p = 1.0 - 2.0*spade::utils::unitary_random();
    for (auto& p: r_amp_4) p = 1.0 - 2.0*spade::utils::unitary_random();
    
    auto ini = [&](const spade::ctrs::array<real_t, 3> x, const spade::grid::cell_idx_t& ii) -> prim_t
    {
        const int i  = ii.i();
        const int j  = ii.j();
        const int k  = ii.k();
        const int lb = ii.lb();
        const real_t shape = 1.0 - pow(x[1]/delta, 4);
        const real_t turb  = du*u_tau*sin(10.0*spade::consts::pi*x[1])*cos(12*x[0])*cos(6*x[2]);
        prim_t output;
        output.p() = p0;
        output.T() = t0;
        output.u() = (20.0*u_tau + 0.0*turb)*shape;
        output.v() = (0.0        + 0.0*turb)*shape;
        output.w() = (0.0        + 0.0*turb)*shape;
        
        int eff_i = i/nidx;
        int eff_j = j/nidx;
        int eff_k = k/nidx;
        
        const real_t per = du*u_tau*(r_amp_1[eff_i] + r_amp_2[eff_j] + r_amp_3[eff_k] + r_amp_4[lb]);
        output.u() += per*shape;
        output.v() += per*shape;
        output.w() += per*shape;
        
        return output;
    };
    
    spade::algs::fill_array(prim_dns, ini);
    local::filter_array(prim_dns, prim_les);
    
    if (control.is_init())
    {
        const std::string filename0 = control.ck_file_name(0, control.get_init_num());
        const std::string filename1 = control.ck_file_name(1, control.get_init_num());
        if (group.isroot()) print("Reading from", filename0, "and", filename1);
        spade::io::binary_read(filename0, prim_dns);
        spade::io::binary_read(filename1, prim_les);
    }


    spade::convective::pressure_diss_lr dscheme(air, 0.025, 0.025);
    spade::convective::totani_lr tscheme(air);
    spade::convective::weno_3    wscheme(air);
    spade::viscous::visc_lr  visc_scheme(visc_law);
    
    struct get_u_t
    {
        const spade::fluid_state::ideal_gas_t<real_t>* gas;
        typedef prim_t arg_type;
        get_u_t(const spade::fluid_state::ideal_gas_t<real_t>& gas_in) {gas = &gas_in;}
        real_t operator () (const prim_t& q) const
        {
            return sqrt(gas->gamma*gas->R*q.T()) + sqrt(q.u()*q.u() + q.v()*q.v() + q.w()*q.w());
        }
    } get_u(air);
    
    spade::reduce_ops::reduce_max<real_t> max_op;
    real_t time0 = 0.0;
    
    
    const real_t dx = spade::utils::min(grid_dns.get_dx(0), grid_dns.get_dx(1), grid_dns.get_dx(2));
    const real_t umax_ini = spade::algs::transform_reduce(prim_dns, get_u, max_op);
    const real_t dt     = targ_cfl*dx/umax_ini;   
    
    auto calc_rhs = [&](auto& rhs_in, auto& q_in, const auto& t_in) -> void
    {
        auto& rhs_in_dns = spade::ctrs::get<0>(rhs_in);
        auto& rhs_in_les = spade::ctrs::get<1>(rhs_in);
        auto& q_in_dns   = spade::ctrs::get<0>(q_in);
        auto& q_in_les   = spade::ctrs::get<1>(q_in);
        rhs_in_les = 0.0;
        rhs_in_dns = 0.0;
        grid_dns.exchange_array(q_in_dns);
        set_channel_noslip(q_in_dns);
        spade::pde_algs::flux_div(q_in_dns, rhs_in_dns, tscheme, visc_scheme);
        spade::pde_algs::source_term(q_in_dns, rhs_in_dns, [&]() -> spade::ctrs::array<real_t, 5> 
        {
            spade::ctrs::array<real_t, 5> rhs_new = 0.0;
            rhs_new[2] += force_term;
            return rhs_new;
        });
        
        local::filter_array(rhs_in_dns, rhs_in_les);
    };
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t trans(transform_state, air);
    tuple_trans_t trans_vec(trans);
    
    // spade::deprecated::rk2 time_int(prim, rhs, time0, dt, calc_rhs, trans);
    spade::time_integration::time_axis_t axis(time0, dt);
    spade::time_integration::ssprk34_t alg;
    spade::ctrs::arith_tuple prim(prim_dns, prim_les);
    spade::ctrs::arith_tuple rhs (rhs_dns,  rhs_les);
    
    spade::time_integration::integrator_data_t q(prim, rhs, alg);
    spade::time_integration::integrator_t time_int(axis, alg, q, calc_rhs, trans_vec);

    spade::utils::mtimer_t tmr("advance");
    std::ofstream myfile("hist.dat");
    for (auto nti: range(0, nt_max))
    {
        const auto& dns_solution = spade::ctrs::get<0>(time_int.solution());
        const auto& les_solution = spade::ctrs::get<1>(time_int.solution());
        int nt = nti;
        const real_t umax   = spade::algs::transform_reduce(dns_solution, get_u, max_op);
        real_t ub, rhob;
        calc_u_bulk(dns_solution, air, ub, rhob);
        const real_t area = bounds.size(0)*bounds.size(2);
        auto conv2 = proto::get_domain_boundary_flux(dns_solution, visc_scheme, 2);
        auto conv3 = proto::get_domain_boundary_flux(dns_solution, visc_scheme, 3);
        conv2 /= area;
        conv3 /= area;
        const real_t tau = 0.5*(spade::utils::abs(conv2.x_momentum()) + spade::utils::abs(conv3.x_momentum()));
        
        if (nt%nt_skip == 0)
        {
            if (group.isroot()) print("Output solutions...");
            spade::io::output_vtk(control.sl_dir_name(0), control.sl_file_name(nt), dns_solution);
            spade::io::output_vtk(control.sl_dir_name(1), control.sl_file_name(nt), les_solution);
        }
        
        if (nt%ck_skip == 0)
        {
            const std::string ck_fn_0 = control.ck_file_name(0, nt);
            const std::string ck_fn_1 = control.ck_file_name(1, nt);
            if (group.isroot()) print("Output checkpoints", ck_fn_0, "and", ck_fn_1);
            spade::io::binary_write(ck_fn_0, dns_solution);
            spade::io::binary_write(ck_fn_1, les_solution);
        }
        
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
                "ftt:", spade::utils::pad_str(20.0*u_tau*time_int.time()/delta, pn)
            );
            myfile << nt << " " << cfl << " " << umax << " " << ub << " " << rhob << " " << tau << " " << dx << " " << dt << std::endl;
            myfile.flush();
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
