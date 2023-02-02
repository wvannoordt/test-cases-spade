#include "spade.h"
#include "PTL.h"
#include "typedef.h"
#include "calc_u_bulk.h"
#include "calc_boundary_flux.h"
#include "phi_trans.h"
#include "phi_rhs.h"

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
                        q_g.p()   =  q_d.p();
                        q_g.u()   = -q_d.u();
                        q_g.v()   = -q_d.v()*n_d[1]/n_g[1];
                        q_g.w()   = -q_d.w();
                        q_g.T()   =  2.0*t_wall - q_d.T();
                        for (auto n: range(0,5)) prims(n, i_g[0], i_g[1], i_g[2], i_g[3]) = q_g[n];
                    }
                }
            }
            ++idc;
        }
    }
}

void set_zero_wall(auto& phi)
{
    const real_t t_wall = 0.1;
    const auto& grid = phi.get_grid();
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
                        phi(i_g) = -phi(i_d);
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
    
    PTL::PropertyTree input;
    input.Read("input.ptl");
    
    const int    blocks_x    = input["Grid"]["blocks_x"];
    const int    blocks_y    = input["Grid"]["blocks_y"];
    const int    blocks_z    = input["Grid"]["blocks_z"];
    const int    cells_x     = input["Grid"]["cells_x"];
    const int    cells_y     = input["Grid"]["cells_y"];
    const int    cells_z     = input["Grid"]["cells_z"];
    const int    exchange_x  = input["Grid"]["exchange_x"];
    const int    exchange_y  = input["Grid"]["exchange_y"];
    const int    exchange_z  = input["Grid"]["exchange_z"];
    
    const real_t re_tau      = input["Flow"]["re_tau"];
    const real_t mach        = input["Flow"]["mach"];
    const real_t phi_tau     = input["Flow"]["phi_tau"];
    const real_t pr_phi      = input["Flow"]["pr_phi"];
    
    const real_t targ_cfl    = input["Time"]["targ_cfl"];
    const int    nt_max      = input["Time"]["nt_max"];
    const int    nt_skip     = input["Time"]["nt_skip"];
    const int    ck_skip     = input["Time"]["ck_skip"];
    const bool   do_ini      = input["Time"]["do_ini"];
    const int    ini_num     = input["Time"]["ini_num"];
    
    spade::ctrs::array<int, 3> num_blocks    (blocks_x,   blocks_y,   blocks_z);
    spade::ctrs::array<int, 3> cells_in_block(cells_x,    cells_y,    cells_z);
    spade::ctrs::array<int, 3> exchange_cells(exchange_x, exchange_y, exchange_z);
    spade::bound_box_t<real_t, 3> bounds;
    const real_t delta = 1.0;
    bounds.min(0) =  0.0;
    bounds.max(0) =  4.0*spade::consts::pi*delta;
    bounds.min(1) = -delta;
    bounds.max(1) =  delta;
    bounds.min(2) =  0.0;
    bounds.max(2) =  2*spade::consts::pi*delta;
    
    spade::coords::identity<real_t> coords;
    
    std::filesystem::path out_path("checkpoint");
    if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
    
    
    spade::grid::cartesian_grid_t grid(num_blocks, cells_in_block, exchange_cells, bounds, coords, group);
    
    
    prim_t fill1 = 0.0;
    flux_t fill2 = 0.0;
    
    spade::grid::cell_array prim (grid, fill1);
    spade::grid::cell_array rhs (grid, fill2);
    
    real_t fphi = 0.0;
    spade::grid::cell_array phi_q (grid, fphi);
    spade::grid::cell_array phi_r (grid, fphi);
    
    spade::viscous_laws::constant_viscosity_t<real_t> visc_law(1.85e-4, 0.72);
    
    spade::fluid_state::ideal_gas_t<real_t> air;
    air.R = 287.15;
    air.gamma = 1.4;
    
    const real_t p0 = 500.0;
    
    const real_t mu = visc_law.get_visc();
    const real_t du = 3.0;
    const real_t umax_utau_ratio = 20.0;
    const real_t t0 = std::pow(p0*mach*delta/(re_tau*umax_utau_ratio*mu),2)*(air.gamma/air.R);
    const real_t u_tau = mach*std::sqrt(air.gamma*air.R*t0)/umax_utau_ratio;
    const real_t u_chan_max = umax_utau_ratio*u_tau;
    const real_t rho = p0/(air.R*t0);
    const real_t force_term = rho*u_tau*u_tau/delta;
    
    if (group.isroot())
    {
        print("Re_tau:", rho*delta*u_tau/mu);
        print("Mach:  ", umax_utau_ratio*u_tau/(std::sqrt(air.gamma*air.R*t0)));
    }
    
    const int nidx = 8;
    std::vector<real_t> r_amp_1(cells_in_block[0]/nidx);
    std::vector<real_t> r_amp_2(cells_in_block[1]/nidx);
    std::vector<real_t> r_amp_3(cells_in_block[2]/nidx);
    std::vector<real_t> r_amp_4(grid.get_partition().get_num_local_blocks());
    
    for (auto& p: r_amp_1) p = 1.0 - 2.0*spade::utils::unitary_random();
    for (auto& p: r_amp_2) p = 1.0 - 2.0*spade::utils::unitary_random();
    for (auto& p: r_amp_3) p = 1.0 - 2.0*spade::utils::unitary_random();
    for (auto& p: r_amp_4) p = 1.0 - 2.0*spade::utils::unitary_random();
    
    auto ini = [&](const spade::ctrs::array<real_t, 3> x, const spade::grid::cell_idx_t& ii) -> prim_t
    {
        int i  = ii.i ();
        int j  = ii.j ();
        int k  = ii.k ();
        int lb = ii.lb();
        const real_t shape = 1.0 - pow(x[1]/delta, 4);
        const real_t turb  = du*u_tau*sin(10.0*spade::consts::pi*x[1])*cos(12*x[0])*cos(6*x[2]);
        prim_t output;
        output.p()   = p0;
        output.T()   = t0;
        output.u()   = u_chan_max*shape;
        output.v()   = 0.0;
        output.w()   = 0.0;
        
        int eff_i = i/nidx;
        int eff_j = j/nidx;
        int eff_k = k/nidx;
        
        const real_t per = du*u_tau*(r_amp_1[eff_i] + r_amp_2[eff_j] + r_amp_3[eff_k] + r_amp_4[lb]);
        output.u() += per*shape;
        output.v() += per*shape;
        output.w() += per*shape;
        
        return output;
    };
    
    spade::algs::fill_array(prim,  ini);
    spade::algs::fill_array(phi_q, [&](const spade::ctrs::array<real_t, 3>& x)->real_t
    {
        return std::sin(x[0])*std::sin(x[0])*(1.0-x[1]*x[1]/(delta*delta));
    });
    
    int nt_min = 0;
    if (do_ini)
    {
        std::string ini_n_str = spade::utils::zfill(ini_num, 8);
        std::string ini_q_cns = spade::utils::strformat("{}/q_cns_{}.bin", std::string(out_path), ini_n_str);
        std::string ini_q_phi = spade::utils::strformat("{}/q_phi_{}.bin", std::string(out_path), ini_n_str);
        
        if (group.isroot()) print("Reading:", ini_q_cns);
        spade::io::binary_read(ini_q_cns, prim);
        
        if (group.isroot()) print("Reading:", ini_q_phi);
        spade::io::binary_read(ini_q_phi, phi_q);
        if (group.isroot()) print("Init done.");

        grid.exchange_array(prim);
        grid.exchange_array(phi_q);
        
        set_channel_noslip(prim);
        set_zero_wall(phi_q);
        
        nt_min = ini_num;
    }


    spade::convective::pressure_diss_lr dscheme(air, 0.025, 0.025);
    spade::convective::totani_lr tscheme(air);
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
    
    
    
    const real_t dx = spade::utils::min(grid.get_dx(0), grid.get_dx(1), grid.get_dx(2));
    const real_t umax_ini = spade::algs::transform_reduce(prim, get_u, max_op);
    const real_t dt     = targ_cfl*dx/umax_ini;
    
    
    
    auto calc_rhs = [&](auto& rhs_in, auto& q_in, const auto& t) -> void
    {
        auto& rhs_prm_in = rhs_in[0_c];
        auto& rhs_phi_in = rhs_in[1_c];
        
        auto& q_prm_in = q_in[0_c];
        auto& q_phi_in = q_in[1_c];
        
        rhs_prm_in = 0.0;
        rhs_phi_in = 0.0;
        
        grid.exchange_array(q_prm_in);
        grid.exchange_array(q_phi_in);
        set_channel_noslip(q_prm_in);
        set_zero_wall(q_phi_in);
        
        //CNS
        spade::pde_algs::flux_div(q_prm_in, rhs_prm_in, tscheme, visc_scheme);
        spade::pde_algs::source_term(q_prm_in, rhs_prm_in, [&]() -> spade::ctrs::array<real_t, 5> 
        {
            spade::ctrs::array<real_t, 5> srctrm = 0.0;
            srctrm[2] += force_term;
            return srctrm;
        });
        
        //Scalar
        local::phi_rhs(rhs_phi_in, q_phi_in, q_prm_in, visc_law.get_visc()/pr_phi, air.R, phi_tau);
    };
    
    cons_t transform_state;
    spade::fluid_state::state_transform_t cns_trans(transform_state, air);
    local::phi_trans_t trans(cns_trans);
    
    spade::ctrs::arith_tuple prim_v(prim, phi_q);
    spade::ctrs::arith_tuple rhs_v (rhs,  phi_r);
    
    spade::time_integration::time_axis_t axis(time0, dt);
    spade::time_integration::ssprk34_t alg;
    spade::time_integration::integrator_data_t q(prim_v, rhs_v, alg);
    spade::time_integration::integrator_t time_int(axis, alg, q, calc_rhs, trans);
    
    std::ofstream myfile("hist.dat");
    for (auto nti: range(nt_min, nt_max+nt_min))
    {
        int nt = nti;
        const auto& sol = time_int.solution();
        const auto& q_cns = sol[0_c];
        const auto& q_phi = sol[1_c];
        const real_t umax = spade::algs::transform_reduce(q_cns, get_u, max_op);
        real_t ub, rhob;
        calc_u_bulk(q_cns, air, ub, rhob);
        const real_t area = bounds.size(0)*bounds.size(2);
        auto conv2 = proto::get_domain_boundary_flux(q_cns, visc_scheme, 2);
        auto conv3 = proto::get_domain_boundary_flux(q_cns, visc_scheme, 3);
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
                "ftt:", spade::utils::pad_str(umax_utau_ratio*u_tau*time_int.time()/delta, pn)
            );
            myfile << nt << " " << cfl << " " << umax << " " << ub << " " << rhob << " " << tau << " " << dx << " " << dt << std::endl;
            myfile.flush();
        }
        if (nt%nt_skip == 0)
        {
            std::string nstr = spade::utils::zfill(nt, 8);
            
            if (group.isroot()) print("Output solution 1 of 2");
            spade::io::output_vtk("output", spade::utils::strformat("cns_{}", nstr), grid, q_cns);
            
            if (group.isroot()) print("Output solution 2 of 2");
            spade::io::output_vtk("output", spade::utils::strformat("phi_{}", nstr), grid, q_phi);
            
            if (group.isroot()) print("Done.");
        }
        if (nt%ck_skip == 0)
        {
            if (group.isroot()) print("Output checkpoint...");
            std::string nstr = spade::utils::zfill(nt, 8);
            std::string filename = "check"+nstr;
            filename = "checkpoint/"+filename+".bin";
            spade::io::binary_write(filename, q_cns);
            if (group.isroot()) print("Done.");
        }
        time_int.advance();
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
