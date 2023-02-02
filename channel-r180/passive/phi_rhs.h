#pragma once

#include "spade.h"
#include "typedef.h"

namespace local
{
    void phi_rhs(auto& rhs, const auto& phi, const auto& prm, const real_t& diffuse, const real_t& rgas, const real_t& phi_tau)
    {
        spade::ctrs::array<real_t, 3> dx;
        dx[0] = prm.get_grid().get_dx(0);
        dx[1] = prm.get_grid().get_dx(1);
        dx[2] = prm.get_grid().get_dx(2);
        spade::algs::fill_array(rhs, [&](const spade::grid::cell_idx_t& ih) -> real_t
        {
            spade::grid::cell_idx_t il = ih; il.i() -= 1;
            spade::grid::cell_idx_t ir = ih; ir.i() += 1;
            spade::grid::cell_idx_t id = ih; id.j() -= 1;
            spade::grid::cell_idx_t iu = ih; iu.j() += 1;
            spade::grid::cell_idx_t ii = ih; ii.k() -= 1;
            spade::grid::cell_idx_t io = ih; io.k() += 1;
            
            real_t rho_h = prm(0,ih)/(rgas*prm(1,ih));
            real_t rho_l = prm(0,il)/(rgas*prm(1,il));
            real_t rho_r = prm(0,ir)/(rgas*prm(1,ir));
            real_t rho_d = prm(0,id)/(rgas*prm(1,id));
            real_t rho_u = prm(0,iu)/(rgas*prm(1,iu));
            real_t rho_i = prm(0,ii)/(rgas*prm(1,ii));
            real_t rho_o = prm(0,io)/(rgas*prm(1,io));
            
            real_t u_h = prm(2,ih);
            real_t v_h = prm(3,ih);
            real_t w_h = prm(4,ih);
            real_t u_l = prm(2,il);
            real_t u_r = prm(2,ir);
            real_t v_d = prm(3,id);
            real_t v_u = prm(3,iu);
            real_t w_i = prm(4,ii);
            real_t w_o = prm(4,io);
            
            real_t phi_h = phi(ih);
            real_t phi_l = phi(il);
            real_t phi_r = phi(ir);
            real_t phi_d = phi(id);
            real_t phi_u = phi(iu);
            real_t phi_i = phi(ii);
            real_t phi_o = phi(io);
            
            real_t rhs_loc = 0.0;
            
            const auto avg = [](real_t a, real_t b) -> real_t {return 0.5*(a+b);};
            
            //convective
            rhs_loc -= ((avg(rho_r,rho_h)*avg(u_r,u_h)*avg(phi_r,phi_h))-(avg(rho_l,rho_h)*avg(u_l,u_h)*avg(phi_l,phi_h)))/dx[0];
            rhs_loc -= ((avg(rho_u,rho_h)*avg(v_u,v_h)*avg(phi_u,phi_h))-(avg(rho_d,rho_h)*avg(v_d,v_h)*avg(phi_d,phi_h)))/dx[1];
            rhs_loc -= ((avg(rho_o,rho_h)*avg(w_o,w_h)*avg(phi_o,phi_h))-(avg(rho_i,rho_h)*avg(w_i,w_h)*avg(phi_i,phi_h)))/dx[2];
            
            //diffusion
            const auto div2 = [](real_t f0, real_t f1, real_t f2, real_t dxloc) -> real_t {return (f0-2*f1+f2)/(dxloc*dxloc);};
            rhs_loc += diffuse*div2(phi_l, phi_h, phi_r, dx[0]);
            rhs_loc += diffuse*div2(phi_d, phi_h, phi_u, dx[1]);
            rhs_loc += diffuse*div2(phi_i, phi_h, phi_o, dx[2]);
            
            //forcing
            rhs_loc += rho_h*phi_tau*phi_tau;
            
            return rhs_loc;
        });
    }
}