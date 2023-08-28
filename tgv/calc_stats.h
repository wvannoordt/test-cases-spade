#pragma once

#include <sstream>

#include "typedef.h"

namespace local
{
    template <typename float_t> struct tgv_stats_t
    {
        float_t kinetic_energy, solenoidal_dissipation, compressible_dissipation;
    };
    
    template <typename float_t> struct flow_config_data_t
    {
        float_t rho0, v0, L, Re, mu0;
    };
    
    template <typename float_t> std::ostream& operator << (std::ostream& os, const flow_config_data_t<float_t>& dd)
    {
        os << "rho0 = " << dd.rho0 << "\n";
        os << "v0   = " << dd.v0   << "\n";
        os << "L    = " << dd.L    << "\n";
        os << "Re   = " << dd.Re   << "\n";
        os << "v0   = " << dd.v0;
        return os;
    }
    
    auto calc_stats(auto& qarr, const auto& config, const auto& visc_law, const auto& gas)
    {
        tgv_stats_t<real_t> output;
        output.kinetic_energy           = 0.0;
        output.solenoidal_dissipation   = 0.0;
        output.compressible_dissipation = 0.0;
        
        
        
        const auto& grid    = qarr.get_grid();
        const auto& bounds  = grid.get_bounds();
        const auto  q       = qarr.image();
        std::size_t count   = 0;
        
        for (int lb = 0; lb < grid.get_num_local_blocks(); ++lb)
        {
            const auto dx = grid.get_dx(lb);
            for (int k = 0; k < grid.get_num_cells(2); ++k)
            {
                for (int j = 0; j < grid.get_num_cells(1); ++j)
                {
                    for (int i = 0; i < grid.get_num_cells(0); ++i)
                    {
                        spade::ctrs::array<prim_t, 3> qgrad;
                        for (int var = 0; var < 5; ++var)
                        {
                            qgrad[0][var] = (q(var, i+1, j, k, lb)-q(var, i-1, j, k, lb))/(2.0*dx[0]);
                            qgrad[1][var] = (q(var, i, j+1, k, lb)-q(var, i, j-1, k, lb))/(2.0*dx[1]);
                            qgrad[2][var] = (q(var, i, j, k+1, lb)-q(var, i, j, k-1, lb))/(2.0*dx[2]);
                        }
                        
                        prim_t qloc = q.get_elem(spade::grid::cell_idx_t(i, j, k, lb));
                        cons_t wloc;
                        spade::fluid_state::convert_state(qloc, wloc, gas);
                        
                        const auto mu  = visc_law.get_visc(qloc);
                        const auto rho = wloc.rho();
                        
                        const auto div  = qgrad[0].u()+qgrad[1].v()+qgrad[2].w();
                        v3d vort(qgrad[1].w()-qgrad[2].v(), qgrad[2].u()-qgrad[0].w(), qgrad[0].v()-qgrad[1].u());
                        auto vortmag2 = vort[0]*vort[0] + vort[1]*vort[1] + vort[2]*vort[2];
                        
                        output.kinetic_energy           += 0.5*rho*(qloc.u()*qloc.u() + qloc.v()*qloc.v() + qloc.w()*qloc.w());
                        output.solenoidal_dissipation   += (mu/config.mu0)*(div*div);
                        output.compressible_dissipation += (mu/config.mu0)*(vortmag2);
                        
                        ++count;
                    }
                }
            }
        }
        
        auto count_glob = grid.group().sum(count);
        output.kinetic_energy           = grid.group().sum(output.kinetic_energy);
        output.solenoidal_dissipation   = grid.group().sum(output.solenoidal_dissipation);
        output.compressible_dissipation = grid.group().sum(output.compressible_dissipation);
        
        output.kinetic_energy           /= real_t(count_glob);
        output.solenoidal_dissipation   /= real_t(count_glob);
        output.compressible_dissipation /= real_t(count_glob);
        
        output.kinetic_energy           *= 1.0/(config.rho0*config.v0*config.v0);
        output.solenoidal_dissipation   *= 4.0*config.L*config.L/(3.0*config.v0*config.v0*config.Re);
        output.compressible_dissipation *= config.L*config.L/(config.v0*config.v0*config.Re);
        
        return output;
    }
    
    template <typename data_t> static std::string to_string(const data_t& data, const int precis)
    {
        std::ostringstream out;
        out.precision(precis);
        out << std::fixed << data;
        return out.str();
    } 
}
