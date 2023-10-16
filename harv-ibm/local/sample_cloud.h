#pragma once

#include "spade.h"

namespace local
{
    template <typename ghost_list_t, typename grid_t, typename sample_dist_func_t>
    requires (!(std::integral<sample_dist_func_t> || std::floating_point<sample_dist_func_t>))
    auto compute_ghost_sample_points(const ghost_list_t& ghosts, const grid_t& grid, const sample_dist_func_t& sfunc)
    {
        using pnt_t = typename ghost_list_t::pnt_t;
        spade::device::shared_vector<pnt_t> ips;
        std::size_t idx = 0;
        for (const auto& ig: ghosts.indices)
        {
            const auto xg   = grid.get_comp_coords(ig);
            const auto xb   = ghosts.boundary_points[idx];
            const auto xc   = ghosts.closest_points[idx];
            auto nv         = xc-xg;
            auto dist       = spade::ctrs::array_norm(nv);
            const auto lb   = spade::utils::tag[spade::partition::local](ig.lb());
            const auto dx   = grid.get_dx(lb);
            const auto diag = spade::ctrs::array_norm(dx);
            
            //Figure out this magic number
            const auto tol  = 5e-2;
            if (dist < tol*diag)
            {
                nv  = 0.0*nv;
                nv += ghosts.boundary_normals[idx];
            }
            nv /= spade::ctrs::array_norm(nv);
            const auto sampldist = sfunc(idx);
            // const auto sampldist = 0.0;
            // const pnt_t ip = xg + sampldist*nv;
            const pnt_t ip = xc + sampldist*nv;
            ips.push_back(ip);
            idx++;
            
            if (ip[2] < 0.0)
            {
                std::vector<pnt_t> pts;
                pts.push_back(xg);
                pts.push_back(xb);
                pts.push_back(xc);
                pts.push_back(ip);
                print(xg);
                print(xb);
                print(xc);
                print(ip);
                spade::io::output_vtk("debug/fuck.vtk", pts);
                print(dist, tol*diag);
                print("SADDD");
                std::cin.get();
            }
        }
        
        ips.transfer();
        return ips;
    }
    
    template <typename ghost_list_t, typename grid_t>
    auto compute_ghost_sample_points(const ghost_list_t& ghosts, const grid_t& grid, const typename grid_t::coord_type& sdist)
    {
        return compute_ghost_sample_points(ghosts, grid, [sdist](const auto&){ return sdist; });
    }
}