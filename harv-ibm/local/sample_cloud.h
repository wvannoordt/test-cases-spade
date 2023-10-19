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
            auto nv = ghosts.closest_normals[idx];
            auto xc = ghosts.closest_points[idx];
            const auto sampldist = sfunc(idx);
            pnt_t ip = xc;
            ip += sampldist*nv;
            ips.push_back(ip);
            
            idx++;
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