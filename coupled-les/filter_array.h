#pragma once

#include <spade.h>

namespace local
{
    static void filter_array(const auto& src, auto& dest)
    {
        const auto& grid_src  = src.get_grid();
        const auto& grid_dest = dest.get_grid();
        
        using src_grid_t  = spade::utils::remove_all<decltype(grid_src)>::type;
        using dest_grid_t = spade::utils::remove_all<decltype(grid_dest)>::type;
        
        using src_arr_t   = spade::utils::remove_all<decltype(src)>::type;
        using dest_arr_t  = spade::utils::remove_all<decltype(dest)>::type;
        
        auto dest_grid_range = grid_dest.get_range(dest_arr_t::centering_type(), spade::grid::exclude_exchanges);
        const int dim = grid_dest.dim();
        spade::ctrs::array<int, dim> filt;
        for (auto d: range(0, dim)) filt[d] = grid_src.get_num_cells(d)/grid_dest.get_num_cells(d);
        
        auto lower_left = [&](const spade::grid::cell_idx_t& i_in) -> spade::grid::cell_idx_t
        {
            spade::grid::cell_idx_t output = i_in;
            output.i() *= filt[0];
            output.j() *= filt[1];
            output.k() *= filt[2];
            return output;
        };
        
        auto di_range = range(0,filt[2])*range(0,filt[1])*range(0,filt[0]);
        
        for (auto i_it: dest_grid_range)
        {
            const spade::grid::cell_idx_t i_dest(i_it[0], i_it[1], i_it[2], i_it[3]);
            const spade::grid::cell_idx_t ll = lower_left(i_dest);
            typename src_arr_t::alias_type result = 0.0;
            for (auto di: di_range)
            {
                const spade::grid::cell_idx_t i_src(ll.i() + di[0], ll.j() + di[1], ll.k() + di[2], i_dest.lb());
                result += src.get_elem(i_src);
            }
            result /= di_range.size();
            dest.set_elem(i_dest, result);
        }
    }
}