#pragma once

#include "spade.h"

namespace local
{
    template <typename arr_t, typename ghost_t>
    static void zero_ghost_rhs(arr_t& array, const ghost_t& ghost)
    {
        auto arr_img  = array.image();
        auto ghst_img = ghost.image(array.device());
        const auto range = spade::dispatch::ranges::from_array(ghst_img.indices, array.device());
        auto fill_ghst = _sp_lambda(const std::size_t& idx) mutable
        {
            typename arr_t::alias_type zer = 0.0;
            arr_img.set_elem(ghst_img.indices[idx], zer);
        };
        spade::dispatch::execute(range, fill_ghst);
    }
    
    template <typename arr_t, typename ghost_t, typename xs_t, typename datas_t>
    static void fill_ghost_vals(arr_t& array, const ghost_t& ghost, const xs_t& xs, const datas_t& datas)
    {
        auto geom_img = array.get_grid().image(spade::partition::local, array.device());
        auto arr_img  = array.image();
        auto ghst_img = ghost.image(array.device());
        auto xs_img   = spade::utils::make_vec_image(xs.data(array.device()));
        auto data_img = spade::utils::make_vec_image(datas);
        
        const auto range = spade::dispatch::ranges::from_array(datas, array.device());
        
        using alias_type = typename arr_t::alias_type;
        using real_t     = typename arr_t::fundamental_type;
        auto fill_ghst = _sp_lambda(const std::size_t idx) mutable
        {
            const auto icell = ghst_img.indices[idx];
            
            const alias_type& sampl_value = data_img[idx];
            
            alias_type ghost_value = real_t(0.0);
            ghost_value.p()   = sampl_value.p();
            ghost_value.T()   = sampl_value.T();
            
            
            const auto& nvec = ghst_img.closest_normals[idx];
            
            using vec_t = spade::ctrs::array<real_t, 3>;
            vec_t u = {sampl_value.u(), sampl_value.v(), sampl_value.w()};
            vec_t unorm = spade::ctrs::dot_prod(u, nvec)*nvec;
            vec_t utang = u - unorm;
            
            const auto& sampl_x = xs_img[idx];
            // bool debug = sampl_x[0] < -0.05;
            // if (debug)
            // {
            //     print("debugging", __FILE__, __LINE__);
            //     print(sampl_x);
            //     print(nvec);
            //     print(u);
            //     print(uhat);
            //     print(unorm);
            //     std::cin.get();
            // }
            
            for (int i = 0; i < 3; ++i)
            {
                ghost_value.u(i) += utang[i];
            }
            
            // const auto& sampl_x = xs_img[idx];
            const auto& bndy_x  = ghst_img.closest_points[idx];
            const auto& ghst_x  = geom_img.get_coords(ghst_img.indices[idx]);
            
            //        +
            //        |      |
            //        |      d1
            //  ______|____ _|_
            //        +      d0
            
            const real_t d1 = spade::ctrs::array_norm(sampl_x - bndy_x);
            const real_t d0 = spade::ctrs::array_norm(bndy_x  - ghst_x);
            
            for (int i = 0; i < 3; ++i)
            {
                ghost_value.u(i) += -(d0/d1)*unorm[i];
            }
            
            arr_img.set_elem(icell, ghost_value);
        };
        
        spade::dispatch::execute(range, fill_ghst);
    }
}