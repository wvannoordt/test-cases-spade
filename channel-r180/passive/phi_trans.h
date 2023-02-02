#pragma once

#include <spade.h>

namespace local
{
    template <typename state_transformation_t> struct phi_trans_t
    {
        const state_transformation_t& cns_trans;
        phi_trans_t(const state_transformation_t& cns_trans_in) : cns_trans{cns_trans_in} {}
        void transform_forward(auto& q) const
        {
            auto& q_cns = q[0_c];
            auto& q_phi = q[1_c];
            cns_trans.transform_forward(q_cns);
            // spade::algs::transform_inplace(q_phi, [&](const spade::grid::cell_idx_t& ii) -> real_t
            // {
            //     const auto rho = q_cns(0,ii);
            //     return rho*q_phi(ii);
            // });
        }
        
        void transform_inverse(auto& q) const
        {
            auto& q_cns = q[0_c];
            auto& q_phi = q[1_c];
            // spade::algs::transform_inplace(q_phi, [&](const spade::grid::cell_idx_t& ii) -> real_t
            // {
            //     const auto rho = q_cns(0,ii);
            //     return q_phi(ii)/rho;
            // });
            cns_trans.transform_inverse(q_cns);
        }
    };
}