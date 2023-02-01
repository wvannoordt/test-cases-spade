#include "spade.h"

void calc_u_bulk(const auto& q, const auto& air, real_t& u_bulk, real_t& rho_bulk)
{
    const auto& grid = q.get_grid();
    const real_t dV = grid.get_dx(0)*grid.get_dx(1)*grid.get_dx(2);
    spade::reduce_ops::reduce_sum<real_t> rsum;
    struct integrate_rho_u_t
    {
        real_t dV;
        const spade::fluid_state::ideal_gas_t<real_t>* gas;
        typedef prim_t arg_type;
        integrate_rho_u_t(const spade::fluid_state::ideal_gas_t<real_t>& gas_in, const real_t& dV_in) {gas = &gas_in; dV = dV_in;}
        real_t operator () (const prim_t& q) const
        {
            real_t rho = q.p()/(gas->get_R(q)*q.T());
            return rho*q.u()*dV;
        }
    } integrate_rho_u(air, dV);
    
    struct integrate_rho_t
    {
        real_t dV;
        const spade::fluid_state::ideal_gas_t<real_t>* gas;
        typedef prim_t arg_type;
        integrate_rho_t(const spade::fluid_state::ideal_gas_t<real_t>& gas_in, const real_t& dV_in) {gas = &gas_in; dV = dV_in;}
        real_t operator () (const prim_t& q) const
        {
            real_t rho = q.p()/(gas->get_R(q)*q.T());
            return rho*dV;
        }
    } integrate_rho(air, dV);
    
    const real_t int_rho_u = spade::algs::transform_reduce(q, integrate_rho_u, rsum);
    const real_t int_rho   = spade::algs::transform_reduce(q, integrate_rho,   rsum);
    const auto bnd = grid.get_bounds();
    const real_t vol  = bnd.size(0)*bnd.size(1)*bnd.size(2);
    rho_bulk = int_rho / vol;
    u_bulk = int_rho_u/int_rho;
}
