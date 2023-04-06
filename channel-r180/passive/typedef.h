#pragma once
typedef double real_t;
typedef spade::ctrs::array<real_t, 3> v3d;
typedef spade::ctrs::array<int,    3> v3i;
typedef spade::ctrs::array<int,    4> v4i;
typedef spade::fluid_state::prim_t<real_t> prim_t;
typedef spade::fluid_state::flux_t<real_t> flux_t;
typedef spade::fluid_state::cons_t<real_t> cons_t;