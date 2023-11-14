#pragma once

namespace local
{
    template <typename geom_t, typename data_t>
    static void output_surf_vtk(const std::string& filename, const geom_t& geom, const data_t& data)
    {
        std::ofstream outf(filename);
        outf << "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS " << geom.points.size() << " double\n";
        for (const auto& x: geom.points) outf << x[0] << " " << x[1] << " " << x[2] << "\n";
        outf << "POLYGONS " << geom.faces.size() << " " << geom.faces.size()*4 << "\n";
        for (const auto& f: geom.faces) outf << 3 << " " << f[0] << " " << f[1] << " " << f[2] << "\n";
        outf << "CELL_DATA " << geom.faces.size() << "\n";
        using alias_type = typename data_t::value_type;
        for (int v = 0; v < alias_type::size(); ++v)
        {
            const std::string name = alias_type::name(v);
            outf << "SCALARS " << name << " double\n";
            outf << "LOOKUP_TABLE default\n";
            for (std::size_t j = 0; j < geom.faces.size(); ++j)
            {
                outf << data[j][v] << "\n";
            }
        }
    }
}